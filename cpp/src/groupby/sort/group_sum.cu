/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/sort/group_single_pass_reduction_util.cuh"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform_reduce.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

/// Largest group a single thread reduces before falling back to the label-based reduction.
constexpr size_type max_offset_reduction_group_size = 1024;

template <typename Source, typename Result>
CUDF_KERNEL void sum_by_offsets_kernel(Source const* values,
                                       size_type const* offsets,
                                       size_type num_groups,
                                       Result* out)
{
  auto const group = cudf::detail::grid_1d::global_thread_id();
  if (group >= num_groups) { return; }
  auto acc       = Result{0};
  auto const end = offsets[group + 1];
  for (auto i = offsets[group]; i < end; ++i) {
    acc += static_cast<Result>(values[i]);
  }
  out[group] = acc;
}

template <typename T>
constexpr bool sums_by_offsets =
  (cudf::is_numeric<T>() && !cudf::is_boolean<T>()) || cudf::is_fixed_point<T>();

struct sum_by_offsets_dispatcher {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& values,
                                     device_span<size_type const> group_offsets,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(sums_by_offsets<T>)
  {
    // A fixed-point SUM keeps the input type and scale, so its representation is summed directly.
    using Source = cudf::device_storage_type_t<T>;
    using Result = std::conditional_t<cudf::is_fixed_point<T>(),
                                      Source,
                                      cudf::detail::target_type_t<T, aggregation::SUM>>;
    auto const result_type =
      cudf::is_fixed_point<T>() ? values.type() : data_type{type_to_id<Result>()};
    auto const num_groups = static_cast<size_type>(group_offsets.size()) - 1;
    auto const largest    = thrust::transform_reduce(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<size_type>{0},
      cuda::counting_iterator<size_type>{num_groups},
      [offsets = group_offsets.data()] __device__(size_type g) -> size_type {
        return offsets[g + 1] - offsets[g];
      },
      size_type{0},
      cuda::maximum<size_type>{});
    if (largest > max_offset_reduction_group_size) { return nullptr; }

    auto result =
      make_fixed_width_column(result_type, num_groups, mask_state::UNALLOCATED, stream, mr);
    if (num_groups == 0) { return result; }
    auto const grid = cudf::detail::grid_1d{num_groups, 256};
    sum_by_offsets_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
      values.data<Source>(),
      group_offsets.data(),
      num_groups,
      result->mutable_view().template data<Result>());
    CUDF_CHECK_CUDA(stream.get());
    return result;
  }

  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
    requires(!sums_by_offsets<T>)
  {
    return nullptr;
  }
};

}  // namespace

std::unique_ptr<column> group_sum_by_offsets(column_view const& values,
                                             device_span<size_type const> group_offsets,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  if (values.has_nulls() || cudf::is_dictionary(values.type())) { return nullptr; }
  return type_dispatcher(
    values.type(), sum_by_offsets_dispatcher{}, values, group_offsets, stream, mr);
}

std::unique_ptr<column> group_sum(column_view const& values,
                                  size_type num_groups,
                                  cudf::device_span<size_type const> group_labels,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  auto values_type = cudf::is_dictionary(values.type())
                       ? dictionary_column_view(values).keys().type()
                       : values.type();
  return type_dispatcher(values_type,
                         group_reduction_dispatcher<aggregation::SUM>{},
                         values,
                         num_groups,
                         group_labels,
                         stream,
                         mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
