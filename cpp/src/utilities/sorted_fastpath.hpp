/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdlib>

namespace cudf::detail {

/**
 * @brief Experimental fast paths for sorted inputs, selected by the bitmask in the
 * `CUDF_EXP_SORTED_FASTPATH` environment variable (unset means all disabled).
 */
enum class sorted_fastpath : unsigned {
  SMJ_TYPED_PROBE      = 1u << 0,  ///< sort_merge_join: typed galloping probe for one primitive key
  GB_PRESORTED_VALUES  = 1u << 1,  ///< sort groupby: use presorted values without a gather
  GB_TYPED_OFFSETS     = 1u << 2,  ///< sort groupby: typed group-boundary detection
  GB_OFFSET_REDUCTIONS = 1u << 3,  ///< sort groupby: reduce over group offsets, not labels
  SMJ_UNIQUE_BUILD     = 1u << 4,  ///< sort_merge_join: direct output when build keys are unique
};

inline bool sorted_fastpath_enabled(sorted_fastpath feature)
{
  static unsigned const mask = [] {
    auto const* env = std::getenv("CUDF_EXP_SORTED_FASTPATH");
    return env == nullptr ? 0u : static_cast<unsigned>(std::strtoul(env, nullptr, 0));
  }();
  return (mask & static_cast<unsigned>(feature)) != 0;
}

}  // namespace cudf::detail
