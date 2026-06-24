/**
 * @file launch_box.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Per-architecture SpMV launch configuration.
 * @version 0.1
 * @date 2026-06-23
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <cstddef>

#include <loops/util/launch_box.hxx>

namespace loops {
namespace algorithms {
namespace spmv {

/**
 * @brief SpMV launch configuration for the architecture being compiled for.
 *
 * SpMV is bandwidth-bound (~2 flop/nonzero), so 128 threads/block (4 warps)
 * gives enough warps to hide DRAM latency while keeping block occupancy
 * granular. @c items_per_thread sets the merge/work-tile granularity: 8-byte
 * values halve it (each atom moves twice the bytes), and Hopper/Blackwell widen
 * it because their larger L2 keeps @c x resident across more atoms, so a wider
 * tile amortizes the per-thread search. Ampere is the fallback floor.
 *
 * @tparam type_t Value type; selects the 4- vs 8-byte items-per-thread.
 */
template <typename type_t>
using launch_t = launch_box::launch_box_t<
    launch_box::launch_params_t<launch_box::sm_90 | launch_box::sm_100,
                                128,
                                (sizeof(type_t) > 4 ? 4 : 8)>,
    launch_box::launch_params_t<launch_box::sm_70 | launch_box::sm_75 |
                                    launch_box::sm_80 | launch_box::sm_86 |
                                    launch_box::sm_89,
                                128,
                                (sizeof(type_t) > 4 ? 4 : 7)>,
    launch_box::launch_params_t<launch_box::fallback,
                                128,
                                (sizeof(type_t) > 4 ? 4 : 7)>>;

}  // namespace spmv
}  // namespace algorithms
}  // namespace loops
