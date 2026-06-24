/**
 * @file launch_box.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Compile-time, architecture-keyed kernel launch configuration.
 * @version 0.1
 * @date 2026-06-23
 *
 * A @c launch_box_t bundles several @c launch_params_t , each tagged with the
 * SM architecture(s) it targets, and resolves at compile time to the first set
 * whose flags match the architecture being built for (@c LOOPS_TARGET_ARCH ,
 * threaded through from CMake). A trailing @c fallback entry catches anything
 * unmatched; a box with neither a match nor a fallback is ill-formed.
 *
 * The flag space is vendor-neutral: NVIDIA @c sm_* bits occupy the low range
 * and the remainder is reserved for AMD @c gfx ISAs, so a HIP build adds flags
 * rather than a parallel mechanism.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <cstddef>
#include <type_traits>

#include <cuda_runtime.h>

#include <loops/util/device.hxx>

namespace loops {
namespace launch_box {

/**
 * @brief SM architecture flags.
 *
 * Combine with @c | to bind one launch config to several architectures (e.g.
 * @c sm_80 @c | @c sm_86 ). @c fallback matches any architecture and belongs
 * last in a box. The bits above the NVIDIA range are reserved for AMD @c gfx
 * ISAs (gfx90a, gfx942, gfx950, RDNA) so they drop in as further flags.
 */
enum sm_flag_t : unsigned int {
  fallback = 1u << 0,
  sm_70 = 1u << 1,
  sm_72 = 1u << 2,
  sm_75 = 1u << 3,
  sm_80 = 1u << 4,
  sm_86 = 1u << 5,
  sm_89 = 1u << 6,
  sm_90 = 1u << 7,
  sm_100 = 1u << 8,
};

constexpr sm_flag_t operator|(sm_flag_t a, sm_flag_t b) {
  return static_cast<sm_flag_t>(static_cast<unsigned int>(a) |
                                static_cast<unsigned int>(b));
}

constexpr sm_flag_t operator&(sm_flag_t a, sm_flag_t b) {
  return static_cast<sm_flag_t>(static_cast<unsigned int>(a) &
                                static_cast<unsigned int>(b));
}

/// SM flag for a compute-capability number (80 -> @c sm_80 ). An unrecognized
/// value carries no bit, so only a @c fallback entry can match it.
constexpr sm_flag_t flag_of(int compute_capability) {
  switch (compute_capability) {
    case 70:
      return sm_70;
    case 72:
      return sm_72;
    case 75:
      return sm_75;
    case 80:
      return sm_80;
    case 86:
      return sm_86;
    case 89:
      return sm_89;
    case 90:
      return sm_90;
    case 100:
      return sm_100;
    default:
      return static_cast<sm_flag_t>(0u);
  }
}

/// Architecture the kernels are compiled for. CMake pins it from a single-arch
/// preset (release-a100 -> 80); multi-arch / native builds leave it at the
/// Ampere floor and lean on each box's @c fallback entry.
#ifndef LOOPS_TARGET_ARCH
#define LOOPS_TARGET_ARCH 80
#endif

constexpr sm_flag_t target_flag = flag_of(LOOPS_TARGET_ARCH);

/**
 * @brief Launch configuration for one or more SM architectures.
 *
 * @tparam sm_flags_             Architectures this config applies to.
 * @tparam block_size_           Threads per block (1-D).
 * @tparam items_per_thread_     Work items each thread owns (default 1).
 * @tparam shared_memory_bytes_  Static shared memory per block (default 0).
 */
template <sm_flag_t sm_flags_,
          std::size_t block_size_,
          std::size_t items_per_thread_ = 1,
          std::size_t shared_memory_bytes_ = 0>
struct launch_params_t {
  static constexpr sm_flag_t sm_flags = sm_flags_;
  static constexpr std::size_t block_size = block_size_;
  static constexpr std::size_t items_per_thread = items_per_thread_;
  static constexpr std::size_t shared_memory_bytes = shared_memory_bytes_;
};

namespace detail {

template <sm_flag_t>
struct dependent_false : std::false_type {};

/// Sentinel chosen when no entry matches; only instantiated when a box is
/// actually used without a viable match, turning that into a clear diagnostic.
template <sm_flag_t target>
struct no_match_t {
  static_assert(dependent_false<target>::value,
                "launch_box_t: no launch_params match LOOPS_TARGET_ARCH and no "
                "fallback entry was provided.");
};

template <sm_flag_t target, typename... params_t>
struct select {
  using type = no_match_t<target>;
};

template <sm_flag_t target, typename head_t, typename... tail_t>
struct select<target, head_t, tail_t...> {
  static constexpr bool matched =
      static_cast<unsigned int>(head_t::sm_flags & target) != 0u ||
      static_cast<unsigned int>(head_t::sm_flags & fallback) != 0u;
  using type = std::
      conditional_t<matched, head_t, typename select<target, tail_t...>::type>;
};

}  // namespace detail

/**
 * @brief Collection of per-architecture launch configs.
 *
 * Resolves to (inherits) the first @c launch_params_t in the pack whose flags
 * match @c target_flag , so @c launch_box_t<...>::block_size and friends are
 * the selected architecture's values.
 *
 * @code
 * using box_t = launch_box_t<
 *     launch_params_t<sm_90 | sm_100, 128, 8>,
 *     launch_params_t<fallback, 128, 7>>;
 * @endcode
 */
template <typename... params_t>
struct launch_box_t : detail::select<target_flag, params_t...>::type {};

/**
 * @brief Blocks for a single full-occupancy wave on the active device.
 *
 * Returns the occupancy API's max resident blocks per SM for @p kernel, times
 * the SM count. Grid-stride / persistent kernels size their grid with it so
 * one wave covers the device and scales with both occupancy and SM count.
 *
 * @tparam kernel_t Kernel function-pointer type.
 * @param kernel                      Kernel the grid will launch.
 * @param block_size                  Threads per block for the launch.
 * @param dynamic_shared_memory_bytes Dynamic shared memory per block.
 */
template <typename kernel_t>
inline std::size_t occupancy_grid(const kernel_t& kernel,
                                  int block_size,
                                  std::size_t dynamic_shared_memory_bytes = 0) {
  int blocks_per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, block_size, dynamic_shared_memory_bytes);
  if (blocks_per_sm < 1)
    blocks_per_sm = 1;
  return static_cast<std::size_t>(blocks_per_sm) *
         static_cast<std::size_t>(device::multi_processor_count());
}

}  // namespace launch_box
}  // namespace loops
