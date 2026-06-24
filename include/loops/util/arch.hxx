/**
 * @file arch.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Vendor-neutral device architecture spec: identity + analytically
 * chosen SpMV launch defaults, and an occupancy-aware grid helper.
 * @version 0.1
 * @date 2026-06-23
 *
 * SpMV is bandwidth-bound (~2 flop/nonzero), so every default below is picked
 * to keep bytes in flight rather than to feed the ALUs. The spec is keyed on a
 * @c vendor + @c generation pair so a single table describes an NVIDIA SM or
 * (in the future) an AMD CU; only NVIDIA is wired today and the AMD branch is
 * called out explicitly.
 *
 * Two halves, deliberately split by what the hardware can tell us and when:
 *   - compile-time (@ref target_spmv_traits): block size and items-per-thread
 *     are kernel template parameters and must be baked into the binary. Single
 *     -arch presets pin them via @c LOOPS_TARGET_ARCH ; otherwise we use the
 *     Ampere baseline.
 *   - run-time (@ref current_id, @ref occupancy_grid): grid sizing scales to
 *     the actual SM count + the compiled kernel's occupancy, so a fat binary
 *     still fills whatever device it lands on.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#include <loops/util/device.hxx>

namespace loops {
namespace arch {

/**
 * @brief Hardware vendor.
 *
 * Vendor-neutral so the spec can describe NVIDIA today and AMD later; the
 * @c amd value is reserved and currently unreachable (see @ref spmv_traits).
 */
enum class vendor_t { unknown, nvidia, amd };

/**
 * @brief Vendor-neutral architecture identity.
 *
 * @c generation flattens the vendor's own ISA number into an int so one
 * switch dispatches tuning:
 *   - NVIDIA: compute capability @c major*10+minor (sm_70 -> 70, sm_90 -> 90).
 *   - AMD (future): gfx ISA, e.g. gfx942 -> 942, gfx90a -> 90a's numeric core.
 */
struct id_t {
  vendor_t vendor;
  int generation;
};

/**
 * @brief Analytically chosen SpMV launch defaults for one architecture.
 *
 * Fields are vendor-neutral: an NVIDIA "SM" and an AMD "CU" are both
 * "processors", a warp and a wavefront are both "lanes".
 */
struct spmv_traits_t {
  /// Threads per block. 128 (4 warps) is the bandwidth-bound sweet spot across
  /// every NVIDIA generation here: enough warps to hide DRAM latency, small
  /// enough to keep block occupancy granular and the launch tail cheap.
  int block_size;
  /// Nonzeros per thread per merge/work tile for 4-byte values; sets the
  /// granularity of the work-oriented and merge-path schedules. Larger values
  /// amortize the per-thread diagonal search, bounded by register/L1 reuse.
  int items_per_thread_f32;
  /// Same for 8-byte values: halved, since each atom moves twice the bytes and
  /// the kernel saturates on bandwidth, not instruction issue.
  int items_per_thread_f64;
  /// Resident blocks per processor used as the analytical fallback for a full
  /// -occupancy wave (grid = this * processor_count). All NVIDIA parts here cap
  /// at 2048 threads/SM, so 128-thread blocks top out at 16; @ref
  /// occupancy_grid refines this against the actual compiled kernel.
  int blocks_per_processor;
  /// Last-level (L2) cache bytes. SpMV's reuse of @c x lives here; a larger L2
  /// makes the extra read traffic of a segmented reduction effectively free.
  long long l2_bytes;
  /// Prefer one-write-per-row segmented reduction (lane shuffle / CUB) over
  /// per-atom atomics. The shuffle path is broadly cheaper for SpMV; the field
  /// exists so vendors/archs with comparatively strong atomics can flip it.
  bool prefer_segmented_reduction;
  /// Asynchronous global->shared copy (NVIDIA cp.async, sm_80+).
  bool has_async_copy;
  /// Bulk tensor-memory-accelerator copies (NVIDIA TMA, sm_90+).
  bool has_tma;
};

/**
 * @brief SpMV defaults for a given architecture identity.
 *
 * Unknown or newer-than-known archs fall through to the closest documented
 * generation at or below @c id.generation (Ampere as the modern baseline), so
 * a brand-new chip launches sensibly instead of with Volta-era constants.
 */
constexpr spmv_traits_t spmv_traits(id_t id) {
  // AMD slot: when wired, branch on `id.vendor == vendor_t::amd` here and
  // return CU-tuned traits (gfx90a/gfx942/gfx950, RDNA). Until then every
  // identity maps onto the NVIDIA table below.
  const int g = id.generation;

  // Blackwell (sm_100, B100/B200): Hopper-class L2 (~50 MB), TMA + cp.async.
  if (g >= 100)
    return spmv_traits_t{128, 8, 4, 16, 50ll * 1024 * 1024, true, true, true};

  // Hopper (sm_90, H100): 50 MB L2, big enough that x of most matrices is
  // L2-resident; TMA + cp.async available. Wider items amortize the search.
  if (g >= 90)
    return spmv_traits_t{128, 8, 4, 16, 50ll * 1024 * 1024, true, true, true};

  // Ampere (sm_80/86, A100): 40 MB L2, cp.async but no TMA. Modern baseline.
  if (g >= 80)
    return spmv_traits_t{128, 7, 4, 16, 40ll * 1024 * 1024, true, true, false};

  // Volta/Turing (sm_70/75, V100): only 6 MB L2, no cp.async, no TMA. Smaller
  // L2 means less x reuse, so keep the merge granularity modest.
  return spmv_traits_t{128, 7, 4, 16, 6ll * 1024 * 1024, true, false, false};
}

/// Compile-time target arch baked into kernel template params. Single-arch
/// presets define it (release-a100 -> 80, release-h100 -> 90); multi-arch and
/// native builds fall back to the Ampere baseline while run-time grid sizing
/// adapts to the actual device.
#ifndef LOOPS_TARGET_ARCH
#define LOOPS_TARGET_ARCH 80
#endif

/// Compile-time identity used to fix the binary's launch template params.
constexpr id_t target_id() {
  return id_t{vendor_t::nvidia, LOOPS_TARGET_ARCH};
}

/// Compile-time SpMV traits for @ref target_id; usable as template arguments.
constexpr spmv_traits_t target_spmv_traits() {
  return spmv_traits(target_id());
}

/// Items-per-thread for a value type, resolved at compile time from the target
/// traits: 8-byte values get the halved (bandwidth-aware) granularity.
template <typename type_t>
constexpr std::size_t target_items_per_thread() {
  return (sizeof(type_t) > 4) ? static_cast<std::size_t>(
                                    target_spmv_traits().items_per_thread_f64)
                              : static_cast<std::size_t>(
                                    target_spmv_traits().items_per_thread_f32);
}

/**
 * @brief Run-time identity of the active device.
 *
 * Reads the cached @c cudaDeviceProp (memoized in device.hxx), so this never
 * hits the ~1 ms driver query on a timed path.
 */
inline id_t current_id() {
  return id_t{vendor_t::nvidia, device::compute_capability()};
}

/// Active device's processor (SM) count (cheap, memoized attribute query).
inline int processor_count() {
  return device::multi_processor_count();
}

/**
 * @brief Blocks for a single full-occupancy wave scaled to the device.
 *
 * Returns (max resident blocks per SM, as the occupancy API reports for *this*
 * compiled kernel) * SM count. Grid-stride / persistent kernels (e.g.
 * work_oriented) use it to fill the device and scale with arch, replacing the
 * old fixed @c 2*SM under-subscription.
 *
 * @tparam kernel_t Kernel function-pointer type.
 * @param kernel              Kernel the grid will launch.
 * @param block_size          Threads per block for the launch.
 * @param dynamic_smem_bytes  Dynamic shared memory per block (default 0).
 */
template <typename kernel_t>
inline std::size_t occupancy_grid(const kernel_t& kernel,
                                  int block_size,
                                  std::size_t dynamic_smem_bytes = 0) {
  int blocks_per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel,
                                                block_size, dynamic_smem_bytes);
  if (blocks_per_sm < 1)
    blocks_per_sm = 1;
  return static_cast<std::size_t>(blocks_per_sm) *
         static_cast<std::size_t>(processor_count());
}

}  // namespace arch
}  // namespace loops
