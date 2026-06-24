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
 * SpMV is bandwidth-bound (~2 flop/nonzero), so the block size is picked to be
 * a few scheduler-width units -- enough resident warps/wavefronts to hide DRAM
 * latency while keeping block occupancy granular -- and @c items_per_thread
 * sets the merge/work-tile granularity. 8-byte values halve the item count
 * (each atom moves twice the bytes); architectures with a larger last-level
 * cache widen it, since @c x stays resident across more atoms and a wider tile
 * amortizes the per-thread diagonal search.
 *
 * @par NVIDIA (warp = 32)
 * 128 threads/block = 4 warps. Hopper/Blackwell (sm_90/sm_100) widen the tile
 * over their larger L2; Ampere-and-below is the floor, and @c fallback catches
 * multi-arch / native CUDA builds.
 *
 * @par AMD CDNA (wavefront = 64, Instinct)
 * 256 threads/block = 4 wavefronts: the wavefront is the 64-wide reduction and
 * scheduling unit (the warp analog), so block sizes and reduction granularity
 * are multiples of 64, not 32. gfx90a (MI210/MI250, CDNA2) has a 16 KB/CU L1,
 * 8 MB L2, and *no* last-level cache, so @c x falls back to HBM and it takes
 * the narrower floor tile; gfx942 (MI300X/MI325X, CDNA3) and gfx950 (MI350X,
 * CDNA4) add a 32 KB L1 and a 256 MB Infinity Cache (MALL) that keeps @c x hot
 * across atoms, so they widen the tile like Hopper. gfx906/gfx908 (MI50/MI100)
 * reuse the CDNA floor. LDS is 64 KB/CU on gfx90a/gfx942 (160 KB on gfx950),
 * all of which cover the per-block merge-path buffer at these sizes. (CDNA2/3/4
 * whitepapers + ROCm gpu-arch-specs.)
 *
 * @par AMD RDNA (native wave32, Radeon)
 * 256 threads/block; RDNA's native wavefront is 32 (the warp analog) and pairs
 * 2 CUs into a WGP, and every RDNA part carries a large Infinity Cache, so it
 * takes the wide tile. gfx1030 RDNA2, gfx1100 RDNA3, gfx1200/gfx1201 RDNA4.
 * Analytically set -- not yet validated on Radeon hardware.
 *
 * @note The CDNA/RDNA tiles are reasoned from the ISA/architecture references
 * (wavefront width, LDS, cache hierarchy), not autotuned; treat them as solid
 * starting points rather than per-matrix optima. Validated on gfx90a (MI210)
 * and gfx942 (MI300X) under ROCm 7.2; gfx950/RDNA are analytical.
 *
 * @tparam type_t Value type; selects the 4- vs 8-byte items-per-thread.
 */
template <typename type_t>
using launch_t = launch_box::launch_box_t<
    // NVIDIA.
    launch_box::launch_params_t<launch_box::sm_90 | launch_box::sm_100,
                                128,
                                (sizeof(type_t) > 4 ? 4 : 8)>,
    launch_box::launch_params_t<launch_box::sm_70 | launch_box::sm_75 |
                                    launch_box::sm_80 | launch_box::sm_86 |
                                    launch_box::sm_89,
                                128,
                                (sizeof(type_t) > 4 ? 4 : 7)>,
    // AMD CDNA: large-LLC parts (gfx942/gfx950) widen the tile.
    launch_box::launch_params_t<launch_box::gfx942 | launch_box::gfx950,
                                256,
                                (sizeof(type_t) > 4 ? 4 : 8)>,
    // AMD CDNA floor (gfx90a + older): 8 MB L2, no last-level cache.
    launch_box::launch_params_t<launch_box::gfx906 | launch_box::gfx908 |
                                    launch_box::gfx90a,
                                256,
                                (sizeof(type_t) > 4 ? 4 : 7)>,
    // AMD RDNA (analytically set; wave32 + Infinity Cache).
    launch_box::launch_params_t<launch_box::gfx1030 | launch_box::gfx1100 |
                                    launch_box::gfx1200 | launch_box::gfx1201,
                                256,
                                (sizeof(type_t) > 4 ? 4 : 8)>,
    launch_box::launch_params_t<launch_box::fallback,
                                128,
                                (sizeof(type_t) > 4 ? 4 : 7)>>;

}  // namespace spmv
}  // namespace algorithms
}  // namespace loops
