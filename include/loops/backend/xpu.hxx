/**
 * @file xpu.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Vendor-neutral device runtime surface (CUDA or HIP).
 * @version 0.1
 * @date 2026-06-23
 *
 * Everything in loops talks to the device runtime through @c loops::xpu so the
 * rest of the tree is not CUDA-specific. This header only selects a backend;
 * the surface itself lives in @c backend/cuda.hxx (nvcc, the default) and
 * @c backend/hip.hxx (hipcc, AMD ROCm), which mirror each other call for call.
 *
 * Backend selection: CMake sets @c LOOPS_BACKEND_HIP (HIP) or
 * @c LOOPS_BACKEND_CUDA (CUDA, the default). Absent an explicit choice we infer
 * HIP when the AMD compiler defines @c __HIP_PLATFORM_AMD__ , otherwise CUDA,
 * so existing nvcc builds are unaffected.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#if !defined(LOOPS_BACKEND_HIP) && !defined(LOOPS_BACKEND_CUDA)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define LOOPS_BACKEND_HIP
#else
#define LOOPS_BACKEND_CUDA
#endif
#endif

#if defined(LOOPS_BACKEND_HIP)
#include <loops/backend/hip.hxx>
#else
#include <loops/backend/cuda.hxx>
#endif

/**
 * @namespace loops::xpu
 * The device-runtime backend. Names read as the vendor-neutral verb
 * (@c xpu::stream_synchronize ) and resolve to CUDA or HIP at compile time.
 */
