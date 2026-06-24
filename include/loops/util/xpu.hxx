/**
 * @file xpu.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Vendor-neutral device runtime surface (CUDA or HIP).
 * @version 0.1
 * @date 2026-06-23
 *
 * Everything in loops talks to the device runtime through @c loops::xpu so the
 * rest of the tree is not CUDA-specific: a CUDA build (nvcc) and a HIP build
 * (hipcc, AMD ROCm) differ only inside this header. The two runtimes are nearly
 * 1:1 -- @c cudaStreamSynchronize maps to @c hipStreamSynchronize ,
 * @c cudaDeviceProp to @c hipDeviceProp_t -- so the shim stays thin: type
 * aliases plus a token-pasted wrapper per call.
 *
 * Backend selection: CMake sets @c LOOPS_BACKEND_HIP (HIP) or
 * @c LOOPS_BACKEND_CUDA (CUDA, the default). Absent an explicit choice we infer
 * HIP when the AMD compiler defines @c __HIP_PLATFORM_AMD__ , otherwise CUDA, so
 * existing nvcc builds are unaffected.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <cstddef>

#if !defined(LOOPS_BACKEND_HIP) && !defined(LOOPS_BACKEND_CUDA)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define LOOPS_BACKEND_HIP
#else
#define LOOPS_BACKEND_CUDA
#endif
#endif

#if defined(LOOPS_BACKEND_HIP)
#include <hip/hip_runtime.h>
#define LOOPS_XPU_(name) hip##name
#else
#include <cuda_runtime.h>
#define LOOPS_XPU_(name) cuda##name
#endif

namespace loops {

/**
 * @namespace xpu
 * The device-runtime backend. Names here read as the vendor-neutral verb
 * (@c xpu::stream_synchronize ) and resolve to CUDA or HIP at compile time.
 */
namespace xpu {

// ----- Types --------------------------------------------------------------
#if defined(LOOPS_BACKEND_HIP)
using error_t = hipError_t;
using stream_t = hipStream_t;
using event_t = hipEvent_t;
using device_properties_t = hipDeviceProp_t;
using memcpy_kind_t = hipMemcpyKind;
using device_attribute_t = hipDeviceAttribute_t;
#else
using error_t = cudaError_t;
using stream_t = cudaStream_t;
using event_t = cudaEvent_t;
using device_properties_t = cudaDeviceProp;
using memcpy_kind_t = cudaMemcpyKind;
using device_attribute_t = cudaDeviceAttr;
#endif

// ----- Constants ----------------------------------------------------------
#if defined(LOOPS_BACKEND_HIP)
inline constexpr error_t success = hipSuccess;

inline constexpr memcpy_kind_t memcpy_host_to_device = hipMemcpyHostToDevice;
inline constexpr memcpy_kind_t memcpy_device_to_host = hipMemcpyDeviceToHost;
inline constexpr memcpy_kind_t memcpy_device_to_device =
    hipMemcpyDeviceToDevice;

inline constexpr device_attribute_t attr_multiprocessor_count =
    hipDeviceAttributeMultiprocessorCount;
inline constexpr device_attribute_t attr_compute_capability_major =
    hipDeviceAttributeComputeCapabilityMajor;
inline constexpr device_attribute_t attr_compute_capability_minor =
    hipDeviceAttributeComputeCapabilityMinor;
inline constexpr device_attribute_t attr_max_grid_dim_x =
    hipDeviceAttributeMaxGridDimX;
#else
inline constexpr error_t success = cudaSuccess;

inline constexpr memcpy_kind_t memcpy_host_to_device = cudaMemcpyHostToDevice;
inline constexpr memcpy_kind_t memcpy_device_to_host = cudaMemcpyDeviceToHost;
inline constexpr memcpy_kind_t memcpy_device_to_device =
    cudaMemcpyDeviceToDevice;

inline constexpr device_attribute_t attr_multiprocessor_count =
    cudaDevAttrMultiProcessorCount;
inline constexpr device_attribute_t attr_compute_capability_major =
    cudaDevAttrComputeCapabilityMajor;
inline constexpr device_attribute_t attr_compute_capability_minor =
    cudaDevAttrComputeCapabilityMinor;
inline constexpr device_attribute_t attr_max_grid_dim_x =
    cudaDevAttrMaxGridDimX;
#endif

// ----- Device management --------------------------------------------------
inline error_t set_device(int ordinal) {
  return LOOPS_XPU_(SetDevice)(ordinal);
}

inline error_t get_device(int* ordinal) {
  return LOOPS_XPU_(GetDevice)(ordinal);
}

inline error_t get_device_properties(device_properties_t* props, int ordinal) {
  return LOOPS_XPU_(GetDeviceProperties)(props, ordinal);
}

inline error_t device_get_attribute(int* value,
                                    device_attribute_t attr,
                                    int ordinal) {
  return LOOPS_XPU_(DeviceGetAttribute)(value, attr, ordinal);
}

inline error_t device_synchronize() {
  return LOOPS_XPU_(DeviceSynchronize)();
}

// ----- Memory -------------------------------------------------------------
inline error_t malloc(void** ptr, std::size_t bytes) {
  return LOOPS_XPU_(Malloc)(ptr, bytes);
}

inline error_t free(void* ptr) {
  return LOOPS_XPU_(Free)(ptr);
}

inline error_t memcpy(void* dst,
                      const void* src,
                      std::size_t bytes,
                      memcpy_kind_t kind) {
  return LOOPS_XPU_(Memcpy)(dst, src, bytes, kind);
}

// ----- Streams ------------------------------------------------------------
inline error_t stream_synchronize(stream_t stream = 0) {
  return LOOPS_XPU_(StreamSynchronize)(stream);
}

// ----- Events -------------------------------------------------------------
inline error_t event_create(event_t* event) {
  return LOOPS_XPU_(EventCreate)(event);
}

inline error_t event_destroy(event_t event) {
  return LOOPS_XPU_(EventDestroy)(event);
}

inline error_t event_record(event_t event, stream_t stream = 0) {
  return LOOPS_XPU_(EventRecord)(event, stream);
}

inline error_t event_synchronize(event_t event) {
  return LOOPS_XPU_(EventSynchronize)(event);
}

inline error_t event_elapsed_time(float* ms, event_t start, event_t stop) {
  return LOOPS_XPU_(EventElapsedTime)(ms, start, stop);
}

// ----- Errors -------------------------------------------------------------
inline const char* get_error_string(error_t status) {
  return LOOPS_XPU_(GetErrorString)(status);
}

// ----- Occupancy ----------------------------------------------------------
/// Max resident blocks per multiprocessor (SM / CU) for @p kernel; the launch
/// boxes turn this into a one-wave grid.
template <typename kernel_t>
inline error_t occupancy_max_active_blocks_per_multiprocessor(
    int* blocks_per_sm,
    kernel_t kernel,
    int block_size,
    std::size_t dynamic_shared_memory_bytes) {
  return LOOPS_XPU_(OccupancyMaxActiveBlocksPerMultiprocessor)(
      blocks_per_sm, kernel, block_size, dynamic_shared_memory_bytes);
}

// ----- Cooperative launch -------------------------------------------------
template <typename func_t>
inline error_t launch_cooperative_kernel(const func_t* kernel,
                                         dim3 grid_dim,
                                         dim3 block_dim,
                                         void** args,
                                         std::size_t shared_memory_bytes,
                                         stream_t stream) {
  return LOOPS_XPU_(LaunchCooperativeKernel)<func_t>(
      kernel, grid_dim, block_dim, args, shared_memory_bytes, stream);
}

}  // namespace xpu
}  // namespace loops
