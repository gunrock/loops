/**
 * @file hip.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief HIP/ROCm implementation of the @c loops::xpu device-runtime surface.
 * @version 0.1
 * @date 2026-06-24
 *
 * Selected by @c loops/backend/xpu.hxx on a HIP build; @c cuda.hxx is the
 * NVIDIA counterpart. The two mirror each other one call at a time so either
 * backend can be updated on its own.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <cstddef>

#include <hip/hip_runtime.h>

namespace loops {
namespace xpu {

// ----- Types --------------------------------------------------------------
using error_t = hipError_t;
using stream_t = hipStream_t;
using event_t = hipEvent_t;
using device_properties_t = hipDeviceProp_t;
using memcpy_kind_t = hipMemcpyKind;
using device_attribute_t = hipDeviceAttribute_t;

// ----- Constants ----------------------------------------------------------
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

// ----- Device management --------------------------------------------------
inline error_t set_device(int ordinal) {
  return hipSetDevice(ordinal);
}

inline error_t get_device(int* ordinal) {
  return hipGetDevice(ordinal);
}

inline error_t get_device_properties(device_properties_t* props, int ordinal) {
  return hipGetDeviceProperties(props, ordinal);
}

inline error_t device_get_attribute(int* value,
                                    device_attribute_t attr,
                                    int ordinal) {
  return hipDeviceGetAttribute(value, attr, ordinal);
}

inline error_t device_synchronize() {
  return hipDeviceSynchronize();
}

// ----- Memory -------------------------------------------------------------
inline error_t malloc(void** ptr, std::size_t bytes) {
  return hipMalloc(ptr, bytes);
}

inline error_t free(void* ptr) {
  return hipFree(ptr);
}

inline error_t memcpy(void* dst,
                      const void* src,
                      std::size_t bytes,
                      memcpy_kind_t kind) {
  return hipMemcpy(dst, src, bytes, kind);
}

// ----- Streams ------------------------------------------------------------
inline error_t stream_synchronize(stream_t stream = 0) {
  return hipStreamSynchronize(stream);
}

// ----- Events -------------------------------------------------------------
inline error_t event_create(event_t* event) {
  return hipEventCreate(event);
}

inline error_t event_destroy(event_t event) {
  return hipEventDestroy(event);
}

inline error_t event_record(event_t event, stream_t stream = 0) {
  return hipEventRecord(event, stream);
}

inline error_t event_synchronize(event_t event) {
  return hipEventSynchronize(event);
}

inline error_t event_elapsed_time(float* ms, event_t start, event_t stop) {
  return hipEventElapsedTime(ms, start, stop);
}

// ----- Errors -------------------------------------------------------------
inline const char* get_error_string(error_t status) {
  return hipGetErrorString(status);
}

// ----- Occupancy ----------------------------------------------------------
/// Max resident blocks per multiprocessor (CU) for @p kernel; the launch boxes
/// turn this into a one-wave grid.
template <typename kernel_t>
inline error_t occupancy_max_active_blocks_per_multiprocessor(
    int* blocks_per_sm,
    kernel_t kernel,
    int block_size,
    std::size_t dynamic_shared_memory_bytes) {
  return hipOccupancyMaxActiveBlocksPerMultiprocessor(
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
  return hipLaunchCooperativeKernel<func_t>(
      kernel, grid_dim, block_dim, args, shared_memory_bytes, stream);
}

}  // namespace xpu
}  // namespace loops
