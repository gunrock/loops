/**
 * @file cuda.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief CUDA implementation of the @c loops::xpu device-runtime surface.
 * @version 0.1
 * @date 2026-06-24
 *
 * Selected by @c loops/backend/xpu.hxx on a CUDA build; @c hip.hxx is the
 * AMD/HIP counterpart. The two mirror each other one call at a time so either
 * backend can be updated on its own.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace loops {
namespace xpu {

// ----- Types --------------------------------------------------------------
using error_t = cudaError_t;
using stream_t = cudaStream_t;
using event_t = cudaEvent_t;
using device_properties_t = cudaDeviceProp;
using memcpy_kind_t = cudaMemcpyKind;
using device_attribute_t = cudaDeviceAttr;

// ----- Constants ----------------------------------------------------------
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

// ----- Device management --------------------------------------------------
inline error_t set_device(int ordinal) {
  return cudaSetDevice(ordinal);
}

inline error_t get_device(int* ordinal) {
  return cudaGetDevice(ordinal);
}

inline error_t get_device_properties(device_properties_t* props, int ordinal) {
  return cudaGetDeviceProperties(props, ordinal);
}

inline error_t device_get_attribute(int* value,
                                    device_attribute_t attr,
                                    int ordinal) {
  return cudaDeviceGetAttribute(value, attr, ordinal);
}

inline error_t device_synchronize() {
  return cudaDeviceSynchronize();
}

// ----- Memory -------------------------------------------------------------
inline error_t malloc(void** ptr, std::size_t bytes) {
  return cudaMalloc(ptr, bytes);
}

inline error_t free(void* ptr) {
  return cudaFree(ptr);
}

inline error_t memcpy(void* dst,
                      const void* src,
                      std::size_t bytes,
                      memcpy_kind_t kind) {
  return cudaMemcpy(dst, src, bytes, kind);
}

// ----- Streams ------------------------------------------------------------
inline error_t stream_synchronize(stream_t stream = 0) {
  return cudaStreamSynchronize(stream);
}

// ----- Events -------------------------------------------------------------
inline error_t event_create(event_t* event) {
  return cudaEventCreate(event);
}

inline error_t event_destroy(event_t event) {
  return cudaEventDestroy(event);
}

inline error_t event_record(event_t event, stream_t stream = 0) {
  return cudaEventRecord(event, stream);
}

inline error_t event_synchronize(event_t event) {
  return cudaEventSynchronize(event);
}

inline error_t event_elapsed_time(float* ms, event_t start, event_t stop) {
  return cudaEventElapsedTime(ms, start, stop);
}

// ----- Errors -------------------------------------------------------------
inline const char* get_error_string(error_t status) {
  return cudaGetErrorString(status);
}

// ----- Occupancy ----------------------------------------------------------
/// Max resident blocks per multiprocessor (SM) for @p kernel; the launch boxes
/// turn this into a one-wave grid.
template <typename kernel_t>
inline error_t occupancy_max_active_blocks_per_multiprocessor(
    int* blocks_per_sm,
    kernel_t kernel,
    int block_size,
    std::size_t dynamic_shared_memory_bytes) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
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
  return cudaLaunchCooperativeKernel<func_t>(kernel, grid_dim, block_dim, args,
                                             shared_memory_bytes, stream);
}

}  // namespace xpu
}  // namespace loops
