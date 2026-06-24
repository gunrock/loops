/**
 * @file device.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Device related functions.
 * @version 0.1
 * @date 2022-07-10
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

namespace loops {
namespace device {
typedef int device_id_t;

/**
 * @brief Set the device to use.
 *
 * @param ordinal device id.
 */
void set(device_id_t ordinal) {
  cudaSetDevice(ordinal);
}

/**
 * @brief Get the device id.
 *
 * @return device_id_t device id.
 */
device_id_t get() {
  device_id_t ordinal;
  cudaGetDevice(&ordinal);
  return ordinal;
}

namespace detail {
/**
 * @brief Memoized @c cudaGetDeviceProperties, one query per (process, device).
 *
 * @c cudaGetDeviceProperties is a ~1 ms driver call; querying it inside a
 * timed launch path used to dominate the small-matrix runtime. We cache the
 * @c cudaDeviceProp per ordinal so callers pay the driver cost once. Host-only
 * and assumes the single-threaded launch path the schedules use.
 */
inline const cudaDeviceProp& cached_properties(device_id_t ordinal) {
  constexpr int max_devices = 16;
  static cudaDeviceProp cache[max_devices];
  static bool valid[max_devices] = {};
  if (ordinal >= 0 && ordinal < max_devices) {
    if (!valid[ordinal]) {
      cudaGetDeviceProperties(&cache[ordinal], ordinal);
      valid[ordinal] = true;
    }
    return cache[ordinal];
  }
  static cudaDeviceProp fallback;
  cudaGetDeviceProperties(&fallback, ordinal);
  return fallback;
}
}  // namespace detail

/**
 * @brief Properties of the device.
 *
 * Backed by a process-lifetime cache (see @ref detail::cached_properties), so
 * constructing one is a cheap copy rather than a driver round-trip.
 */
struct properties_t {
  typedef cudaDeviceProp device_properties_t;
  device_properties_t properties;
  device_id_t ordinal;

  properties_t()
      : properties(detail::cached_properties(device::get())),
        ordinal(device::get()) {}

  int multi_processor_count() { return properties.multiProcessorCount; }
};

/**
 * @brief SM (processor) count, memoized.
 *
 * Uses @c cudaDeviceGetAttribute (one value, ~microseconds) rather than the
 * full @c cudaGetDeviceProperties (~1 ms). This matters even with the props
 * cache above: in a single-shot timed launch the first, un-warmed full query
 * would still land inside the timed region, so the launch path must take the
 * cheap attribute road instead.
 */
inline int multi_processor_count(device_id_t ordinal = device::get()) {
  constexpr int max_devices = 16;
  static int cache[max_devices];
  static bool valid[max_devices] = {};
  if (ordinal < 0 || ordinal >= max_devices) {
    int value = 0;
    cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, ordinal);
    return value;
  }
  if (!valid[ordinal]) {
    cudaDeviceGetAttribute(&cache[ordinal], cudaDevAttrMultiProcessorCount,
                           ordinal);
    valid[ordinal] = true;
  }
  return cache[ordinal];
}

/// Compute capability flattened to @c major*10+minor (sm_80 -> 80), memoized.
inline int compute_capability(device_id_t ordinal = device::get()) {
  constexpr int max_devices = 16;
  static int cache[max_devices];
  static bool valid[max_devices] = {};
  if (ordinal >= 0 && ordinal < max_devices && valid[ordinal])
    return cache[ordinal];
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, ordinal);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, ordinal);
  const int value = major * 10 + minor;
  if (ordinal >= 0 && ordinal < max_devices) {
    cache[ordinal] = value;
    valid[ordinal] = true;
  }
  return value;
}

}  // namespace device
}  // namespace loops