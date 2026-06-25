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

#include <loops/backend/xpu.hxx>

namespace loops {
namespace device {
typedef int device_id_t;

/**
 * @brief Set the device to use.
 *
 * @param ordinal device id.
 */
void set(device_id_t ordinal) {
  xpu::set_device(ordinal);
}

/**
 * @brief Get the device id.
 *
 * @return device_id_t device id.
 */
device_id_t get() {
  device_id_t ordinal;
  xpu::get_device(&ordinal);
  return ordinal;
}

namespace detail {
/**
 * @brief Memoized device-properties query, one per (process, device).
 *
 * The properties query is a ~1 ms driver call, heavy enough to dominate
 * small-matrix runtime if hit on a timed launch path. Caching the
 * @c xpu::device_properties_t per ordinal lets callers pay the driver cost
 * once. Host-only and assumes the single-threaded launch path the schedules
 * use.
 */
inline const xpu::device_properties_t& cached_properties(device_id_t ordinal) {
  constexpr int max_devices = 16;
  static xpu::device_properties_t cache[max_devices];
  static bool valid[max_devices] = {};
  if (ordinal >= 0 && ordinal < max_devices) {
    if (!valid[ordinal]) {
      xpu::get_device_properties(&cache[ordinal], ordinal);
      valid[ordinal] = true;
    }
    return cache[ordinal];
  }
  static xpu::device_properties_t fallback;
  xpu::get_device_properties(&fallback, ordinal);
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
  typedef xpu::device_properties_t device_properties_t;
  device_properties_t properties;
  device_id_t ordinal;

  properties_t()
      : properties(detail::cached_properties(device::get())),
        ordinal(device::get()) {}

  int multi_processor_count() { return properties.multiProcessorCount; }
};

/**
 * @brief Multiprocessor (SM on NVIDIA, CU on AMD) count, memoized.
 *
 * Uses @c xpu::device_get_attribute (one value, ~microseconds), keeping the
 * launch path clear of the ~1 ms properties query -- including the first,
 * un-warmed call that a single-shot timed launch cannot hide behind a struct
 * cache.
 */
inline int multi_processor_count(device_id_t ordinal = device::get()) {
  constexpr int max_devices = 16;
  static int cache[max_devices];
  static bool valid[max_devices] = {};
  if (ordinal < 0 || ordinal >= max_devices) {
    int value = 0;
    xpu::device_get_attribute(&value, xpu::attr_multiprocessor_count, ordinal);
    return value;
  }
  if (!valid[ordinal]) {
    xpu::device_get_attribute(&cache[ordinal], xpu::attr_multiprocessor_count,
                              ordinal);
    valid[ordinal] = true;
  }
  return cache[ordinal];
}

/// Compute capability flattened to @c major*10+minor (sm_80 -> 80), memoized.
/// On AMD the runtime reports the CDNA/RDNA major.minor here; kernel tuning
/// keys off the compile-time @c LOOPS_TARGET_* instead, so this stays a
/// diagnostic-only value there.
inline int compute_capability(device_id_t ordinal = device::get()) {
  constexpr int max_devices = 16;
  static int cache[max_devices];
  static bool valid[max_devices] = {};
  if (ordinal >= 0 && ordinal < max_devices && valid[ordinal])
    return cache[ordinal];
  int major = 0, minor = 0;
  xpu::device_get_attribute(&major, xpu::attr_compute_capability_major,
                            ordinal);
  xpu::device_get_attribute(&minor, xpu::attr_compute_capability_minor,
                            ordinal);
  const int value = major * 10 + minor;
  if (ordinal >= 0 && ordinal < max_devices) {
    cache[ordinal] = value;
    valid[ordinal] = true;
  }
  return value;
}

}  // namespace device
}  // namespace loops
