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

/**
 * @brief Properties of the device.
 *
 */
struct properties_t {
  typedef cudaDeviceProp device_properties_t;
  device_properties_t properties;
  device_id_t ordinal;

  properties_t() : ordinal(device::get()) {
    cudaGetDeviceProperties(&properties, ordinal);
  }

  int multi_processor_count() { return properties.multiProcessorCount; }
};

}  // namespace device
}  // namespace loops