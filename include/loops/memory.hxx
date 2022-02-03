/**
 * @file memory.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <iostream>
#include <memory>

#include <thrust/device_ptr.h>

namespace loops {
namespace memory {

/**
 * @brief memory space; cuda (device) or host.
 * Can be extended to support uvm and multi-gpu.
 *
 * @todo change this enum to support cudaMemoryType
 * (see ref;  std::underlying_type<cudaMemoryType>::type)
 * instead of some random enums, we can rely
 * on cudaMemoryTypeHost/Device/Unregistered/Managed
 * for this.
 *
 */
enum memory_space_t { device, host, managed };

/**
 * @brief Wrapper around thrust::raw_pointer_cast() to accept .data() or raw
 * pointer and return a raw pointer. Useful when we would like to return a raw
 * pointer of either a thrust device vector or a host vector. Because thrust
 * device vector's raw pointer is accessed by `.data().get()`, whereas thrust
 * host vector's raw pointer is simply `data()`. So, when calling these
 * functions on `.data()`, it can cast either a host or device vector.
 *
 * @tparam type_t
 * @param pointer
 * @return type_t*
 */
template <typename type_t>
inline type_t* raw_pointer_cast(thrust::device_ptr<type_t> pointer) {
  return thrust::raw_pointer_cast(pointer);
}

/**
 * @brief Wrapper around thrust::raw_pointer_cast() to accept .data() or raw
 * pointer and return a raw pointer. Useful when we would like to return a raw
 * pointer of either a thrust device vector or a host vector. Because thrust
 * device vector's raw pointer is accessed by `.data().get()`, whereas thrust
 * host vector's raw pointer is simply `data()`. So, when calling these
 * functions on `.data()`, it can cast either a host or device vector.
 *
 * @tparam type_t
 * @param pointer
 * @return type_t*
 */
template <typename type_t>
__host__ __device__ inline type_t* raw_pointer_cast(type_t* pointer) {
  return thrust::raw_pointer_cast(pointer);
}

}  // namespace memory
}  // namespace loops