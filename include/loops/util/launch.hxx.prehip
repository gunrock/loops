/**
 * @file launch.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Kernel launch C++ related functions.
 * @version 0.1
 * @date 2022-07-10
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once
#include <cstddef>
#include <cuda.h>

namespace loops {
namespace launch {
namespace detail {
inline void for_each_argument_address(void**) {}

template <typename arg_t, typename... args_t>
inline void for_each_argument_address(void** collected_addresses,
                                      arg_t&& arg,
                                      args_t&&... args) {
  collected_addresses[0] = const_cast<void*>(static_cast<const void*>(&arg));
  for_each_argument_address(collected_addresses + 1,
                            ::std::forward<args_t>(args)...);
}
}  // namespace detail

template <typename func_t, typename... args_t>
void cooperative(cudaStream_t stream,
                 const func_t& kernel,
                 std::size_t number_of_blocks,
                 std::size_t threads_per_block,
                 args_t&&... args) {
  constexpr const auto non_zero_num_params =
      sizeof...(args_t) == 0 ? 1 : sizeof...(args_t);
  void* argument_ptrs[non_zero_num_params];
  detail::for_each_argument_address(argument_ptrs,
                                    ::std::forward<args_t>(args)...);
  cudaLaunchCooperativeKernel<func_t>(
      &kernel, number_of_blocks, threads_per_block, argument_ptrs, 0, stream);
}

/**
 * @brief Launch a kernel.
 *
 * @par Overview
 * This function is a reimplementation of `std::apply`, that allows
 * for launching cuda kernels with launch param members of the class
 * and a context argument. It follows the "possible implementation" of
 * `std::apply` in the C++ reference:
 * https://en.cppreference.com/w/cpp/utility/apply.
 *
 * @tparam func_t The type of the kernel function being passed in.
 * @tparam args_tuple_t The type of the tuple of arguments being
 * passed in.
 * @param kernel Kernel function to call.
 * @param args Tuple of arguments to be expanded as the arguments of
 * the kernel function.
 * @param context Reference to the context used to launch the kernel
 * (used for the context's stream).
 * \return void
 */
template <typename func_t, typename... args_t>
void non_cooperative(cudaStream_t stream,
                     const func_t& kernel,
                     dim3 number_of_blocks,
                     dim3 threads_per_block,
                     args_t&&... args) {
  kernel<<<number_of_blocks, threads_per_block, 0, stream>>>(
      std::forward<args_t>(args)...);
}
}  // namespace launch
}  // namespace loops