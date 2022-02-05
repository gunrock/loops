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
                 const func_t& f,
                 std::size_t threads_per_block,
                 std::size_t number_of_blocks,
                 args_t&&... args) {
  constexpr const auto non_zero_num_params =
      sizeof...(args_t) == 0 ? 1 : sizeof...(args_t);
  void* argument_ptrs[non_zero_num_params];
  detail::for_each_argument_address(argument_ptrs,
                                    ::std::forward<args_t>(args)...);
  cudaLaunchCooperativeKernel<func_t>(&f, number_of_blocks, threads_per_block,
                                      argument_ptrs, 0, stream);
}
}  // namespace launch
}  // namespace loops