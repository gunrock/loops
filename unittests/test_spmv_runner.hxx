/**
 * @file test_spmv_runner.hxx
 * @author Loops contributors
 * @brief Format-specific kernel-runner helpers used by the SpMV tests.
 *
 * Each @c run_*_spmv pulls a host CSR into the target format, allocates
 * device-side vectors, calls the kernel via a user-supplied callable, and
 * returns the output as a @c std::vector<float> on the host. The tests
 * stay focused on which kernel they're driving; the bookkeeping lives
 * here.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "test_helpers.hxx"

#include <loops/container/bcsr.hxx>
#include <loops/container/coo.hxx>
#include <loops/container/csc.hxx>
#include <loops/container/csr.hxx>
#include <loops/container/dia.hxx>
#include <loops/container/ell.hxx>
#include <loops/container/vector.hxx>
#include <loops/memory.hxx>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <vector>

namespace loops {
namespace testing {

namespace detail {

inline std::vector<float> pull_y(const vector_t<float>& y_d,
                                 std::size_t rows) {
  std::vector<float> y_h(rows);
  cudaMemcpy(y_h.data(), thrust::raw_pointer_cast(y_d.data()),
             rows * sizeof(float), cudaMemcpyDeviceToHost);
  return y_h;
}

inline vector_t<float> push_x(const x_host_t& x) {
  vector_t<float> x_d(x.size());
  thrust::copy(x.begin(), x.end(), x_d.begin());
  return x_d;
}

}  // namespace detail

/// Run a CSR-input SpMV kernel. @c kernel_fn (csr_d, x_d, y_d) returns void
/// or anything (the return is ignored).
template <typename kernel_fn_t>
std::vector<float> run_csr_spmv(const csr_host_t& csr,
                                const x_host_t& x,
                                kernel_fn_t&& kernel_fn) {
  csr_t<int, int, float> csr_d(csr);
  auto x_d = detail::push_x(x);
  vector_t<float> y_d(csr.rows, 0.0f);
  kernel_fn(csr_d, x_d, y_d);
  return detail::pull_y(y_d, csr.rows);
}

/// Run an ELL-input SpMV kernel.
template <typename kernel_fn_t>
std::vector<float> run_ell_spmv(const csr_host_t& csr,
                                const x_host_t& x,
                                kernel_fn_t&& kernel_fn) {
  ell_t<int, float, memory::memory_space_t::host> ell_h(csr);
  ell_t<int, float, memory::memory_space_t::device> ell_d(ell_h);
  auto x_d = detail::push_x(x);
  vector_t<float> y_d(csr.rows, 0.0f);
  kernel_fn(ell_d, x_d, y_d);
  return detail::pull_y(y_d, csr.rows);
}

/// Run a COO-input SpMV kernel.
template <typename kernel_fn_t>
std::vector<float> run_coo_spmv(const csr_host_t& csr,
                                const x_host_t& x,
                                kernel_fn_t&& kernel_fn) {
  coo_t<int, float, memory::memory_space_t::host> coo_h(csr);
  coo_t<int, float, memory::memory_space_t::device> coo_d(coo_h);
  auto x_d = detail::push_x(x);
  vector_t<float> y_d(csr.rows, 0.0f);
  kernel_fn(coo_d, x_d, y_d);
  return detail::pull_y(y_d, csr.rows);
}

/// Run a CSC-input SpMV kernel.
template <typename kernel_fn_t>
std::vector<float> run_csc_spmv(const csr_host_t& csr,
                                const x_host_t& x,
                                kernel_fn_t&& kernel_fn) {
  csc_t<int, int, float, memory::memory_space_t::host> csc_h(csr);
  csc_t<int, int, float, memory::memory_space_t::device> csc_d(csc_h);
  auto x_d = detail::push_x(x);
  vector_t<float> y_d(csr.rows, 0.0f);
  kernel_fn(csc_d, x_d, y_d);
  return detail::pull_y(y_d, csr.rows);
}

/// Run a DIA-input SpMV kernel.
template <typename kernel_fn_t>
std::vector<float> run_dia_spmv(const csr_host_t& csr,
                                const x_host_t& x,
                                kernel_fn_t&& kernel_fn) {
  dia_t<int, int, float, memory::memory_space_t::host> dia_h(csr);
  dia_t<int, int, float, memory::memory_space_t::device> dia_d(dia_h);
  auto x_d = detail::push_x(x);
  vector_t<float> y_d(csr.rows, 0.0f);
  kernel_fn(dia_d, x_d, y_d);
  return detail::pull_y(y_d, csr.rows);
}

/// Run a BCSR-input SpMV kernel. The block size is a compile-time pair so
/// the BCSR ctor and kernel template instantiation match.
///
/// The kernel reads up to @c bcsr.num_block_cols * C contiguous floats from
/// @c x ; we pad the device-side input vector so out-of-bounds reads are
/// always safe (and zero, since they multiply against zero-padding values
/// in the dense block payload).
template <std::size_t R, std::size_t C, typename kernel_fn_t>
std::vector<float> run_bcsr_spmv(const csr_host_t& csr,
                                 const x_host_t& x,
                                 kernel_fn_t&& kernel_fn) {
  bcsr_t<R, C, int, int, float, memory::memory_space_t::host> bcsr_h(csr);
  bcsr_t<R, C, int, int, float, memory::memory_space_t::device> bcsr_d(bcsr_h);

  const std::size_t padded_cols = bcsr_d.num_block_cols * C;
  vector_t<float> x_d(padded_cols, 0.0f);
  thrust::copy(x.begin(), x.end(), x_d.begin());

  vector_t<float> y_d(bcsr_d.num_block_rows * R, 0.0f);
  kernel_fn(bcsr_d, x_d, y_d);

  // Strip any padding rows before returning to the test harness.
  std::vector<float> y_h(bcsr_d.num_block_rows * R);
  cudaMemcpy(y_h.data(), thrust::raw_pointer_cast(y_d.data()),
             y_h.size() * sizeof(float), cudaMemcpyDeviceToHost);
  y_h.resize(csr.rows);
  return y_h;
}

}  // namespace testing
}  // namespace loops
