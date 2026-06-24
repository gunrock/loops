/**
 * @file reference.hxx
 * @author Loops contributors
 * @brief Host-side reference SpMV and validation helpers.
 * @version 0.1
 * @date 2026-05-06
 *
 * Single source of truth for the "what should the GPU output be?"
 * computation. Used by the example binaries (@c examples/spmv/*.cu )
 * and the unit tests (@c unittests/test_spmv_*.cu ) so both speak the
 * same correctness contract.
 *
 * The reference is intentionally simple-and-slow: row-major scan over
 * a host CSR, single-precision (or whatever @c value_t the matrix
 * uses). It's not a competitive baseline, just a known-good answer.
 *
 * The namespace is deliberately @c loops::reference rather than
 * @c loops::cpu : the example binaries already use a top-level
 * @c cpu namespace in @c examples/spmv/helpers.hxx for their pretty
 * stdout summary, and @c using @c namespace @c loops in those binaries
 * would otherwise make every @c cpu::* call ambiguous.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <loops/container/csr.hxx>
#include <loops/container/vector.hxx>
#include <loops/memory.hxx>
#include <loops/backend/xpu.hxx>

#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>

namespace loops {
namespace reference {

using namespace memory;

/**
 * @brief Host CSR @c y = A @c * @c x scan.
 *
 * Cross-space safe: any input @c csr_t / @c vector_t is pulled to host
 * before the loop, the result lives on host.
 *
 * @tparam index_t  CSR column-index type.
 * @tparam offset_t CSR row-offset type.
 * @tparam value_t  Non-zero / vector-element type.
 * @tparam space    Memory space of the input arguments.
 */
template <typename index_t,
          typename offset_t,
          typename value_t,
          memory_space_t space>
loops::vector_t<value_t, memory_space_t::host> spmv(
    const csr_t<index_t, offset_t, value_t, space>& csr,
    const vector_t<value_t, space>& x) {
  csr_t<index_t, offset_t, value_t, memory_space_t::host> csr_h(csr);
  vector_t<value_t, memory_space_t::host> x_h(x);
  vector_t<value_t, memory_space_t::host> y_h(csr_h.rows, value_t{0});

  for (std::size_t row = 0; row < csr_h.rows; ++row) {
    value_t sum = value_t{0};
    for (auto k = csr_h.offsets[row]; k < csr_h.offsets[row + 1]; ++k) {
      sum += csr_h.values[k] * x_h[csr_h.indices[k]];
    }
    y_h[row] = sum;
  }
  return y_h;
}

/**
 * @brief Default approximate-equality predicate for SpMV outputs.
 *
 * SpMV float32 results carry two distinct error sources that a single
 * tolerance has to cover:
 *
 *   - **Magnitude-scaled round-off.** A row of length @c k accumulates
 *     up to @c k float32 multiply-adds. Two valid summation orders
 *     (CPU reference vs. GPU kernel) can disagree by
 *     @c O(k * eps * sum_of_abs(a_ij * x_j)) , which on the SuiteSparse
 *     Williams set ( @c cant , @c pdb1HYS ) is in the @c 1e-3 - @c 1e-2
 *     absolute range even though the relative error is single-digit ULPs.
 *
 *   - **Catastrophic cancellation.** Circuit-style matrices ( @c scircuit ,
 *     ill-conditioned rows) end up with row sums orders of magnitude
 *     smaller than @c sum_of_abs(a_ij * x_j) ; both the CPU and the GPU
 *     are then in their respective float32 noise floors, and the
 *     difference between them can be O(1e-3) absolute even though
 *     neither answer is "wrong".
 *
 * The mixed scheme:
 *
 *   |a - b| > atol + rtol * |b|
 *
 * with @c atol = 1e-2 absorbs the near-zero cancellation floor, and
 * @c rtol = 1e-3 absorbs the magnitude-scaled round-off. This is a
 * coarse predicate; for definitive bug-vs-round-off classification
 * use @c rigorously_validate_spmv (which compares against a double-
 * precision reference and Wilkinson's per-row error bound).
 *
 * Real algorithmic bugs (dropped nonzeros, race-conditioned
 * accumulators, off-by-one row pointers) typically induce relative
 * errors of 0.1 % or worse, which still trigger the predicate.
 *
 * Tests / callers that need strict bit-equality can pass their own
 * predicate to @c count_errors .
 */
template <typename value_t>
struct default_tolerance {
  /// Absolute tolerance (floor for cancellation-dominated rows).
  static constexpr value_t atol() { return value_t{1e-2}; }
  /// Relative tolerance. Loose enough to absorb the cancellation slack on
  /// stiffness-matrix rows (e.g. SuiteSparse @c cant where row sums are
  /// O(10) but the underlying sum-of-abs is O(1000), so float32 noise
  /// is O(1e-4) relative to the result), but still 1000x tighter than
  /// the kind of mismatch a real algorithmic bug produces (dropped
  /// nonzeros / race conditions are typically @c O(0.5) relative).
  static constexpr value_t rtol() { return value_t{1e-3}; }

  static __host__ __device__ bool ne(value_t a, value_t b) {
    using std::abs;
    return abs(a - b) > atol() + rtol() * abs(b);
  }
};

/**
 * @brief Host CSR @c y = A @c * @c x scan with double-precision
 * accumulation, cast back to @c value_t at the end.
 *
 * Eliminates the reference-side float32 round-off: @c value_t inputs
 * are promoted to @c double for the multiply-add chain and the result
 * is rounded to @c value_t once at the end. The disagreement between
 * this and a pure @c value_t scan is the inherent float32 noise floor
 * for that row -- not a bug.
 *
 * Used by @c rigorously_validate_spmv to establish a "near-gold"
 * baseline to compare GPU outputs against.
 */
template <typename index_t,
          typename offset_t,
          typename value_t,
          memory_space_t space>
loops::vector_t<value_t, memory_space_t::host> spmv_f64(
    const csr_t<index_t, offset_t, value_t, space>& csr,
    const vector_t<value_t, space>& x) {
  csr_t<index_t, offset_t, value_t, memory_space_t::host> csr_h(csr);
  vector_t<value_t, memory_space_t::host> x_h(x);
  vector_t<value_t, memory_space_t::host> y_h(csr_h.rows, value_t{0});

  for (std::size_t row = 0; row < csr_h.rows; ++row) {
    double sum = 0.0;
    for (auto k = csr_h.offsets[row]; k < csr_h.offsets[row + 1]; ++k) {
      sum += static_cast<double>(csr_h.values[k]) *
             static_cast<double>(x_h[csr_h.indices[k]]);
    }
    y_h[row] = static_cast<value_t>(sum);
  }
  return y_h;
}

/**
 * @brief Per-row L1 norm of the products: @c L1[i] = @c sum_j |a_ij * x_j|.
 *
 * Computed in @c double to avoid losing precision on rows where the
 * products span a large dynamic range. Returned as @c value_t .
 *
 * This is the "natural scale" of a row's SpMV: a row sum can be
 * arbitrarily small relative to @c L1 (cancellation) but the round-off
 * budget is fixed by @c L1 .
 */
template <typename index_t,
          typename offset_t,
          typename value_t,
          memory_space_t space>
loops::vector_t<value_t, memory_space_t::host> row_l1_products(
    const csr_t<index_t, offset_t, value_t, space>& csr,
    const vector_t<value_t, space>& x) {
  csr_t<index_t, offset_t, value_t, memory_space_t::host> csr_h(csr);
  vector_t<value_t, memory_space_t::host> x_h(x);
  vector_t<value_t, memory_space_t::host> l1(csr_h.rows, value_t{0});

  for (std::size_t row = 0; row < csr_h.rows; ++row) {
    double sum = 0.0;
    for (auto k = csr_h.offsets[row]; k < csr_h.offsets[row + 1]; ++k) {
      sum += std::abs(static_cast<double>(csr_h.values[k]) *
                      static_cast<double>(x_h[csr_h.indices[k]]));
    }
    l1[row] = static_cast<value_t>(sum);
  }
  return l1;
}

/**
 * @brief Float-type unit roundoff: @c eps_machine / 2 (round-to-nearest).
 */
template <typename value_t>
constexpr value_t unit_roundoff();

template <>
constexpr float unit_roundoff<float>() {
  return 5.96046447753906e-08f;  // 2^-24
}

template <>
constexpr double unit_roundoff<double>() {
  return 1.1102230246251565e-16;  // 2^-53
}

/**
 * @brief Outcome of a rigorous SpMV correctness check.
 *
 * Counts are @e per-row counts; a row may simultaneously contribute to
 * @c naive_mismatches (flagged by @c default_tolerance ) and to
 * @c f32_baseline_overruns (the CPU f32 reference itself exceeds the
 * Wilkinson bound vs the f64 reference -- i.e. the noise floor for
 * this row genuinely is large). The bug-detection signal is whether
 * @c gpu_overruns is significantly larger than @c f32_baseline_overruns .
 */
struct rigorous_report {
  std::size_t total_rows = 0;
  /// Rows flagged by the simple @c default_tolerance predicate.
  std::size_t naive_mismatches = 0;
  /// Rows where the CPU f32 reference itself exceeds Wilkinson(K) vs
  /// the f64 reference -- the inherent float32 noise floor for the row.
  std::size_t f32_baseline_overruns = 0;
  /// Rows where the GPU result exceeds Wilkinson(K) vs the f64 reference.
  /// Healthy kernels should have this @e at most a small multiple of
  /// @c f32_baseline_overruns ; a much larger number is evidence of a bug.
  std::size_t gpu_overruns = 0;
  /// Largest @c |y_gpu - y_f64| seen across all rows.
  double max_gpu_abs_error = 0.0;
  /// Largest @c |y_gpu - y_f64| / max(|y_f64|, 1) seen across all rows.
  double max_gpu_rel_error = 0.0;
  /// Wilkinson safety multiplier used (matches @c rigorously_validate_spmv ).
  double wilkinson_k = 0.0;
};

/**
 * @brief Definitive correctness check for an SpMV kernel output.
 *
 * Compares the GPU result against a double-precision reference, then
 * uses Wilkinson's row-wise floating-point summation bound to decide
 * whether each row's discrepancy is consistent with valid float32
 * round-off or whether it suggests a real algorithmic bug.
 *
 * Per-row bound (Wilkinson @e gamma_n , relaxed by @c K ):
 *
 *   bound[i] = max(atol_floor, K * nnz_i * eps * row_L1[i])
 *
 * with @c K accommodating both summation orders ( @c CPU vs @c GPU ) and
 * the linear-vs-cascading @e gamma_n approximation. @c K = 8 is the
 * default and is satisfied by every kernel in this repository on the
 * SuiteSparse Williams + Hamm benchmark set.
 *
 * @param csr             Source CSR (any memory space).
 * @param x               Input vector (any memory space).
 * @param d_y_gpu         Pointer to GPU output (length @c csr.rows ).
 * @param wilkinson_k     Safety multiplier on the per-row bound.
 *                        4-8 is conservative; pass smaller values to
 *                        tighten the test, larger to absorb additional
 *                        cancellation slack.
 * @param atol_floor      Absolute floor for the per-row bound; covers
 *                        rows where @c row_L1 is tiny but the kernel
 *                        result is naturally near zero.
 * @param verbose         Print per-row diagnostics for any GPU overrun.
 */
template <typename index_t,
          typename offset_t,
          typename value_t,
          memory_space_t space>
rigorous_report rigorously_validate_spmv(
    const csr_t<index_t, offset_t, value_t, space>& csr,
    const vector_t<value_t, space>& x,
    const value_t* d_y_gpu,
    double wilkinson_k = 8.0,
    double atol_floor = 1e-3,
    bool verbose = false) {
  csr_t<index_t, offset_t, value_t, memory_space_t::host> csr_h(csr);
  vector_t<value_t, memory_space_t::host> x_h(x);

  auto y_f32 = spmv(csr_h, x_h);      // value_t accumulation
  auto y_f64 = spmv_f64(csr_h, x_h);  // double accumulation, cast to value_t
  auto l1 =
      row_l1_products(csr_h, x_h);  // double L1 of products, cast to value_t

  thrust::host_vector<value_t> y_gpu(csr_h.rows);
  xpu::memcpy(y_gpu.data(), d_y_gpu, csr_h.rows * sizeof(value_t),
              xpu::memcpy_device_to_host);

  rigorous_report report;
  report.total_rows = csr_h.rows;
  report.wilkinson_k = wilkinson_k;

  const double eps = static_cast<double>(unit_roundoff<value_t>());

  for (std::size_t r = 0; r < csr_h.rows; ++r) {
    const std::size_t nnz_r = csr_h.offsets[r + 1] - csr_h.offsets[r];
    const double bound =
        std::max(atol_floor, wilkinson_k * static_cast<double>(nnz_r) * eps *
                                 static_cast<double>(l1[r]));

    const double ref = static_cast<double>(y_f64[r]);
    const double f32_err = std::abs(static_cast<double>(y_f32[r]) - ref);
    const double gpu_err = std::abs(static_cast<double>(y_gpu[r]) - ref);
    const double scale = std::max(std::abs(ref), 1.0);
    const double gpu_rel = gpu_err / scale;

    if (default_tolerance<value_t>::ne(y_gpu[r], y_f32[r]))
      ++report.naive_mismatches;
    if (f32_err > bound)
      ++report.f32_baseline_overruns;
    if (gpu_err > bound) {
      ++report.gpu_overruns;
      if (verbose) {
        std::printf(
            "GPU_OVERRUN row=%zu nnz=%zu L1=%.6g y_gpu=%.8g y_f64=%.8g "
            "abs_err=%.6g bound=%.6g\n",
            r, nnz_r, static_cast<double>(l1[r]), static_cast<double>(y_gpu[r]),
            ref, gpu_err, bound);
      }
    }

    if (gpu_err > report.max_gpu_abs_error)
      report.max_gpu_abs_error = gpu_err;
    if (gpu_rel > report.max_gpu_rel_error)
      report.max_gpu_rel_error = gpu_rel;
  }

  return report;
}

/**
 * @brief Compare a device output vector against a host reference.
 *
 * Pulls @c d_y to host, applies @c ne element-wise, returns a count of
 * mismatches. Verbose mode prints each mismatch with full-precision
 * values for easy debugging.
 *
 * @tparam value_t Element type.
 * @tparam ne_t    Predicate type returning @c true if the inputs are
 *                 unequal. Defaults to @c default_tolerance<value_t>::ne .
 *
 * @param d_y     Device pointer to candidate output (length n).
 * @param h_ref   Host pointer to reference output (length n).
 * @param n       Vector length.
 * @param ne      Mismatch predicate.
 * @param verbose If true, print per-mismatch diagnostics to @c std::cout .
 * @return Count of mismatches (0 means perfect agreement).
 */
template <typename value_t, typename ne_t>
std::size_t count_errors(const value_t* d_y,
                         const value_t* h_ref,
                         std::size_t n,
                         ne_t ne,
                         bool verbose = false) {
  thrust::host_vector<value_t> y_h(n);
  xpu::memcpy(y_h.data(), d_y, n * sizeof(value_t), xpu::memcpy_device_to_host);

  std::size_t errors = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (ne(y_h[i], h_ref[i])) {
      if (verbose) {
        std::printf("Error[%zu]: %.8g != %.8g\n", i,
                    static_cast<double>(y_h[i]), static_cast<double>(h_ref[i]));
      }
      ++errors;
    }
  }
  return errors;
}

template <typename value_t>
std::size_t count_errors(const value_t* d_y,
                         const value_t* h_ref,
                         std::size_t n,
                         bool verbose = false) {
  return count_errors(
      d_y, h_ref, n,
      [](value_t a, value_t b) { return default_tolerance<value_t>::ne(a, b); },
      verbose);
}

}  // namespace reference
}  // namespace loops
