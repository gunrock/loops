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

#include <thrust/host_vector.h>

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
 * @c rtol = 1e-4 absorbs ~1 ULP of float32 precision on the magnitude
 * of the result. Real algorithmic bugs (dropped nonzeros, race-conditioned
 * accumulators, off-by-one row pointers) typically induce relative errors
 * of 0.1 % or worse, which still trigger the predicate.
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
  cudaMemcpy(y_h.data(), d_y, n * sizeof(value_t), cudaMemcpyDeviceToHost);

  std::size_t errors = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (ne(y_h[i], h_ref[i])) {
      if (verbose) {
        std::printf("Error[%zu]: %.8g != %.8g\n", i,
                    static_cast<double>(y_h[i]),
                    static_cast<double>(h_ref[i]));
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
  return count_errors(d_y, h_ref, n,
                      [](value_t a, value_t b) {
                        return default_tolerance<value_t>::ne(a, b);
                      },
                      verbose);
}

}  // namespace reference
}  // namespace loops
