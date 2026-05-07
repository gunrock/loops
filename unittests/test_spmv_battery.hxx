/**
 * @file test_spmv_battery.hxx
 * @author Loops contributors
 * @brief Shared battery of synthetic test cases for every SpMV kernel.
 *
 * Each kernel test in @c test_spmv_*.cu pulls in this header and
 * invokes @c run_battery with a small lambda that maps a host CSR +
 * input vector into the format the kernel under test expects (e.g.,
 * convert to ELL, push to device, call @c ell_thread_mapped ). The
 * battery enumerates a fixed set of matrices designed to surface
 * different correctness bugs:
 *
 *   - identity                  : sanity check, no summation order risk
 *   - banded(1, 1)              : tridiagonal, simplest non-trivial case
 *   - banded(3, 4)              : asymmetric band, exercises pitch math
 *   - banded(0, 0)              : pure diagonal
 *   - block_diag(4, 2) (R = C = 2) : 4 disjoint 2x2 blocks
 *   - block_diag(3, 3) (R = C = 3) : 3 disjoint 3x3 blocks
 *   - skewed(64, 16, 2)         : one heavy row + many light, load-balance
 *   - empty_rows(20, 12, 0.3, 4): every-4th row empty, density 30%
 *   - random(50, 50, 0.05)      : general-purpose smoke
 *   - single_row                : edge-case 1xN
 *   - empty_matrix              : 0x0 (kernels should not OOB)
 *
 * Kernels that can't faithfully represent some shape (e.g., DIA on
 * highly-unstructured matrices, BCSR with R != block_size) can
 * filter via the @c options argument.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <catch2/catch_test_macros.hpp>

#include "test_helpers.hxx"

#include <functional>
#include <string>
#include <vector>

namespace loops {
namespace testing {

/// A single labeled test case in the battery.
struct battery_case {
  std::string name;
  csr_host_t csr;
};

inline std::vector<battery_case> standard_battery() {
  std::vector<battery_case> b;
  b.push_back({"identity-16", make_identity_csr(16)});
  b.push_back({"banded(0,0)/diag-16", make_banded_csr(16, 0, 0)});
  b.push_back({"banded(1,1)/tridiag-16", make_banded_csr(16, 1, 1)});
  b.push_back({"banded(3,4)/asym-32", make_banded_csr(32, 3, 4)});
  b.push_back({"block_diag(4,2)", make_block_diag_csr(4, 2)});
  b.push_back({"block_diag(3,3)", make_block_diag_csr(3, 3)});
  b.push_back({"skewed(20,50,h=16,l=2)",
               make_skewed_csr(20, 50, 16, 2)});
  b.push_back({"empty_rows(20,12,0.3,every-4)",
               make_empty_row_csr(20, 12, 0.3f, /*empty_every=*/4)});
  b.push_back({"random(50,50,0.05)", make_random_csr(50, 50, 0.05f)});
  return b;
}

/**
 * @brief Execute @c kernel_fn on every matrix in the battery and fail the
 * current Catch2 SECTION on any mismatch.
 *
 * @c kernel_fn receives the host CSR and the host input vector and must
 * return a host @c std::vector<float> of length @c csr.rows containing the
 * kernel's output. The reference SpMV is computed automatically; mismatches
 * are counted via @c nearly_equal and reported with the matrix label so
 * the failing case is obvious.
 */
template <typename kernel_fn_t>
void run_battery(const std::string& kernel_label, kernel_fn_t&& kernel_fn) {
  for (const auto& tc : standard_battery()) {
    INFO(kernel_label << " on " << tc.name);

    auto x = make_input_vector(tc.csr);
    auto y_ref = reference_spmv(tc.csr, x);
    auto y = kernel_fn(tc.csr, x);

    REQUIRE(y.size() == y_ref.size());
    std::size_t errors = 0;
    for (std::size_t i = 0; i < y_ref.size(); ++i) {
      if (!nearly_equal(y[i], y_ref[i])) ++errors;
    }
    CHECK(errors == 0);
  }
}

}  // namespace testing
}  // namespace loops
