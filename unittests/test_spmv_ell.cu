/**
 * @file test_spmv_ell.cu
 * @author Loops contributors
 * @brief End-to-end correctness for every ELL-input SpMV kernel.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/algorithms/spmv/ell_merge_path.cuh>
#include <loops/algorithms/spmv/ell_thread_mapped.cuh>

#include "test_spmv_battery.hxx"
#include "test_spmv_runner.hxx"

using namespace loops;
using namespace loops::testing;

TEST_CASE("spmv: ell_thread_mapped (one thread per ELL row)",
          "[spmv][ell][thread_mapped]") {
  run_battery(
      "ell_thread_mapped", [](const csr_host_t& csr, const x_host_t& x) {
        return run_ell_spmv(csr, x, [](auto& ell_d, auto& x_d, auto& y_d) {
          algorithms::spmv::ell_thread_mapped(ell_d, x_d, y_d);
        });
      });
}

TEST_CASE("spmv: ell_merge_path (merge-path on ELL)",
          "[spmv][ell][merge_path]") {
  run_battery("ell_merge_path", [](const csr_host_t& csr, const x_host_t& x) {
    return run_ell_spmv(csr, x, [](auto& ell_d, auto& x_d, auto& y_d) {
      algorithms::spmv::ell_merge_path(ell_d, x_d, y_d);
    });
  });
}
