/**
 * @file test_spmv_csc.cu
 * @author Loops contributors
 * @brief End-to-end correctness for the CSC @c thread_mapped SpMV kernel.
 *
 * CSC inverts the row/column roles: one thread per column, accumulates
 * into @c y via @c atomicAdd at the row index. This pins the offsets-size
 * bug fix in the CSC ctor and the column-stationary kernel iteration.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/algorithms/spmv/csc_thread_mapped.cuh>

#include "test_spmv_battery.hxx"
#include "test_spmv_runner.hxx"

using namespace loops;
using namespace loops::testing;

TEST_CASE("spmv: csc_thread_mapped (one thread per column, atomicAdd)",
          "[spmv][csc][thread_mapped]") {
  run_battery(
      "csc_thread_mapped", [](const csr_host_t& csr, const x_host_t& x) {
        return run_csc_spmv(csr, x, [](auto& csc_d, auto& x_d, auto& y_d) {
          algorithms::spmv::csc_thread_mapped(csc_d, x_d, y_d);
        });
      });
}
