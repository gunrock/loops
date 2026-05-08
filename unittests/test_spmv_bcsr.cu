/**
 * @file test_spmv_bcsr.cu
 * @author Loops contributors
 * @brief End-to-end correctness for the BCSR @c thread_mapped SpMV kernel.
 *
 * Exercises both 2x2 and 3x3 block sizes against the standard battery so
 * matrices that don't divide evenly by the block size also get coverage
 * (the kernel reads padded zeros from the over-aligned input vector).
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/algorithms/spmv/bcsr_thread_mapped.cuh>

#include "test_spmv_battery.hxx"
#include "test_spmv_runner.hxx"

using namespace loops;
using namespace loops::testing;

TEST_CASE("spmv: bcsr_thread_mapped<2,2> (one thread per block-row)",
          "[spmv][bcsr][thread_mapped]") {
  run_battery("bcsr_thread_mapped<2,2>", [](const csr_host_t& csr,
                                            const x_host_t& x) {
    return run_bcsr_spmv<2, 2>(csr, x, [](auto& bcsr_d, auto& x_d, auto& y_d) {
      algorithms::spmv::bcsr_thread_mapped(bcsr_d, x_d, y_d);
    });
  });
}

TEST_CASE("spmv: bcsr_thread_mapped<3,3> (3x3 blocks)",
          "[spmv][bcsr][thread_mapped]") {
  run_battery("bcsr_thread_mapped<3,3>", [](const csr_host_t& csr,
                                            const x_host_t& x) {
    return run_bcsr_spmv<3, 3>(csr, x, [](auto& bcsr_d, auto& x_d, auto& y_d) {
      algorithms::spmv::bcsr_thread_mapped(bcsr_d, x_d, y_d);
    });
  });
}
