/**
 * @file test_spmv_dia.cu
 * @author Loops contributors
 * @brief End-to-end correctness for the DIA @c thread_mapped SpMV kernel.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/algorithms/spmv/dia_thread_mapped.cuh>

#include "test_spmv_battery.hxx"
#include "test_spmv_runner.hxx"

using namespace loops;
using namespace loops::testing;

TEST_CASE("spmv: dia_thread_mapped (one thread per row, no atomics)",
          "[spmv][dia][thread_mapped]") {
  run_battery("dia_thread_mapped",
              [](const csr_host_t& csr, const x_host_t& x) {
                return run_dia_spmv(
                    csr, x, [](auto& dia_d, auto& x_d, auto& y_d) {
                      algorithms::spmv::dia_thread_mapped(dia_d, x_d, y_d);
                    });
              });
}
