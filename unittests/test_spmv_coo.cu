/**
 * @file test_spmv_coo.cu
 * @author Loops contributors
 * @brief End-to-end correctness for the COO @c thread_mapped SpMV kernel.
 *
 * COO splits one nonzero per thread and accumulates into @c y via
 * @c atomicAdd , so this test pins both the kernel itself and the COO
 * layout's degenerate "tile is one nonzero" semantics.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/algorithms/spmv/coo_thread_mapped.cuh>

#include "test_spmv_battery.hxx"
#include "test_spmv_runner.hxx"

using namespace loops;
using namespace loops::testing;

TEST_CASE("spmv: coo_thread_mapped (one thread per nonzero, atomicAdd)",
          "[spmv][coo][thread_mapped]") {
  run_battery("coo_thread_mapped",
              [](const csr_host_t& csr, const x_host_t& x) {
                return run_coo_spmv(
                    csr, x, [](auto& coo_d, auto& x_d, auto& y_d) {
                      algorithms::spmv::coo_thread_mapped(coo_d, x_d, y_d);
                    });
              });
}
