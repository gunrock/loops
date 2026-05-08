/**
 * @file test_spmv_partitioned.cu
 * @author Loops contributors
 * @brief End-to-end correctness for the @c flat_partitioned SpMV kernel.
 *
 * @c flat_partitioned wraps a CSR base layout in a
 * @c flat_uniform_occupancy<K> partitioner; every K nonzeros become a tile
 * regardless of row boundaries. The kernel uses @c atomicAdd to merge
 * partial sums back to the correct row. This test pins both the
 * partitioner adaptor and the kernel against a battery of synthetic
 * matrices, so any regression in either component will surface here.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/algorithms/spmv/flat_partitioned.cuh>

#include "test_spmv_battery.hxx"
#include "test_spmv_runner.hxx"

using namespace loops;
using namespace loops::testing;

TEST_CASE("spmv: flat_partitioned (K=4 atoms-per-tile via partitioner)",
          "[spmv][partitioned][flat]") {
  run_battery("flat_partitioned", [](const csr_host_t& csr, const x_host_t& x) {
    return run_csr_spmv(csr, x, [](auto& csr_d, auto& x_d, auto& y_d) {
      algorithms::spmv::flat_partitioned(csr_d, x_d, y_d);
    });
  });
}
