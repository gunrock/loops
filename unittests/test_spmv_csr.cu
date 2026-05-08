/**
 * @file test_spmv_csr.cu
 * @author Loops contributors
 * @brief End-to-end correctness for every CSR-input SpMV kernel.
 *
 * Every kernel here consumes @c csr_t and writes @c y = A @c * @c x .
 * The test runs each one against the standard synthetic battery and
 * checks the result against a host reference. A failure flags either:
 *   - a wrong inner accumulation,
 *   - a wrong row->thread mapping,
 *   - a missed output row, or
 *   - a regression in the CSR layout / schedule plumbing.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/algorithms/spmv/group_mapped.cuh>
#include <loops/algorithms/spmv/merge_path_flat.cuh>
#include <loops/algorithms/spmv/original.cuh>
#include <loops/algorithms/spmv/thread_mapped.cuh>
#include <loops/algorithms/spmv/work_oriented.cuh>

#include "test_spmv_battery.hxx"
#include "test_spmv_runner.hxx"

using namespace loops;
using namespace loops::testing;

TEST_CASE("spmv: original (one-thread-per-row baseline)",
          "[spmv][csr][original]") {
  run_battery("original", [](const csr_host_t& csr, const x_host_t& x) {
    return run_csr_spmv(csr, x, [](auto& csr_d, auto& x_d, auto& y_d) {
      algorithms::spmv::original(csr_d, x_d, y_d);
    });
  });
}

TEST_CASE("spmv: thread_mapped (config.tiles / config.atoms)",
          "[spmv][csr][thread_mapped]") {
  run_battery("thread_mapped", [](const csr_host_t& csr, const x_host_t& x) {
    return run_csr_spmv(csr, x, [](auto& csr_d, auto& x_d, auto& y_d) {
      algorithms::spmv::thread_mapped(csr_d, x_d, y_d);
    });
  });
}

TEST_CASE("spmv: group_mapped (cooperative-group reduction)",
          "[spmv][csr][group_mapped]") {
  run_battery("group_mapped", [](const csr_host_t& csr, const x_host_t& x) {
    return run_csr_spmv(csr, x, [](auto& csr_d, auto& x_d, auto& y_d) {
      algorithms::spmv::group_mapped(csr_d, x_d, y_d);
    });
  });
}

TEST_CASE("spmv: work_oriented (NZ-balanced)", "[spmv][csr][work_oriented]") {
  run_battery("work_oriented", [](const csr_host_t& csr, const x_host_t& x) {
    return run_csr_spmv(csr, x, [](auto& csr_d, auto& x_d, auto& y_d) {
      algorithms::spmv::work_oriented(csr_d, x_d, y_d);
    });
  });
}

TEST_CASE("spmv: merge_path (Merrill-Garland)", "[spmv][csr][merge_path]") {
  run_battery("merge_path", [](const csr_host_t& csr, const x_host_t& x) {
    return run_csr_spmv(csr, x, [](auto& csr_d, auto& x_d, auto& y_d) {
      algorithms::spmv::merge_path_flat(csr_d, x_d, y_d);
    });
  });
}
