/**
 * @file work_oriented.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Benchmark for Sparse Matrix-Vector Multiplication.
 * @version 0.1
 * @date 2022-07-18
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "parameters.hxx"

#include <loops/memory.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/device.hxx>
#include <loops/util/sample.hxx>
#include <loops/util/generate.hxx>
#include <loops/algorithms/spmv/work_oriented.cuh>

#include <nvbench/nvbench.cuh>

#define LOOPS_CUPTI_SUPPORTED 0

using namespace loops;

template <typename value_t>
void work_oriented_bench(nvbench::state& state, nvbench::type_list<value_t>) {
  using index_t = int;
  using offset_t = int;
  using type_t = value_t;

  /// Get sample CSR matrix and create x, y vectors.
  std::size_t rows = 1 << 20;
  std::size_t cols = 1 << 20;

  csr_t<index_t, offset_t, value_t> csr;
  generate::random::csr<index_t, offset_t, type_t>(rows, cols, 0.0000001, csr);

  vector_t<type_t> x(csr.rows);
  vector_t<type_t> y(csr.rows);

  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10);

#if LOOPS_CUPTI_SUPPORTED
  /// Add CUPTI metrics to collect for the state.
  state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();
#endif

  /// Execute the benchmark.
  state.exec(nvbench::exec_tag::sync, [&csr, &x, &y](nvbench::launch& launch) {
    algorithms::spmv::work_oriented(csr, x, y);
  });
}

// Define a type_list to use for the type axis:
using value_types = nvbench::type_list<int, float, double>;
NVBENCH_BENCH_TYPES(work_oriented_bench, NVBENCH_TYPE_AXES(value_types));