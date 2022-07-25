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

#include <loops/util/generate.hxx>
#include <loops/container/container.hxx>
#include <loops/algorithms/spmv/work_oriented.cuh>

#include <nvbench/nvbench.cuh>

#define LOOPS_CUPTI_SUPPORTED 0

using namespace loops;

template <typename value_t>
void work_oriented_bench(nvbench::state& state, nvbench::type_list<value_t>) {
  using index_t = int;
  using offset_t = int;
  using type_t = value_t;

  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(filename));

  vector_t<type_t> x(csr.rows);
  vector_t<type_t> y(csr.rows);

  generate::random::uniform_distribution(x.begin(), x.end(), type_t(1.0),
                                         type_t(10.0));

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

int main(int argc, char** argv) {
  parameters_t params(argc, argv);
  filename = params.filename;

  if (params.help) {
    // Print NVBench help.
    const char* args[1] = {"-h"};
    NVBENCH_MAIN_BODY(1, args);
  } else {
    // Create a new argument array without matrix filename to pass to NVBench.
    char* args[argc - 2];
    int j = 0;
    for (int i = 0; i < argc; i++) {
      if (strcmp(argv[i], "--market") == 0 || strcmp(argv[i], "-m") == 0) {
        i++;
        continue;
      }
      args[j] = argv[i];
      j++;
    }
    NVBENCH_MAIN_BODY(argc - 2, args);
  }
}