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
#include <nvbench/nvbench.cuh>
#include <loops/memory.hxx>
#include <loops/util/device.hxx>
#include <loops/container/market.hxx>
#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/generate.hxx>
#include <loops/algorithms/spmv/work_oriented.cuh>

using namespace loops;

void work_oriented_bench(nvbench::state& state) {
  using index_t = int;
  using offset_t = int;
  using type_t = float;

  csr_t<index_t, offset_t, type_t> csr;
  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr.from_coo(mtx.load(filename));

  vector_t<type_t> x(csr.rows);
  vector_t<type_t> y(csr.rows);

  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10);

  // --
  // Run SPMV with NVBench
  state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    algorithms::spmv::work_oriented(csr, x, y);
  });
}

int main(int argc, char** argv) {
  parameters_t params(argc, argv);
  filename = params.filename;

  if (params.help) {
    // Print NVBench help.
    char* args[1] = {"-h"};
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

    NVBENCH_BENCH(work_oriented_bench);
    NVBENCH_MAIN_BODY(argc - 2, args);
  }
}
