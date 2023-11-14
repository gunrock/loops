/**
 * @file thread_mapped.cu
 * @author 
 * @brief SpGEMM example
 * @version 0.1
 * @date 2023
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "helpers.hxx"
#include <loops/algorithms/spgemm/thread_mapped.cuh>
#include <loops/algorithms/spgemm/estimate_nnz.cuh>
#include <loops/algorithms/spgemm/estimate_nnz_test.cuh>
#include <loops/algorithms/spgemm/find_explicit_zeros.cuh>

#include "test_spgemm.cpp"

using namespace loops;

int main(int argc, char** argv) {
  util::timer_t timer;

  using index_t = int;
  using offset_t = int;
  using type_t = float;

  // ... I/O parameters, mtx, etc.
  parameters_t parameters(argc, argv);

  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));
  csc_t<index_t, offset_t, type_t> csc(mtx.load(parameters.filename));

  int* h_nnz_C = new int[csc.cols]();
  int* d_nnz_C;

  cudaMalloc(&d_nnz_C, csc.cols * sizeof(int));
  cudaMemcpy(d_nnz_C, h_nnz_C, csc.cols * sizeof(int), cudaMemcpyHostToDevice);

  // algorithms::spgemm::estimate_nnz_test(csr, csc, d_nnz_C);
  // copyAndSumEstimateNnzToHost(d_nnz_C, csc.cols);

  // find_explicit_zeros NOT WORKING
  algorithms::spgemm::find_explicit_zeros(csr, csc, d_nnz_C);
  copyAndSumEstimateNnzToHost(d_nnz_C, csc.cols);


/* SpGEMM */
/*
  // Output matrix.
  matrix_t<type_t> C(csr.rows, csc.cols);

  // Run the benchmark.
  timer.start();
  algorithms::spgemm::thread_mapped(csr, csc, C);
  timer.stop();

  std::cout << "Elapsed (ms):\t" << timer.milliseconds() << std::endl;


  loops::matrix_t<type_t, loops::memory_space_t::host> h_C;
  copyDeviceMtxToHost(C, h_C);
  writeMtxToFile(h_C, csr.rows, csc.cols, "/home/ychenfei/research/libs/loops/examples/spgemm/export_mtx/test.txt");
*/
}