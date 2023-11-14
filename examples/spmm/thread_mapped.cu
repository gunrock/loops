/**
 * @file thread_mapped.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse Matrix-Matrix Multiplication example.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "helpers.hxx"
#include <loops/algorithms/spmm/thread_mapped.cuh>

#include "/home/ychenfei/research/libs/loops/examples/spgemm/test_spgemm.cpp"

using namespace loops;

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = float;

  // ... I/O parameters, mtx, etc.
  parameters_t parameters(argc, argv);

  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));

  // Input and output matrices.
  std::size_t n = 10;
  matrix_t<type_t> B(csr.cols, n);
  // matrix_t<type_t> B(mtx.load(parameters.filename));
  matrix_t<type_t> C(csr.rows, n);

  // Generate random numbers between [0, 10].
  generate::random::uniform_distribution(B.m_data.begin(), B.m_data.end(), 1,
                                         10);

  

  // Run the benchmark.
  util::timer_t timer;
  timer.start();
  algorithms::spmm::thread_mapped(csr, B, C);
  timer.stop();

  std::cout << "Elapsed (ms):\t" << timer.milliseconds() << std::endl;

  // loops::matrix_t<type_t, loops::memory_space_t::host> h_B;
  // copyDeviceMtxToHost(B, h_B);
  // writeMtxToFile(h_B, csr.cols, n, "/home/ychenfei/research/libs/loops/examples/spmm/mtx_B.txt");

  // loops::matrix_t<type_t, loops::memory_space_t::host> h_C;
  // copyDeviceMtxToHost(C, h_C);
  // writeMtxToFile(h_C, csr.rows, n, "/home/ychenfei/research/libs/loops/examples/spmm/mxt_C.txt");


  // Copy C from device to host
/*  
  type_t *C_host = new type_t[csr.rows * n];
  cudaError_t err = cudaMemcpy(C_host, C.m_data_ptr, csr.rows * n * sizeof(type_t), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "Error copying data from device to host: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }else{
    std::cout << "Succeeded copying data from device to host!" << std::endl;
  }

  writeMatrixToFile(C_host, csr.rows, n, "/home/ychenfei/research/libs/loops/examples/spmm/spmm_result_cuda.txt");
*/ 
}
