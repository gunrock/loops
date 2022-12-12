/**
 * @file rocsparse.cu
 * @author Muhammad Osama (muhammad.osama@amd.com)
 * @brief Sparse Matrix-Vector Multiplication using rocsparse.
 * @version 0.1
 * @date 2022-12-12
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "helpers.hxx"
#include <rocsparse.h>

using namespace loops;

template <typename index_t, typename offset_t, typename type_t>
double roc_csrmv(csr_t<index_t, offset_t, type_t>& csr,
                 vector_t<type_t>& x,
                 vector_t<type_t>& y) {
  rocsparse_handle handle;
  rocsparse_create_handle(&handle);

  rocsparse_mat_descr descr;
  rocsparse_create_mat_descr(&descr);

  // Create matrix info structure
  rocsparse_mat_info info;
  rocsparse_create_mat_info(&info);

  // Perform analysis step to obtain meta data
  rocsparse_scsrmv_analysis(handle, rocsparse_operation_none, csr.rows,
                            csr.cols, csr.nnzs, descr, csr.values.data().get(),
                            csr.offsets.data().get(), csr.indices.data().get(),
                            info);

  util::timer_t timer;
  timer.start();

  // Compute y = Ax
  type_t beta = 0;
  type_t alpha = 1;
  rocsparse_scsrmv(handle, rocsparse_operation_none, csr.rows, csr.cols,
                   csr.nnzs, &alpha, descr, csr.values.data().get(),
                   csr.offsets.data().get(), csr.indices.data().get(), info,
                   x.data().get(), &beta, y.data().get());
  timer.stop();

  // Clean up
  rocsparse_destroy_mat_info(info);
  rocsparse_destroy_mat_descr(descr);
  rocsparse_destroy_handle(handle);

  return timer.milliseconds();
}

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = float;

  // ... I/O parameters, mtx, etc.
  parameters_t parameters(argc, argv);

  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));

  // Input and output vectors.
  vector_t<type_t> x(csr.cols);
  vector_t<type_t> y(csr.rows);

  // Generate random numbers between [0, 1].
  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10);

  // Run the benchmark.
  auto elapsed = roc_csrmv(csr, x, y);

  std::cout << "rocsparse," << mtx.dataset << "," << csr.rows << "," << csr.cols
            << "," << csr.nnzs << "," << elapsed << std::endl;

  //   // Validation.
  //   if (parameters.validate)
  //     cpu::validate(parameters, csr, x, y);
}