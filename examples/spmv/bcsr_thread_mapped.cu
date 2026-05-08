/**
 * @file bcsr_thread_mapped.cu
 * @author Loops contributors
 * @brief SpMV over a BCSR matrix via the thread_mapped schedule.
 * @version 0.1
 * @date 2026-05-06
 *
 * Loads a matrix as CSR, builds a 2x2 Block-CSR view of it, then runs
 * SpMV with one thread per block-row. Validates against the same
 * CSR-CPU reference used by every other in-tree SpMV example.
 *
 * Note: @c x is allocated to the padded column count
 * (@c num_block_cols * C ) so the inner kernel loop can read
 * @c x[bc * C + j] unconditionally for every stored block. The trailing
 * padding entries are left zero by @c vector_t 's default-construction.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "helpers.hxx"

#include <loops/container/bcsr.hxx>
#include <loops/algorithms/spmv/bcsr_thread_mapped.cuh>

using namespace loops;

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = LOOPS_VALUE_T;

  constexpr std::size_t kR = 2;
  constexpr std::size_t kC = 2;

  parameters_t parameters(argc, argv);
  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));
  bcsr_t<kR, kC, index_t, offset_t, type_t> bcsr(csr);

  vector_t<type_t> x(bcsr.num_block_cols * kC);
  vector_t<type_t> y(csr.rows);
  generate::random::uniform_distribution(x.begin(), x.begin() + csr.cols, 1,
                                         10, /*seed=*/42u);

  auto timer = algorithms::spmv::bcsr_thread_mapped<kR, kC>(bcsr, x, y);

  std::cout << "bcsr_thread_mapped," << mtx.dataset << "," << csr.rows << ","
            << csr.cols << "," << csr.nnzs << "," << timer.milliseconds()
            << std::endl;

  if (parameters.validate) {
    vector_t<type_t> x_csr(csr.cols);
    thrust::copy(x.begin(), x.begin() + csr.cols, x_csr.begin());
    cpu::validate(parameters, csr, x_csr, y);
  }
}
