/**
 * @file coo_thread_mapped.cu
 * @author Loops contributors
 * @brief SpMV over a COO matrix via the thread_mapped schedule.
 * @version 0.1
 * @date 2026-05-06
 *
 * Loads a matrix as CSR (Matrix-Market parser still produces CSR), then
 * converts to COO and runs the canonical scalar-COO SpMV (one thread per
 * nonzero, atomic-add into @c y ). Demonstrates that @c layout::coo plugs
 * into the same @c thread_mapped schedule as every other in-tree layout.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "helpers.hxx"

#include <loops/container/coo.hxx>
#include <loops/algorithms/spmv/coo_thread_mapped.cuh>

using namespace loops;

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = LOOPS_VALUE_T;

  parameters_t parameters(argc, argv);
  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));
  coo_t<index_t, type_t> coo(csr);

  vector_t<type_t> x(csr.cols);
  vector_t<type_t> y(csr.rows);
  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10, /*seed=*/42u);

  auto timer = algorithms::spmv::coo_thread_mapped(coo, x, y);

  std::cout << "coo_thread_mapped," << mtx.dataset << "," << csr.rows << ","
            << csr.cols << "," << csr.nnzs << "," << timer.milliseconds()
            << std::endl;

  if (parameters.validate)
    cpu::validate(parameters, csr, x, y);
}
