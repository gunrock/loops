/**
 * @file ell_merge_path.cu
 * @author Loops contributors
 * @brief Merge-path SpMV on an ELL matrix.
 *
 * Validates that the merge-path search machinery (which binary-searches the
 * layout's tile_end iterator) works on a non-CSR layout. For ELL the
 * `tile_end_iter()` returns a thrust transform_iterator that synthesizes
 * end-offsets on the fly; if this run matches the CSR reference then the
 * layout abstraction is genuinely format-generic at the merge-path layer.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "helpers.hxx"
#include <loops/algorithms/spmv/ell_merge_path.cuh>

using namespace loops;

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = float;

  parameters_t parameters(argc, argv);

  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));
  ell_t<index_t, type_t> ell(csr);

  vector_t<type_t> x(csr.cols);
  vector_t<type_t> y(csr.rows);
  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10);

  auto timer = algorithms::spmv::ell_merge_path(ell, x, y);

  std::cout << "ell_merge_path," << mtx.dataset << "," << csr.rows << ","
            << csr.cols << "," << csr.nnzs << ",pitch=" << ell.pitch << ","
            << timer.milliseconds() << std::endl;

  if (parameters.validate)
    cpu::validate(parameters, csr, x, y);
}
