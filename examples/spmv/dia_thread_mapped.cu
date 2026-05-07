/**
 * @file dia_thread_mapped.cu
 * @author Loops contributors
 * @brief SpMV over a DIA matrix via the thread_mapped schedule.
 * @version 0.1
 * @date 2026-05-06
 *
 * Loads a matrix as CSR, builds a DIA view, and runs SpMV with one
 * thread per row. Validates against the same CSR-CPU reference used
 * elsewhere. Sparse but structurally banded matrices (e.g.,
 * chesapeake) compress well; pathological inputs explode the diagonal
 * count and you should pick a different layout.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "helpers.hxx"

#include <loops/container/dia.hxx>
#include <loops/algorithms/spmv/dia_thread_mapped.cuh>

using namespace loops;

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = LOOPS_VALUE_T;

  parameters_t parameters(argc, argv);
  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));

  // DIA degenerates to dense storage on unstructured matrices (one diagonal
  // per stored (col-row) value). Probe the diagonal count up-front so we can
  // bail out with a structured message instead of OOM-killing the process.
  // 4 GiB is generous for a single dense buffer; matrices that would need
  // more should pick a different layout.
  using dia_type = dia_t<index_t, offset_t, type_t>;
  const std::size_t num_diagonals = dia_type::count_diagonals(csr);
  const std::size_t bytes = num_diagonals * csr.rows * sizeof(type_t);
  constexpr std::size_t kMaxDiaBytes = std::size_t{4} << 30;  // 4 GiB
  if (bytes > kMaxDiaBytes) {
    std::cout << "dia_thread_mapped," << mtx.dataset << "," << csr.rows << ","
              << csr.cols << "," << csr.nnzs
              << ",num_diagonals=" << num_diagonals
              << ",SKIP_TOO_LARGE_FOR_DIA" << std::endl;
    std::cout << "Matrix:\t\t" << extract_filename(parameters.filename)
              << std::endl;
    std::cout << "Dimensions:\t" << csr.rows << " x " << csr.cols << " ("
              << csr.nnzs << ")" << std::endl;
    std::cout << "Errors:\t\tSKIP" << std::endl;
    return 0;
  }

  dia_type dia(csr);

  vector_t<type_t> x(csr.cols);
  vector_t<type_t> y(csr.rows);
  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10, /*seed=*/42u);

  auto timer = algorithms::spmv::dia_thread_mapped(dia, x, y);

  std::cout << "dia_thread_mapped," << mtx.dataset << "," << csr.rows << ","
            << csr.cols << "," << csr.nnzs
            << ",num_diagonals=" << dia.num_diagonals << ","
            << timer.milliseconds() << std::endl;

  if (parameters.validate)
    cpu::validate(parameters, csr, x, y);
}
