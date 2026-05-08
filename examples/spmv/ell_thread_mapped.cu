/**
 * @file ell_thread_mapped.cu
 * @author Loops contributors
 * @brief SpMV on an ELL matrix using the thread-mapped schedule.
 *
 * The kernel/schedule code is unchanged from the CSR thread_mapped example;
 * only the layout view is swapped (CSR -> ELL). Validation still uses the
 * CSR reference, since the math is independent of storage.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "helpers.hxx"
#include <loops/algorithms/spmv/ell_thread_mapped.cuh>

using namespace loops;

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = LOOPS_VALUE_T;

  parameters_t parameters(argc, argv);

  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));

  // ELL degenerates to dense storage on power-law / hub-heavy matrices
  // (one entry of pitch per max-degree vertex). Probe the pitch up-front
  // so we bail out with a structured message instead of OOM-killing the
  // process; matrices that need more should pick a different layout.
  //
  // Each ELL cell occupies sizeof(index_t) + sizeof(type_t) bytes (the
  // indices and values arrays are both pitch-by-rows). Compare against
  // the cap as a cell count to keep the multiplication overflow-safe:
  // pitch * rows can wrap on pathological inputs (hub-heavy graphs with
  // billions of rows) and silently slip past a post-multiply check.
  using ell_type = ell_t<index_t, type_t>;
  const std::size_t pitch = ell_type::max_nnz_per_row(csr);
  constexpr std::size_t kMaxEllBytes = std::size_t{4} << 30;  // 4 GiB
  constexpr std::size_t kBytesPerCell = sizeof(index_t) + sizeof(type_t);
  constexpr std::size_t kMaxCells = kMaxEllBytes / kBytesPerCell;
  const bool too_large = pitch != 0 && csr.rows > kMaxCells / pitch;
  if (too_large) {
    std::cout << "ell_thread_mapped," << mtx.dataset << "," << csr.rows << ","
              << csr.cols << "," << csr.nnzs << ",pitch=" << pitch
              << ",SKIP_TOO_LARGE_FOR_ELL" << std::endl;
    std::cout << "Matrix:\t\t" << extract_filename(parameters.filename)
              << std::endl;
    std::cout << "Dimensions:\t" << csr.rows << " x " << csr.cols << " ("
              << csr.nnzs << ")" << std::endl;
    std::cout << "Errors:\t\tSKIP" << std::endl;
    return 0;
  }

  ell_t<index_t, type_t> ell(csr);

  vector_t<type_t> x(csr.cols);
  vector_t<type_t> y(csr.rows);
  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10,
                                         /*seed=*/42u);

  util::timer_t timer;
  timer.start();
  algorithms::spmv::ell_thread_mapped(ell, x, y);
  timer.stop();

  std::cout << "ell_thread_mapped," << mtx.dataset << "," << csr.rows << ","
            << csr.cols << "," << csr.nnzs << ",pitch=" << ell.pitch << ","
            << timer.milliseconds() << std::endl;

  if (parameters.validate)
    cpu::validate(parameters, csr, x, y);
}
