/**
 * @file flat_partitioned.cu
 * @author Loops contributors
 * @brief SpMV driven by a layout-level partitioner.
 * @version 0.1
 * @date 2026-05-06
 *
 * Wires a CSR matrix into the @c thread_mapped schedule via the
 * @c layout::flat_uniform_occupancy<K, csr_layout> adaptor. The schedule
 * sees @c ceil(nnz/K) tiles of @c K atoms each (the last possibly
 * smaller); the kernel atomic-adds into @c y because tiles cross row
 * boundaries.
 *
 * Compared against the non-partitioned @c thread_mapped run (which uses
 * one tile per row), this example demonstrates that the partitioner is
 * just another layout view as far as the schedule is concerned — the
 * schedule code is identical.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "helpers.hxx"
#include <loops/algorithms/spmv/flat_partitioned.cuh>

using namespace loops;

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = float;

  parameters_t parameters(argc, argv);
  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));

  vector_t<type_t> x(csr.cols);
  vector_t<type_t> y(csr.rows);
  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10);

  /// Partitioner trade-off: smaller K gives the schedule more tiles
  /// (better load balance, more atomic contention); larger K reduces
  /// atomics but lets long rows dominate. K=8 is a reasonable default.
  constexpr std::size_t kAtomsPerTile = 8;
  auto timer = algorithms::spmv::flat_partitioned<kAtomsPerTile>(csr, x, y);

  std::cout << "flat_partitioned," << mtx.dataset << "," << csr.rows << ","
            << csr.cols << "," << csr.nnzs << ",K=" << kAtomsPerTile << ","
            << timer.milliseconds() << std::endl;

  if (parameters.validate)
    cpu::validate(parameters, csr, x, y);
}
