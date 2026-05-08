/**
 * @file test_schedule_coverage.cu
 * @author Loops contributors
 * @brief Direct schedule-iteration coverage tests.
 *
 * Verifies that @c thread_mapped 's @c config.tiles() / @c config.atoms()
 * range pair visits every atom in @c [0, num_atoms) exactly once across
 * all threads of the grid. Implicit coverage of the same property exists
 * through the SpMV correctness battery; this test provides direct
 * triangulation when an SpMV kernel and its schedule fail together.
 *
 * The other three production schedules (group_mapped, work_oriented,
 * merge_path_flat) have setup classes whose iteration is non-trivially
 * tied to per-thread state (cooperative groups, even-share preprocessing,
 * merge-path search). Their coverage is enforced transitively by the
 * SpMV correctness tests against the standard battery, which would
 * surface any double-visit or skip immediately.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/container/csr.hxx>
#include <loops/container/vector.hxx>
#include <loops/schedule.hxx>

#include "test_helpers.hxx"

#include <thrust/copy.h>
#include <thrust/host_vector.h>

#include <cuda_runtime.h>

#include <vector>

using namespace loops;
using namespace loops::testing;

namespace {

template <typename setup_t, typename atom_size_t>
__global__ void thread_mapped_visit_kernel(setup_t config, int* visit_counts) {
  for (auto t : config.tiles()) {
    for (auto a : config.atoms(t)) {
      atomicAdd(&visit_counts[a], 1);
    }
  }
}

}  // namespace

TEST_CASE("schedule::thread_mapped visits every atom exactly once",
          "[schedule][thread_mapped][coverage]") {
  // 4 tiles, 9 atoms, the third tile deliberately empty -> the schedule
  // must not stall or double-visit on the empty-tile boundary.
  auto csr_h = coords_to_csr(4, 4,
                             /*row_idx=*/{0, 0, 0, 1, 1, 3, 3, 3, 3},
                             /*col_idx=*/{0, 1, 2, 0, 3, 0, 1, 2, 3},
                             /*values=*/{1, 1, 1, 1, 1, 1, 1, 1, 1});
  REQUIRE(csr_h.nnzs == 9);

  csr_t<int, int, float> csr_d(csr_h);
  vector_t<int> counts_d(csr_h.nnzs, 0);

  using setup_t =
      schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1, int, int>;
  setup_t config(csr_d.offsets.data().get(),
                 static_cast<std::size_t>(csr_h.rows),
                 static_cast<std::size_t>(csr_h.nnzs));

  // Pick a small block / grid; the schedule should be insensitive to it.
  const int block = 32;
  const int grid = 2;
  thread_mapped_visit_kernel<setup_t, int>
      <<<grid, block>>>(config, thrust::raw_pointer_cast(counts_d.data()));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  thrust::host_vector<int> counts_h(counts_d);
  for (std::size_t a = 0; a < csr_h.nnzs; ++a) {
    INFO("atom " << a);
    CHECK(counts_h[a] == 1);
  }
}

TEST_CASE("schedule::thread_mapped is robust to over-subscribed grids",
          "[schedule][thread_mapped][coverage]") {
  // Same matrix as above but launch with a grid much wider than num_tiles
  // so the grid-stride range exercises the "thread has nothing to do" path.
  auto csr_h = make_banded_csr(8, 1, 1);
  csr_t<int, int, float> csr_d(csr_h);
  vector_t<int> counts_d(csr_h.nnzs, 0);

  using setup_t =
      schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1, int, int>;
  setup_t config(csr_d.offsets.data().get(),
                 static_cast<std::size_t>(csr_h.rows),
                 static_cast<std::size_t>(csr_h.nnzs));

  const int block = 256;
  const int grid = 16;  // 16 * 256 = 4096 threads >> 8 tiles
  thread_mapped_visit_kernel<setup_t, int>
      <<<grid, block>>>(config, thrust::raw_pointer_cast(counts_d.data()));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  thrust::host_vector<int> counts_h(counts_d);
  for (std::size_t a = 0; a < csr_h.nnzs; ++a) {
    INFO("atom " << a);
    CHECK(counts_h[a] == 1);
  }
}
