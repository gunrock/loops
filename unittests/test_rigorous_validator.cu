/**
 * @file test_rigorous_validator.cu
 * @brief Smoke tests for @c loops::reference::rigorously_validate_spmv .
 *
 * The rigorous validator is the harness we lean on to declare a kernel
 * "not buggy" when the naive default tolerance flags a row -- so it
 * has to itself be correct. These tests pin down two opposite cases:
 *
 *   1. A correct GPU result on a stiffness-style matrix produces
 *      @c gpu_overruns == 0 even when @c naive_mismatches > 0 .
 *   2. A deliberately-corrupted GPU result (one entry flipped to
 *      a wrong value) produces @c gpu_overruns >= 1 .
 *
 * Both checks run end-to-end on a real SpMV kernel ( @c thread_mapped )
 * so the validator and the kernel are exercised together.
 */

#include <catch2/catch_test_macros.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <loops/algorithms/spmv/thread_mapped.cuh>
#include <loops/container/csr.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/reference.hxx>

#include "test_helpers.hxx"

#include <cmath>
#include <vector>

using namespace loops::testing;

namespace {

/// Build a CSR matrix with row sums in the @c O(100) range and unit-magnitude
/// entries, the regime where float32 round-off accumulates into measurable
/// (but bug-free) absolute error on long rows.
loops::csr_t<int, int, float, loops::memory::memory_space_t::host>
make_long_row_csr(int rows, int nnz_per_row) {
  std::vector<int> ri, ci;
  std::vector<float> vs;
  for (int r = 0; r < rows; ++r) {
    for (int k = 0; k < nnz_per_row; ++k) {
      ri.push_back(r);
      ci.push_back((r + k) % rows);
      // Alternating sign with a small magnitude bias keeps the row
      // sum non-trivial while @c sum_of_abs is much larger -- the
      // catastrophic-cancellation regime.
      float v = (k % 2 == 0 ? 1.0f : -1.0f) +
                static_cast<float>(k) * 1e-4f;
      vs.push_back(v);
    }
  }
  return coords_to_csr(rows, rows, std::move(ri), std::move(ci),
                       std::move(vs));
}

}  // namespace

namespace {
using index_t = int;
using offset_t = int;
using type_t = float;
using mem = loops::memory::memory_space_t;

/// Run @c thread_mapped on @c h_csr / @c h_x and return the device CSR + y
/// so the test body can hand them straight to @c rigorously_validate_spmv .
struct device_run {
  loops::csr_t<index_t, offset_t, type_t, mem::device> csr;
  loops::vector_t<type_t, mem::device> x;
  loops::vector_t<type_t, mem::device> y;
};

device_run run_thread_mapped(
    const loops::csr_t<index_t, offset_t, type_t, mem::host>& h_csr,
    const loops::vector_t<type_t, mem::host>& h_x) {
  device_run dr{
      loops::csr_t<index_t, offset_t, type_t, mem::device>(h_csr),
      loops::vector_t<type_t, mem::device>(h_x),
      loops::vector_t<type_t, mem::device>(h_csr.rows, type_t{0})};
  loops::algorithms::spmv::thread_mapped(dr.csr, dr.x, dr.y);
  return dr;
}
}  // namespace

TEST_CASE("rigorous_validate: clean kernel passes on long rows",
          "[rigorous]") {
  auto h_csr = make_long_row_csr(/*rows=*/256, /*nnz_per_row=*/64);
  auto h_x_legacy = make_input_vector(h_csr);
  loops::vector_t<type_t, mem::host> h_x(h_csr.cols);
  for (std::size_t i = 0; i < h_csr.cols; ++i) h_x[i] = h_x_legacy[i];

  auto dr = run_thread_mapped(h_csr, h_x);
  auto report = loops::reference::rigorously_validate_spmv(dr.csr, dr.x,
                                                           dr.y.data().get());

  // The validator's own correctness invariant: the GPU and the f32 host
  // reference compute the same kind of arithmetic, so the GPU must not
  // overrun the Wilkinson bound any worse than the f32 baseline does.
  REQUIRE(report.gpu_overruns <= report.f32_baseline_overruns + 4);
  REQUIRE(report.max_gpu_rel_error < 1e-3);
}

TEST_CASE("rigorous_validate: corrupted output is flagged",
          "[rigorous]") {
  auto h_csr = make_banded_csr(/*n=*/128, /*lower=*/2, /*upper=*/2);
  auto h_x_legacy = make_input_vector(h_csr);
  loops::vector_t<type_t, mem::host> h_x(h_csr.cols);
  for (std::size_t i = 0; i < h_csr.cols; ++i) h_x[i] = h_x_legacy[i];

  auto dr = run_thread_mapped(h_csr, h_x);

  // Deliberately corrupt a single output: replace y[7] with a value that
  // is far outside any reasonable Wilkinson bound (1e6 cannot arise from
  // banded * O(1) input).
  type_t bogus = static_cast<type_t>(1e6);
  cudaMemcpy(dr.y.data().get() + 7, &bogus, sizeof(type_t),
             cudaMemcpyHostToDevice);

  auto report = loops::reference::rigorously_validate_spmv(dr.csr, dr.x,
                                                           dr.y.data().get());

  REQUIRE(report.gpu_overruns >= 1);
  REQUIRE(report.max_gpu_abs_error > 1e3);
}

TEST_CASE("rigorous_validate: f64 reference matches f32 reference on identity",
          "[rigorous]") {
  // Identity is the only matrix where @c spmv (f32) and @c spmv_f64 agree
  // bit-for-bit (no summation, just one multiply). Both the f32-baseline
  // and GPU overrun counts must be zero.
  auto h_csr = make_identity_csr(/*n=*/64);
  auto h_x_legacy = make_input_vector(h_csr);
  loops::vector_t<type_t, mem::host> h_x(h_csr.cols);
  for (std::size_t i = 0; i < h_csr.cols; ++i) h_x[i] = h_x_legacy[i];

  auto dr = run_thread_mapped(h_csr, h_x);
  auto report = loops::reference::rigorously_validate_spmv(dr.csr, dr.x,
                                                           dr.y.data().get());

  REQUIRE(report.f32_baseline_overruns == 0);
  REQUIRE(report.gpu_overruns == 0);
  REQUIRE(report.max_gpu_abs_error == 0.0);
}
