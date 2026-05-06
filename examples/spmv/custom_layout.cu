/**
 * @file custom_layout.cu
 * @author Loops contributors
 * @brief How to write a user-defined layout and run any schedule on it.
 *
 * This example shows that the layout contract in
 * `loops/container/layout.hxx` is genuinely user-extensible: nothing in the
 * schedules is specific to CSR or even to the layouts shipped in-tree. Any
 * struct that implements the 5+ method contract can be plugged into any of
 * the four schedules.
 *
 * The "format" used here is intentionally toy:
 *
 *   struct row_padded { num_rows, pitch, indices, values; }
 *
 * which is essentially ELL by another name. The point is *not* the format;
 * it's that the user defines `row_padded_layout` in *their* translation unit
 * and hands it to `schedule::setup<...>`. The schedule machinery has zero
 * knowledge of `row_padded` yet drives the whole computation.
 *
 * The same pattern works for any matrix format you might imagine:
 * BCSR (tile = row of blocks), DIA (tile = row, atoms = diagonals), CSR5,
 * sliced ELL, partitioned/distributed slices of a larger matrix, etc.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "helpers.hxx"

#include <loops/schedule.hxx>
#include <loops/container/csr.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/launch.hxx>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>

#include <algorithm>

using namespace loops;

namespace {

// ---------------------------------------------------------------------------
// 1. The user's custom layout view.
//
// Satisfies the contract in loops/container/layout.hxx. POD-like, passed
// by value into kernels. Does NOT own any storage; the index/value buffers
// live separately in user-managed device memory.
// ---------------------------------------------------------------------------
template <typename TileId, typename AtomId>
struct row_padded_layout {
 private:
  struct tile_end_fn {
    AtomId pitch;
    __host__ __device__ AtomId operator()(TileId i) const {
      return static_cast<AtomId>(i + 1) * pitch;
    }
  };

 public:
  using tile_id_t = TileId;
  using atom_id_t = AtomId;
  using tile_end_iterator_t = thrust::transform_iterator<
      tile_end_fn,
      thrust::counting_iterator<TileId>,
      AtomId>;

  TileId n_rows_;
  AtomId pitch_;

  __host__ __device__ row_padded_layout() : n_rows_(0), pitch_(0) {}
  __host__ __device__ row_padded_layout(TileId n, AtomId p)
      : n_rows_(n), pitch_(p) {}

  __host__ __device__ TileId num_tiles() const { return n_rows_; }
  __host__ __device__ AtomId num_atoms() const {
    return static_cast<AtomId>(n_rows_) * pitch_;
  }
  __host__ __device__ AtomId tile_begin(TileId t) const {
    return static_cast<AtomId>(t) * pitch_;
  }
  __host__ __device__ AtomId tile_end(TileId t) const {
    return static_cast<AtomId>(t + 1) * pitch_;
  }
  __host__ __device__ AtomId tile_size(TileId /*t*/) const { return pitch_; }

  __host__ __device__ tile_end_iterator_t tile_end_iter() const {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<TileId>(0), tile_end_fn{pitch_});
  }
};

// ---------------------------------------------------------------------------
// 2. A trivial host-side bucket-fill: pull a CSR matrix into a row-padded
//    flat layout (sentinel = -1 for missing entries).
// ---------------------------------------------------------------------------
template <typename index_t, typename offset_t, typename value_t>
struct row_padded_storage {
  std::size_t num_rows;
  std::size_t pitch;
  thrust::device_vector<index_t> indices;
  thrust::device_vector<value_t> values;
  static constexpr index_t kPad = static_cast<index_t>(-1);

  static row_padded_storage from_csr(
      const csr_t<index_t, offset_t, value_t>& csr) {
    csr_t<index_t, offset_t, value_t, memory_space_t::host> h(csr);

    std::size_t mpr = 0;
    for (std::size_t r = 0; r < h.rows; ++r) {
      mpr = std::max<std::size_t>(mpr, h.offsets[r + 1] - h.offsets[r]);
    }

    thrust::host_vector<index_t> h_idx(h.rows * mpr, kPad);
    thrust::host_vector<value_t> h_val(h.rows * mpr, value_t(0));
    for (std::size_t r = 0; r < h.rows; ++r) {
      auto begin = h.offsets[r];
      auto end = h.offsets[r + 1];
      for (auto k = begin; k < end; ++k) {
        const std::size_t slot = r * mpr + (k - begin);
        h_idx[slot] = h.indices[k];
        h_val[slot] = h.values[k];
      }
    }

    row_padded_storage out;
    out.num_rows = h.rows;
    out.pitch = mpr;
    out.indices = h_idx;
    out.values = h_val;
    return out;
  }
};

// ---------------------------------------------------------------------------
// 3. The kernel: pure schedule-based iteration. Note that nothing here
//    knows about row_padded_layout in particular. Any layout satisfying
//    the contract works.
// ---------------------------------------------------------------------------
template <typename setup_t, typename index_t, typename type_t>
__global__ void __custom_layout_spmv(setup_t config,
                                     const index_t* indices,
                                     const type_t* values,
                                     const type_t* x,
                                     type_t* y) {
  for (auto row : config.tiles()) {
    type_t sum = 0;
    for (auto atom : config.atoms(row)) {
      const index_t col = indices[atom];
      if (col >= 0)
        sum += values[atom] * x[col];
    }
    y[row] = sum;
  }
}

}  // namespace

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = float;

  parameters_t parameters(argc, argv);

  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));
  auto storage = row_padded_storage<index_t, offset_t, type_t>::from_csr(csr);

  vector_t<type_t> x(csr.cols);
  vector_t<type_t> y(csr.rows);
  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10);

  // ---------------------------------------------------------------------
  // Wire the user-defined layout into the existing thread-mapped schedule.
  // ---------------------------------------------------------------------
  using tile_id_t = index_t;
  using atom_id_t = index_t;
  using my_layout_t = row_padded_layout<tile_id_t, atom_id_t>;
  using setup_t = schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1,
                                  tile_id_t, atom_id_t,
                                  std::size_t, std::size_t,
                                  my_layout_t>;

  my_layout_t lay(static_cast<tile_id_t>(storage.num_rows),
                  static_cast<atom_id_t>(storage.pitch));
  setup_t config(lay);

  util::timer_t timer;
  timer.start();
  constexpr std::size_t block_size = 128;
  std::size_t grid_size = (storage.num_rows + block_size - 1) / block_size;
  launch::non_cooperative(
      0, __custom_layout_spmv<setup_t, index_t, type_t>, grid_size, block_size,
      config, thrust::raw_pointer_cast(storage.indices.data()),
      thrust::raw_pointer_cast(storage.values.data()), x.data().get(),
      y.data().get());
  cudaStreamSynchronize(0);
  timer.stop();

  std::cout << "custom_layout," << mtx.dataset << "," << csr.rows << ","
            << csr.cols << "," << csr.nnzs << ",pitch=" << storage.pitch << ","
            << timer.milliseconds() << std::endl;

  if (parameters.validate)
    cpu::validate(parameters, csr, x, y);
}
