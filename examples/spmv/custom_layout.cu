/**
 * @file custom_layout.cu
 * @author Loops contributors
 * @brief How to write a user-defined layout and run any schedule on it.
 * @version 0.1
 * @date 2026-05-05
 *
 * This example shows that the layout contract in
 * `loops/container/layout.hxx` is genuinely user-extensible: nothing in the
 * schedules is specific to CSR or even to the layouts shipped in-tree. Any
 * struct that implements the 5+ method contract can be plugged into any of
 * the four schedules.
 *
 * The "format" used here is intentionally toy:
 *
 *     struct row_padded { num_rows, pitch, indices, values; };
 *
 * which is essentially ELL by another name. The point is *not* the format;
 * it's that the user defines @c row_padded_layout in *their* translation
 * unit and hands it to @c schedule::setup<...> . The schedule machinery
 * has zero knowledge of @c row_padded yet drives the whole computation.
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

/**
 * @brief User-defined row-padded layout view.
 *
 * Satisfies the layout contract in @c loops/container/layout.hxx so it can
 * be plugged into any of the in-tree schedules. POD-like and passed by
 * value into @c __global__ kernels; does *not* own any storage. The index
 * and value buffers live separately in user-managed device memory.
 *
 * Each tile holds exactly @c pitch_ atoms; rows shorter than @c pitch_
 * carry a sentinel column id (@c -1 ) and a zero value, which the kernel
 * skips.
 *
 * @tparam tile_id_type Tile-id type (e.g., row id).
 * @tparam atom_id_type Atom-id type (flat index into the per-row buckets).
 */
template <typename tile_id_type, typename atom_id_type>
struct row_padded_layout {
 private:
  /// Functor used to materialize tile_end values lazily for merge-path
  /// schedules.
  struct tile_end_fn {
    atom_id_type pitch;
    __host__ __device__ atom_id_type operator()(tile_id_type i) const {
      return static_cast<atom_id_type>(i + 1) * pitch;
    }
  };

 public:
  using tile_id_t = tile_id_type;
  using atom_id_t = atom_id_type;
  using tile_end_iterator_t = thrust::transform_iterator<
      tile_end_fn,
      thrust::counting_iterator<tile_id_t>,
      atom_id_t>;

  tile_id_t n_rows_;
  atom_id_t pitch_;  /// atoms per tile (uniform); = max-non-zeros-per-row.

  __host__ __device__ row_padded_layout() : n_rows_(0), pitch_(0) {}
  __host__ __device__ row_padded_layout(tile_id_t n, atom_id_t p)
      : n_rows_(n), pitch_(p) {}

  __host__ __device__ tile_id_t num_tiles() const { return n_rows_; }
  __host__ __device__ atom_id_t num_atoms() const {
    return static_cast<atom_id_t>(n_rows_) * pitch_;
  }
  __host__ __device__ atom_id_t tile_begin(tile_id_t t) const {
    return static_cast<atom_id_t>(t) * pitch_;
  }
  __host__ __device__ atom_id_t tile_end(tile_id_t t) const {
    return static_cast<atom_id_t>(t + 1) * pitch_;
  }
  __host__ __device__ atom_id_t tile_size(tile_id_t /*t*/) const {
    return pitch_;
  }

  /// Random-access iterator @c i where @c i[k] @c == @c tile_end(k).
  __host__ __device__ tile_end_iterator_t tile_end_iter() const {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<tile_id_t>(0), tile_end_fn{pitch_});
  }
};

/**
 * @brief Toy host-side storage bucket-filled from a CSR matrix.
 *
 * Computes @c pitch as the matrix-wide max non-zeros-per-row, then
 * row-major fills two dense arrays of length @c rows*pitch . Missing
 * entries use the sentinel column id @c kPad and a zero value.
 *
 * The data layout (row-major dense, sentinel-padded) matches what
 * @c row_padded_layout assumes; the layout view above is a non-owning
 * window into this storage.
 *
 * @tparam index_t  Type of the column indices.
 * @tparam offset_t Type of the source CSR offsets.
 * @tparam value_t  Type of the non-zero values.
 */
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

/**
 * @brief SpMV kernel driven by a layout-generic schedule.
 *
 * Nothing here knows about @c row_padded_layout in particular; the kernel
 * walks @c config.tiles() and @c config.atoms(row) , both of which are
 * provided by the schedule on top of whatever layout the caller passed.
 * The same kernel body would compile against CSR, ELL, or any other
 * layout satisfying the contract.
 *
 * @tparam setup_t Schedule setup (e.g., @c schedule::setup<thread_mapped,...> ).
 * @tparam index_t Column-index type.
 * @tparam type_t  Value type.
 */
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

  /// Wire the user-defined layout into the existing thread-mapped schedule.
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
