/**
 * @file spmv.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse Matrix-Vector Multiplication example.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "spmv.hxx"

// #define DEBUG 1

// template <typename index_t, typename offset_t, typename type_t>
// __global__ void spmv(const std::size_t rows,
//                      const std::size_t cols,
//                      const std::size_t nnz,
//                      const offset_t* offsets,
//                      const index_t* indices,
//                      const type_t* values,
//                      const type_t* x,
//                      type_t* y) {
//   /// Equivalent to:
//   /// row = blockIdx.x * blockDim.x + threadIdx.x; (init)
//   /// row < rows; (boundary condition)
//   /// row += gridDim.x * blockDim.x. (step)
//   for (auto row : grid_stride_range(std::size_t(0), rows)) {
//     type_t sum = 0;

//     /// Equivalent to:
//     /// for (offset_t nz = offset; nz < end; ++nz)
//     for (auto nz : range(offsets[row], offsets[row + 1])) {
//       sum += values[nz] * x[indices[nz]];
//     }

//     // Output
//     y[row] = sum;
//   }
// }

using namespace loops;

template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void spmv(setup_t config,
                     const std::size_t rows,
                     const std::size_t cols,
                     const std::size_t nnz,
                     const offset_t* offsets,
                     const index_t* indices,
                     const type_t* values,
                     const type_t* x,
                     type_t* y) {
  /// Equivalent to:
  /// row = blockIdx.x * blockDim.x + threadIdx.x; (init)
  /// row < rows; (boundary condition)
  /// row += gridDim.x * blockDim.x. (step)
  for (auto row : config.tiles()) {
    type_t sum = 0;

    /// Equivalent to:
    /// for (offset_t nz = offset; nz < end; ++nz)
    for (auto nz : config.atoms(row)) {
      sum += values[nz] * x[indices[nz]];
    }

    // Output
    y[row] = sum;
  }
}

template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __launch_bounds__(setup_t::threads_per_block, 2)
    tiled_spmv(setup_t config,
               const std::size_t rows,
               const std::size_t cols,
               const std::size_t nnz,
               const offset_t* offsets,
               const index_t* indices,
               const type_t* values,
               const type_t* x,
               type_t* y) {
  using storage_t = typename setup_t::storage_t;
  using tile_storage_t = typename setup_t::tile_storage_t;

  // reserve shared memory for thread_block_tile usage.
  __shared__ cooperative_groups::experimental::block_tile_memory<
      4, setup_t::threads_per_block>
      groups_space;

  __shared__ storage_t
      tile_aggregates[setup_t::threads_per_block / setup_t::threads_per_tile];
  __shared__ storage_t storage[setup_t::threads_per_block];
  __shared__ tile_storage_t tile_id_storage[setup_t::threads_per_block];
  storage_t thread_offsets[1];

#ifdef DEBUG
  if (threadIdx.x == 0) {
    printf("block id = %d\n", blockIdx.x);
  }
#endif

  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < rows) {
#ifdef DEBUG
    printf("idx, rows : (%d, %d)\n", (int)idx, (int)rows);
#endif
    tile_id_storage[threadIdx.x] = idx;
    thread_offsets[0] = offsets[idx + 1] - offsets[idx];
  } else {
    tile_id_storage[threadIdx.x] = -1;
    thread_offsets[0] = 0;
  }

  auto g = cooperative_groups::this_grid();
  auto b = cooperative_groups::experimental::this_thread_block(groups_space);
  auto p = cooperative_groups::experimental::tiled_partition<
      setup_t::threads_per_tile>(b);

  for (auto virtual_atom :
       config.virtual_atoms(storage, thread_offsets, tile_aggregates, p)) {
    auto row = config.tile_id(storage, virtual_atom, p);

    if (!(config.is_valid_tile(row, p)))
      continue;

    typename setup_t::tiles_t r = tile_id_storage[row];
    if (r == -1)
      continue;

    auto nz = config.atom_id(storage, virtual_atom, r, row, p);
    atomicAdd(&y[r], values[nz] * x[indices[nz]]);

#ifdef DEBUG
    printf("y[%d] @ [%d] = %f * %f @ %d\n",  // <---
           r, nz, values[nz], x[indices[nz]], indices[nz]);
#endif
  }
}

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = float;

  // ... I/O parameters, mtx, etc.
  parameters_t parameters(argc, argv);

  csr_t<index_t, offset_t, type_t> csr;
  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr.from_coo(mtx.load(parameters.filename));

  // Input and output vectors.
  vector_t<type_t> x(csr.rows);
  vector_t<type_t> y(csr.rows);

  // Generate random numbers between [0, 1].
  generate::random::uniform_distribution(x.begin(), x.end());

  // Create a schedule.
  constexpr std::size_t block_size = 32;
  constexpr std::size_t tile_size = 32;
  using setup_t = schedule::setup<schedule::algroithms_t::block_mapped,
                                  block_size, tile_size, index_t, offset_t>;

  setup_t config(csr.offsets.data().get(), csr.rows, csr.nnzs);

  // Compute the spmv.
  std::size_t grid_size = (csr.rows + block_size - 1) / block_size;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // launch::non_cooperative(
  //     stream, tiled_spmv<setup_t, index_t, offset_t, type_t>, grid_size,
  //     block_size, config, csr.rows, csr.cols, csr.nnzs,
  //     csr.offsets.data().get(), csr.indices.data().get(),
  //     csr.values.data().get(), x.data().get(), y.data().get());

  launch::cooperative(stream, tiled_spmv<setup_t, index_t, offset_t, type_t>,
                      grid_size, block_size, config, csr.rows, csr.cols,
                      csr.nnzs, csr.offsets.data().get(),
                      csr.indices.data().get(), csr.values.data().get(),
                      x.data().get(), y.data().get());

  cudaStreamSynchronize(stream);

  if (parameters.validate) {
    auto h_y = cpu::spmv(csr, x);

    std::size_t errors = util::equal(
        y.data().get(), h_y.data(), csr.rows,
        [](const type_t a, const type_t b) { return std::abs(a - b) > 1e-4; },
        parameters.verbose);

    std::cout << "Matrix:\t\t" << extract_filename(parameters.filename)
              << std::endl;
    std::cout << "Dimensions:\t" << csr.rows << " x " << csr.cols << " ("
              << csr.nnzs << ")" << std::endl;
    std::cout << "Errors:\t\t" << errors << std::endl;
  }
}