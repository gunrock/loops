/**
 * @file group_mapped.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse Matrix-Vector Multiplication example.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "spmv.hxx"
using namespace loops;

template <std::size_t threads_per_block,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __launch_bounds__(threads_per_block, 2)
    tiled_spmv(std::size_t rows,
               std::size_t cols,
               std::size_t nnz,
               offset_t* offsets,
               index_t* indices,
               const type_t* values,
               const type_t* x,
               type_t* y) {
  using setup_t = schedule::setup<schedule::algorithms_t::tile_mapped,
                                  threads_per_block, 32, index_t, offset_t>;

  /// Allocate temporary storage for the schedule.
  using storage_t = typename setup_t::storage_t;
  __shared__ storage_t temporary_storage;

  /// Construct the schedule.
  setup_t config(temporary_storage, offsets, rows, nnz);
  auto p = config.partition();

  for (auto virtual_atom : config.atom_accessor(p)) {
    auto virtual_tile = config.tile_accessor(virtual_atom, p);

    if (!(config.is_valid_accessor(virtual_tile, p)))
      continue;

    auto row = config.tile_id(virtual_tile, p);

    auto nz_idx = config.atom_id(virtual_atom, row, virtual_tile, p);
    atomicAdd(&(y[row]), values[nz_idx] * x[indices[nz_idx]]);
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
  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10);

  // Create a schedule.
  constexpr std::size_t block_size = 128;

  /// Set-up kernel launch parameters and run the kernel.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  /// Traditional kernel launch, this is nice for tile mapped scheduling, which
  /// will allow blocks to be scheduled in and out as needed. And will rely on
  /// NVIDIA's hardware schedule to schedule the blocks efficiently.
  std::size_t grid_size = (csr.rows + block_size - 1) / block_size;
  launch::non_cooperative(
      stream, tiled_spmv<block_size, index_t, offset_t, type_t>, grid_size,
      block_size, csr.rows, csr.cols, csr.nnzs, csr.offsets.data().get(),
      csr.indices.data().get(), csr.values.data().get(), x.data().get(),
      y.data().get());

  /// Cooperative kernel launch; requires a fixed number of blocks per grid to
  /// be launched, this number can be determined by using CUDA's occupancy API
  /// to figure out how many blocks will run concurrently at all times per SM.
  /// And then we simply loop over the entire work within the kernel.
  // launch::cooperative(stream, tiled_spmv<block_size, index_t, offset_t,
  // type_t>,
  //                     grid_size, block_size, csr.rows, csr.cols, csr.nnzs,
  //                     csr.offsets.data().get(), csr.indices.data().get(),
  //                     csr.values.data().get(), x.data().get(),
  //                     y.data().get());

  cudaStreamSynchronize(stream);

  /// Validation code, can be safely ignored.
  if (parameters.validate) {
    auto h_y = cpu::spmv(csr, x);

    std::size_t errors = util::equal(
        y.data().get(), h_y.data(), csr.rows,
        [](const type_t a, const type_t b) { return std::abs(a - b) > 1e-2; },
        parameters.verbose);

    std::cout << "Matrix:\t\t" << extract_filename(parameters.filename)
              << std::endl;
    std::cout << "Dimensions:\t" << csr.rows << " x " << csr.cols << " ("
              << csr.nnzs << ")" << std::endl;
    std::cout << "Errors:\t\t" << errors << std::endl;
  }
}