[README](/README.md) > **Load-Balancing API**

# Load-Balancing API

SpMV problem-specific kernel parameters.

```cpp
template <std::size_t threads_per_block,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __launch_bounds__(threads_per_block, 2)
          spmv(std::size_t rows,
               std::size_t cols,
               std::size_t nnz,
               offset_t* offsets,
               index_t* indices,
               const type_t* values,
               const type_t* x,
               type_t* y) {
```

### (1) Define and configure load-balancing schedule.
Allocates any temporary memory required for load-balancing, as well as constructs a schedule per processors partition (defined using cooperative groups).
```cpp
  using setup_t = schedule::setup<schedule::algroithms_t::tile_mapped,
                                  threads_per_block, 32, index_t, offset_t>;

  /// Allocate temporary storage for the schedule.
  using storage_t = typename setup_t::storage_t;
  __shared__ storage_t temporary_storage;

  /// Construct the schedule.
  setup_t config(temporary_storage, offsets, rows, nnz);
  auto p = config.partition();
```

### (2) Load-balanced ranged loops. (also see; [C++ ranges](https://en.cppreference.com/w/cpp/header/ranges))
In this example, we define two iteration spaces; virtual and real. Virtual spaces allow us to balance atoms and tiles onto the processor ids and link directly to the real iteration space, which returns the exact atom or tile being processed. The code below loops over all balanced number of atoms fetches the tile corresponding to the atom being processed and allows user to define their computation.
```cpp
  for (auto virtual_atom : config.atom_accessor(p)) {
    auto virtual_tile = config.tile_accessor(virtual_atom, p);

    if (!(config.is_valid_accessor(virtual_tile, p)))
      continue;

    auto row = config.tile_id(virtual_tile, p);

    auto nz_idx = config.atom_id(virtual_atom, row, virtual_tile, p);
```

### (3) User-defined computation.
Once the user has access to the atom, tile, and the processor id, they implement the desired computation on the given tuple. In this example, we use a simple `atomicAdd` to perform SpMV (can be improved).
```cpp
    atomicAdd(&(y[row]), values[nz_idx] * x[indices[nz_idx]]);
  }
}
```

[**work_oriented.cuh**](https://github.com/neoblizz/loops/blob/main/include/loops/algorithms/spmv/work_oriented.cuh) (another example)

```cpp
#include <loops/schedule.hxx>

template <std::size_t threads_per_block,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __launch_bounds__(threads_per_block, 2)
    __work_oriented(std::size_t rows,
                    std::size_t cols,
                    std::size_t nnz,
                    offset_t* offsets,
                    index_t* indices,
                    const type_t* values,
                    const type_t* x,
                    type_t* y) {
  using setup_t =
      schedule::setup<schedule::algorithms_t::work_oriented, threads_per_block,
                      1, index_t, offset_t, std::size_t, std::size_t>;

  setup_t config(offsets, rows, nnz);
  auto map = config.init();

  /// Accumulate the complete tiles.
  type_t sum = 0;
  for (auto row : config.tiles(map)) {
    for (auto nz : config.atoms(row, map)) {
      sum += values[nz] * x[indices[nz]];
    }
    y[row] = sum;
    sum = 0;
  }

  /// Process remaining tiles.
  for (auto row : config.remainder_tiles(map)) {
    for (auto nz : config.remainder_atoms(map)) {
      sum += values[nz] * x[indices[nz]];
    }
    /// Accumulate the remainder.
    if (sum != 0)
      atomicAdd(&(y[row]), sum);
  }
}
```