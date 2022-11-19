# 🐧 `loops`: Expressing Parallel Irregular Computations
We propose an open-source GPU load-balancing framework for applications that exhibit irregular parallelism. The set of applications and algorithms we consider are fundamental to computing tasks ranging from sparse machine learning, large numerical simulations, and on through to graph analytics. The underlying data and data structures that drive these applications present access patterns that naturally don't map well to the GPU's architecture that is designed with dense and regular patterns in mind. 

Prior to the work we present and propose here, the only way to unleash the GPU's full power on these problems has been to workload balance through tightly coupled load-balancing techniques. Our proposed load-balancing abstraction decouples load balancing from work processing and aims to support both static and dynamic schedules with a programmable interface to implement new load-balancing schedules in the future. 

With our open-source framework, we hope to not only improve programmers' productivity when developing irregular-parallel algorithms on the GPU but also improve the overall performance characteristics for such applications by allowing a quick path to experimentation with a variety of existing load-balancing techniques. Consequently, we also hope that by separating the concerns of load-balancing from work processing within our abstraction, managing and extending existing code to future architectures becomes easier.

## Table of contents

- [GitHub actions status.](#wrenchgithub-actions-status)
- [Getting started.](#rotating_light-getting-started)
- [Downloading all datasets.](#ledger-datasets)
- [Background information.](#musical_note-a-little-background)
  - [Where this project fits in and how?](#-a-small-and-important-piece-of-a-larger-puzzle)
  - [Load-balancing problem and a solution.](#%EF%B8%8F-load-balancing-problem-and-a-silver-lining)
- [GPU load-balancing abstraction.](#%EF%B8%8F-gpu-load-balancing-abstraction)
  - [As function and set notation.](#%EF%B8%8F-as-function-and-set-notation)
  - [As three domains: data, schedule and computation.](#-as-three-domains-data-schedule-and-computation)
- [Composable API: Load-balanced loops.](#composable-api-load-balanced-loops)
  - Define and configure load-balancing schedule.
  - Load-balanced ranged loops.
  - User-defined computation.

## :wrench:	GitHub actions status.

| System  | Version                                                                                                                                                    | CUDA   | Status                                                                                                                                                   |
|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ubuntu  | [Ubuntu 20.04](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources)        | 11.7.0 | [![Ubuntu](https://github.com/gunrock/loops/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/gunrock/loops/actions/workflows/ubuntu.yml)    |
| Windows | [Windows Server 2019](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources) | 11.7.0 | [![Windows](https://github.com/gunrock/loops/actions/workflows/windows.yml/badge.svg)](https://github.com/gunrock/loops/actions/workflows/windows.yml) |

## :rotating_light: Getting Started
Before building `loops` make sure you have CUDA Toolkit and cmake installed on your system. Other external dependencies such as `NVIDIA/thrust`, `NVIDIA/cub`, etc. are automatically fetched using cmake.

```bash
git clone https://github.com/gunrock/loops.git
cd loops
mkdir build && cd build
cmake .. 
make loops.spmv.merge_path # or use: make -j$(nproc)
bin/loops.spmv.merge_path -m ../datasets/chesapeake/chesapeake.mtx
```

## :ledger: Datasets

To download the SuiteSparse Matrix Collection[^1], simply run the following command. We recommend using a `tmux` session, because downloading the entire collection can take a significant time.
```bash
wget --recursive --no-parent --force-directories -l inf -X RB,mat --accept "*.tar.gz" "https://suitesparse-collection-website.herokuapp.com/"
```

- `--recursive` recursively download
- `--no-parent` prevent wget from starting to fetch links in the parent of the website
- `--l inf` keep downloading for an infinite level
- `-X RB,mat` ignore subdirectories RB and mat, since I am only downloading matrix market MM, you can choose to download any of the others or remove this entirely to download all formats
- `--accept` accept the following extension only
- `--force-directories` create a hierarchy of directories, even if one would not have been created otherwise

[^1]: Timothy A. Davis and Yifan Hu. 2011. The University of Florida Sparse Matrix Collection. ACM Transactions on Mathematical Software 38, 1, Article 1 (December 2011), 25 pages. DOI: https://doi.org/10.1145/2049662.2049663

# :musical_note: A little background.
**DARPA** announced [**Software Defined Hardware (SDH)**](https://www.darpa.mil/program/software-defined-hardware)[^2], a program that aims "*to build runtime-reconfigurable hardware and software that enables near ASIC performance without sacrificing programmability for data-intensive algorithms.*" **NVIDIA** leading the charge on the program, internally called, [**Symphony**](https://blogs.nvidia.com/blog/2018/07/24/darpa-research-post-moores-law/). Our work is a small but important piece of this larger puzzle. The "data-intensive algorithms" part of the program includes domains like Machine Learning, Graph Processing, Sparse-Matrix-Vector algorithms, etc. where there is a large amount of data available to be processed. And the problems being addressed are either already based on irregular data structures and workloads, or are trending towards it (such as sparse machine learning problems). For these irregular workload computations to be successful, we require efficient load-balancing schemes targetting specialized hardware such as the GPUs or Symphony.

[^2]: [DARPA Selects Teams to Unleash Power of Specialized, Reconfigurable Computing Hardware](https://www.darpa.mil/news-events/2018-07-24a)

## 🧩 A small (and important) piece of a larger puzzle.
The predominant approach today to addressing irregularity is to build application-dependent solutions. These are not portable between applications. This is a shame because We believe the underlying techniques that are currently used to address irregularity have the potential to be expressed in a generic, portable, powerful way. We build a generic open-source library for load balancing that will expose high-performance, intuitive load-balancing strategies to any irregular-parallel application.

## ⚖️ Load-balancing problem, and a silver lining.
Today's GPUs follow a Single Instruction Multiple Data (SIMD) model, where different work components (for example a node in a graph) are mapped to a single thread. Each thread runs a copy of the program and threads run in parallel (this is a simple explanation, there are other work units in NVIDIA's GPUs such as warps, cooperative thread arrays, streaming multiprocessors etc.). Let's take a graph problem as an example to understand load imbalance. One key operation in graph problems is traversal, given a set of vertices, a traversal operation visits all the neighboring vertices of the input. If we naïvely map each input vertex to a GPU thread it can result in a massive imbalance of work. As some threads within the GPU will get a lot more work than others, causing inefficient utilization of hardware resources. In our example; this could happen for a social-network graph where one input vertex may have millions of connections while other input vertices in the same traversal pass may only have tens of neighbors.

The silver lining here is that there are more intelligent workload mappings that address this problem the load imbalance problem for various types of graphs and other irregular workloads. We extend these previously tightly-coupled scheduling algorithms to an abstraction.

# ♻️ GPU load-balancing abstraction.

The simple idea behind our load-balancing abstraction is to represent sparse formats as atoms, tiles and set functional abstraction elements described in the "Function and Set Notation" below. Once represented as such, we can develop load-balancing algorithms that create balanced ranges of atoms and tiles and map them to processor ids. This information can be abstracted to the user with a simple API (such as ranged-for-loops) to capture user-defined computations. Some benefits of this approach are: (1) the user-defined computation remains largely the same for many different static or dynamic load-balancing schedules, (2) these schedules can now be extended to other computations and (3) dramatically reduces code complexity.

## ✒️ As function and set notation.

Given a sparse-irregular problem $S$ made of many subsets called tiles, $T$. $T_i$ is defined as a collection of atoms, where an atom is the smallest possible processing element (for example, a nonzero element within a sparse-matrix). Using a scheduler, our abstraction's goal is to create a new set, $M$, which maps the processor ids (thread ids for a given kernel execution) $P_{id}$ to a group of subsets of $T$: $M = \{ P_{id}, T_i ... T_j \}$, map of processor ids to tiles, and the scheduler responsible for creating the maps: $L(S) = \{ M_0, ..., M_m\}$.

## 🧫 As three domains: data, schedule and computation.
![illustration](https://user-images.githubusercontent.com/9790745/168728299-6b125b44-894a-49bb-92fd-ee85aaa80ae4.png)

## Composable API: Load-balanced loops.

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
