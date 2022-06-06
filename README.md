# 🐧 `loops`: Expressing Parallel Irregular Computations
We propose an open-source GPU load-balancing framework for applications that exhibit irregular parallelism. The set of applications and algorithms we consider are fundamental to computing tasks ranging from sparse machine learning, large numerical simulations, and on through to graph analytics. The underlying data and data structures that drive these applications present access patterns that naturally don't map well to the GPU's architecture that is designed with dense and regular patterns in mind. Prior to the work we present and propose here, the only way to unleash the GPU's full power on these problems has been to workload balance through tightly coupled load-balancing techniques. Our proposed load-balancing abstraction decouples load balancing from work processing and aims to support both static and dynamic schedules with a programmable interface to implement new load-balancing schedules in the future. With our open-source framework, we hope to not only improve programmers' productivity when developing irregular-parallel algorithms on the GPU but also improve the overall performance characteristics for such applications by allowing a quick path to experimentation with a variety of existing load-balancing techniques. Consequently, we also hope that by separating the concerns of load-balancing from work processing within our abstraction, managing and extending existing code to future architectures becomes easier.

| System  | Version                                                                                                                                                    | CUDA   | Status                                                                                                                                                   |
|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ubuntu  | [Ubuntu 20.04](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources)        | 11.7.0 | [![Ubuntu](https://github.com/neoblizz/loops/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/neoblizz/loops/actions/workflows/ubuntu.yml)    |
| Windows | [Windows Server 2019](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources) | 11.7.0 | [![Windows](https://github.com/neoblizz/loops/actions/workflows/windows.yml/badge.svg)](https://github.com/neoblizz/loops/actions/workflows/windows.yml) |

## :musical_note: A little background.
**DARPA** announced [**Software Defined Hardware (SDH)**](https://www.darpa.mil/program/software-defined-hardware), a program that aims "*to build runtime-reconfigurable hardware and software that enables near ASIC performance without sacrificing programmability for data-intensive algorithms.*" **NVIDIA** leading the charge on the program, internally called, [**Symphony**](https://blogs.nvidia.com/blog/2018/07/24/darpa-research-post-moores-law/). Our work is a small but important piece of this larger puzzle. The "data-intensive algorithms" part of the program includes domains like Machine Learning, Graph Processing, Sparse-Matrix-Vector algorithms, etc. where there is a large amount of data available to be processed. And the problems being addressed are either already based on irregular data structures and workloads, or are trending towards it (such as sparse machine learning problems). For these irregular workload computations to be successful, we require efficient load-balancing schemes targetting specialized hardware such as the GPUs or Symphony.
- [DARPA Selects Teams to Unleash Power of Specialized, Reconfigurable Computing Hardware](https://www.darpa.mil/news-events/2018-07-24a)

## 🧩 A small (and important) piece of a larger puzzle.
The predominant approach today to addressing irregularity is to build application-dependent solutions. These are not portable between applications. This is a shame because We believe the underlying techniques that are currently used to address irregularity have the potential to be expressed in a generic, portable, powerful way. We build a generic open-source library for load balancing that will expose high-performance, intuitive load-balancing strategies to any irregular-parallel application.

## ⚖️ Load-balancing problem, and a silver lining.
Today's GPUs follow a Single Instruction Multiple Data (SIMD) model, where different work components (for example a node in a graph) are mapped to a single thread. Each thread runs a copy of the program and threads run in parallel (this is a simple explanation, there are other work units in NVIDIA's GPUs such as warps, cooperative thread arrays, streaming multiprocessors etc.). Let's take a graph problem as an example to understand load imbalance. One key operation in graph problems is traversal, given a set of vertices, a traversal operation visits all the neighboring vertices of the input. If we naïvely map each input vertex to a GPU thread it can result in a massive imbalance of work. As some threads within the GPU will get a lot more work than others, causing inefficient utilization of hardware resources. In our example; this could happen for a social-network graph where one input vertex may have millions of connections while other input vertices in the same traversal pass may only have tens of neighbors.

The silver lining here is that there are more intelligent workload mappings that address this problem the load imbalance problem for various types of graphs and other irregular workloads. We extend these previously tightly-coupled scheduling algorithms to an abstraction.

# GPU Load-balancing Abstraction

![image](https://user-images.githubusercontent.com/9790745/168728352-27758e82-5f37-46cd-8052-99ca571edbfa.png)
![illustration](https://user-images.githubusercontent.com/9790745/168728299-6b125b44-894a-49bb-92fd-ee85aaa80ae4.png)

We provide two APIs for our library, one that focuses on a beginner-friendly approach to load-balancing irregular sparse computations and another that allows advanced programmers to retain control of the GPU kernels and express load-balanced execution as ranged loops. Both approaches are highlighted below.

## Beginner API: Load-balanced Transformations and Primitives

Our Load-balanced execution API builds on the approach defined in `gunrock/essentials` where we identify key primitives used in computing sparse linear algebra, graph analytics, and other irregular computations alike. Load-balanced versions of these primitives are then implemented, such that the user gets access to the atom, tile, and processor id they are working on as the [C++ lambda](https://en.cppreference.com/w/cpp/language/lambda) signature.

Users define their computation within the C++ lambda, which gets called by the load-balanced primitive for every instance of the work atom.

### (1) Sparse Layout
In this simple example we are using Compressed Sparse Row (CSR) format and simply returning the number of `atoms` (nonzeros) in each row as our layout.
```cpp
auto layout = [=] __device__ (std::size_t tile_id) {
  return offsets[tile_id+1] – offsets[tile_id];
}
```

### (2) User-defined Compute (C++ Lambda)

```cpp
// user-defined compute: y = Ax
auto spmv = [=] __device__ (std::size_t atom_id, std::size_t tile_id, std::size_t proc_id) {
  return values[atom_id] * 	x[column_indices[atom_id]];
}
```

### (3) Load-balanced primitive (Transform Segmented Reduce)
Requires the load-balancing schedule (`work_oriented` in this example) as a templated parameter.
The transformation as a C++ lambda expressions (`spmv`) and the `layout` as an input to perform a segmented reduction.
The output of the C++ lambda expression gets reduced by segments defined using `A.offsets`.
```cpp
lb::transform_segreduce<lb::work_oriented>
                       (spmv, layout, A.nonzeros, A.offsets,
                        G.rows, y, lb::plus_t(),
                        0.0f, stream);
```

#### Advantage:
- Requires no knowledge of how to implement segmented reduction,
- Very simple API if the computation can be defined using C++ lambda expressions.

#### Disadvantages:
- No control over kernel execution and dispatch configuration,
- No composability; cannot implement more complicated computations that may have cooperative properties among processors.

## Advanced API: Load-balanced Loops

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

### (1) Define and Configure Load-balancing Schedule
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

### (2) Load-balanced Ranged Loops (see; [C++ ranges](https://en.cppreference.com/w/cpp/header/ranges))
In this example, we define two iteration spaces; virtual and real. Virtual spaces allow us to balance atoms and tiles onto the processor ids and link directly to the real iteration space, which returns the exact atom or tile being processed. The code below loops over all balanced number of atoms fetches the tile corresponding to the atom being processed and allows user to define their computation.
```cpp
  for (auto virtual_atom : config.atom_accessor(p)) {
    auto virtual_tile = config.tile_accessor(virtual_atom, p);

    if (!(config.is_valid_accessor(virtual_tile, p)))
      continue;

    auto row = config.tile_id(virtual_tile, p);

    auto nz_idx = config.atom_id(virtual_atom, row, virtual_tile, p);
```

### (3) User-defined Computation
Once the user has access to the atom, tile, and the processor id, they implement the desired computation on the given tuple. In this example, we use a simple `atomicAdd` to perform SpMV (can be improved).
```cpp
    atomicAdd(&(y[row]), values[nz_idx] * x[indices[nz_idx]]);
  }
}
```