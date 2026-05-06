# 🐧 `loops`: Expressing Parallel Irregular Computations

[![build](https://github.com/gunrock/loops/actions/workflows/build.yml/badge.svg)](https://github.com/gunrock/loops/actions/workflows/build.yml) [![clang-format](https://github.com/gunrock/loops/actions/workflows/clang-format.yml/badge.svg)](https://github.com/gunrock/loops/actions/workflows/clang-format.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7465053.svg)](https://doi.org/10.5281/zenodo.7465053)

We propose an open-source GPU load-balancing framework for applications that exhibit irregular parallelism. The set of applications and algorithms we consider are fundamental to computing tasks ranging from sparse machine learning, large numerical simulations, and on through to graph analytics. The underlying data and data structures that drive these applications present access patterns that naturally don't map well to the GPU's architecture that is designed with dense and regular patterns in mind. 

Prior to the work we present and propose here, the only way to unleash the GPU's full power on these problems has been to workload balance through tightly coupled load-balancing techniques. Our proposed load-balancing abstraction decouples load balancing from work processing and aims to support both static and dynamic schedules with a programmable interface to implement new load-balancing schedules in the future. 

With our open-source framework, we hope to not only improve programmers' productivity when developing irregular-parallel algorithms on the GPU but also improve the overall performance characteristics for such applications by allowing a quick path to experimentation with a variety of existing load-balancing techniques. Consequently, we also hope that by separating the concerns of load-balancing from work processing within our abstraction, managing and extending existing code to future architectures becomes easier.

## Requirements

- **OS:** Linux (Ubuntu 22.04 / 24.04 tested) or Windows.
- **Hardware:** NVIDIA GPU with compute capability ≥ 7.0 (Volta or newer).
- **Software:** CUDA Toolkit ≥ 11.7 and CMake ≥ 3.24.
- **CUDA architecture:** Auto-detected by default (`CMAKE_CUDA_ARCHITECTURES=native`). Override at configure time, e.g. `-DCMAKE_CUDA_ARCHITECTURES=90` for an H100-only build, or `"70;80;90"` for a fat binary.

`loops` is a header-only library. Thrust, CUB and libcu++ ship with the CUDA Toolkit and are picked up automatically — no separate fetch step is required. Only `cxxopts` (CLI parsing) is fetched as an external dependency.

## Quick Start

```bash
git clone https://github.com/gunrock/loops.git
cd loops

# Auto-detect the GPU(s) on this host (recommended default).
cmake --preset release-native
cmake --build --preset release-native -j

# Sanity check on the bundled chesapeake matrix.
./build/release-native/bin/loops.spmv.merge_path \
    -m datasets/chesapeake/chesapeake.mtx --validate
```

Other configure presets (`release-h100`, `release-a100`, `release-multi`, `debug-native`, `release-with-tests`, `ci-multi-arch`), CMake-presets-free fallback, individual example targets, and Docker setup are all covered in [docs/build.md](docs/build.md).

## Format-Generic Schedules

The four scheduling algorithms (`thread_mapped`, `group_mapped`, `work_oriented`, `merge_path_flat`) talk to the workload through a small **layout view** contract rather than directly poking at CSR offset arrays. Any struct that exposes the contract — `num_tiles()`, `num_atoms()`, `tile_begin(t)`, `tile_end(t)`, `tile_size(t)`, `tile_end_iter()` — can drive any of the schedules without modification.

In-tree layouts:

- `loops::layout::csr<tile_id_t, atom_id_t>` — backed by a row-offset prefix-sum array (the default).
- `loops::layout::ell<tile_id_t, atom_id_t>` — uniform pitch per row, no offsets array; `tile_end_iter()` is a `thrust::transform_iterator` synthesized on the fly.

To plug in your own format, write a struct satisfying the contract documented in [`include/loops/container/layout.hxx`](include/loops/container/layout.hxx) and pass it as the trailing template argument to `schedule::setup<...>`. A worked example lives in [`examples/spmv/custom_layout.cu`](examples/spmv/custom_layout.cu).

### Tile partitioners

A *partitioner* is a layout adaptor that re-bins atoms into a different tile grouping while still satisfying the same contract — so the schedules continue to drive it unchanged. In-tree:

- `loops::layout::flat_uniform_occupancy<K, base_layout_t>` — flatten the base layout's atoms and chunk them into tiles of `K` atoms each (last tile may be smaller). Tiles can cross the base layout's natural boundaries (e.g., CSR rows), so kernels that need per-atom output addressing should atomic-add via `partitioner.base().tile_of(atom)`.

A worked SpMV example using `flat_uniform_occupancy<8, csr>` lives in [`examples/spmv/flat_partitioned.cu`](examples/spmv/flat_partitioned.cu); it shares the standard `thread_mapped` schedule with no schedule-side changes.

## Documentation

Long-form documentation lives in [`docs/`](docs/):

- [Building](docs/build.md) — full CMake-presets table, CUDA-architecture overrides, optional dependencies, and Docker.
- [Datasets](docs/datasets.md) — fetching the SuiteSparse Matrix Collection.
- [Experimentation](docs/experimentation.md) — running the bundled examples and the sanity check.
- [Reproducing Results](docs/reproducing-results.md) — re-running the paper's full experiment sweep and regenerating the plots.
- [Abstraction](docs/abstraction.md), [Background](docs/background.md), and [Load-Balancing API](docs/loadbalancing_api.md) — design notes on the underlying model.

## How to Cite Loops
Thank you for citing our work.

```bibtex
@inproceedings{Osama:2023:APM,
  author       = {Muhammad Osama and Serban D. Porumbescu and John D. Owens},
  title        = {A Programming Model for {GPU} Load Balancing},
  booktitle    = {Proceedings of the 28th ACM SIGPLAN Symposium on
                  Principles and Practice of Parallel Programming},
  series       = {PPoPP 2023},
  year         = 2023,
  month        = feb # "\slash " # mar,
  acceptance   = {31 of 131 submissions, 23.7\%},
  code         = {https://github.com/gunrock/loops},
  doi          = {10.1145/3572848.3577434},
}
```

```bibtex
@software{Osama:2022:LAP:Code,
  author       = {Muhammad Osama and Serban D. Porumbescu and John D. Owens},
  title        = {Loops: A Programming Model for GPU Load Balancing},
  month        = dec,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v0.1.0-alpha},
  doi          = {10.5281/zenodo.7465053},
  url          = {https://doi.org/10.5281/zenodo.7465053}
}
```
