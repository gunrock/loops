---
title: Building
description: Build instructions for loops — CMake presets, CUDA architecture selection, optional dependencies, and Docker.
---

# Building

`loops` is header-only; the only thing you build are the example/benchmark/test binaries that exercise the headers. The repository ships a `CMakePresets.json` so most users never need to type a raw `-D` flag.

## Available configure presets

| Preset | Architectures | Use when |
|---|---|---|
| `release-native`     | Host's GPU(s)        | Local development on a single machine |
| `release-h100`       | sm_90                | H100 nodes |
| `release-a100`       | sm_80                | A100 nodes |
| `release-multi`      | sm_70…sm_90          | Distributing a fat binary |
| `debug-native`       | Host's GPU(s)        | Debug build with `-G -lineinfo` |
| `release-with-tests` | Host's GPU(s)        | Build with unit tests and benchmarks enabled |
| `ci-multi-arch`      | sm_80;sm_90          | CI hosts without a GPU (CUDA 13+ compatible) |

Configure and build with any of them:

```bash
cmake --preset release-h100
cmake --build --preset release-h100 -j
```

The output binaries land in `build/<preset>/bin/`.

## Picking a CUDA architecture

The `release-native` preset sets `CMAKE_CUDA_ARCHITECTURES=native`, so CMake auto-detects the GPU(s) on the host at configure time. To override for cross-compilation or fat-binary builds, pass it explicitly:

```bash
# H100-only build
cmake --preset release-native -DCMAKE_CUDA_ARCHITECTURES=90

# Fat binary covering Volta through Hopper
cmake --preset release-native -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90"
```

Note that CUDA 13.0 dropped `sm_70` (Volta); use `"80;90"` or higher there.

## Without CMake presets

If your CMake is older than 3.24, the presets are unavailable. Configure the old-fashioned way:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Building specific examples

Each `.cu` file under `examples/` becomes its own target named `loops.<group>.<name>` (for example, the SpMV thread-mapped example is `loops.spmv.thread_mapped`). To build just one:

```bash
cmake --build --preset release-native --target loops.spmv.merge_path
```

Available SpMV example targets:

- `loops.spmv.original` — cuSPARSE reference
- `loops.spmv.thread_mapped`, `loops.spmv.group_mapped`, `loops.spmv.work_oriented`, `loops.spmv.merge_path` — CSR-backed schedules
- `loops.spmv.ell_thread_mapped`, `loops.spmv.ell_merge_path` — same schedules driving an ELL layout
- `loops.spmv.custom_layout` — user-defined layout
- `loops.spmv.flat_partitioned` — `flat_uniform_occupancy<K, csr>` partitioner

Other groups: `loops.spmm.thread_mapped`, `loops.saxpy`, `loops.range`.

## Optional dependencies

| Knob | Default | Effect |
|---|---|---|
| `LOOPS_BUILD_TESTS`     | `OFF` | Build the unit tests under `unittests/`. |
| `LOOPS_BUILD_BENCHMARKS` | `OFF` | Build the NVBench-based benchmarks. |
| `LOOPS_USE_BUNDLED_CCCL` | `ON`  | Use the Thrust / CUB / libcu++ that ship with the CUDA Toolkit. Set to `OFF` to fetch the pinned NVIDIA/CCCL via `FetchContent` instead. |

The `release-with-tests` preset is the easiest way to flip the first two on:

```bash
cmake --preset release-with-tests
cmake --build --preset release-with-tests -j
ctest --preset release-with-tests
```

## Docker

A multi-stage `docker/Dockerfile` and matching `docker-compose.yml` ship in the repo root for users who'd rather build inside a container. See [`docker/`](../docker) for the supported `CUDA_VERSION` / `UBUNTU_VERSION` build-args and how to wire NVIDIA Container Toolkit into Compose.
