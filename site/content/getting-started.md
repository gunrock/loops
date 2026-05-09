---
title: Getting Started
description: Quick start guide for the loops GPU load-balancing library.
---

# Getting Started

## Requirements

- **OS:** Linux (Ubuntu 22.04 / 24.04 tested) or Windows
- **Hardware:** NVIDIA GPU with compute capability >= 7.0 (Volta or newer)
- **Software:** CUDA Toolkit >= 11.7 and CMake >= 3.24
- **CUDA architecture:** Auto-detected by default (`CMAKE_CUDA_ARCHITECTURES=native`)

`loops` is a header-only library. Thrust, CUB and libcu++ ship with the CUDA Toolkit and are picked up automatically.

## Clone & Build

```bash
git clone https://github.com/gunrock/loops.git
cd loops

# Auto-detect the GPU(s) on this host.
cmake --preset release-native
cmake --build --preset release-native -j
```

## Sanity Check

```bash
./build/release-native/bin/loops.spmv.merge_path.f32 \
    -m datasets/chesapeake/chesapeake.mtx --validate
```

You should see `Errors: 0` — the elapsed time varies with hardware.

## Build Presets

| Preset | Description |
| --- | --- |
| `release-native` | Auto-detect host GPU (recommended default) |
| `release-h100` | H100 (sm_90) |
| `release-a100` | A100 (sm_80) |
| `release-multi` | Fat binary: sm_70, sm_80, sm_90 |
| `debug-native` | Debug mode, auto-detect GPU |
| `release-with-tests` | Release + unit tests |
| `ci-multi-arch` | For GPU-less CI runners |

## Next Steps

- [Abstraction](/docs/concepts/abstraction/) — understand tiles, atoms, and schedules
- [Schedules](/docs/concepts/schedules/) — the four scheduling algorithms
- [API Reference](/api/) — full type and function documentation
