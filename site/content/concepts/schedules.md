---
title: Schedules
description: The four load-balancing scheduling algorithms in loops.
---

# Scheduling Algorithms

`loops` provides four static load-balancing schedules. All four consume workloads through the same layout contract, so switching between them requires changing only the `schedule::setup<>` template parameter.

## thread_mapped

The simplest schedule: one tile per thread.

```cpp
using setup_t = schedule::setup<
    schedule::algorithms_t::thread_mapped,
    1, 1, index_t, offset_t>;
```

Each thread iterates over all atoms in its assigned tile. Best for workloads where tiles have roughly uniform size. No shared memory or synchronization needed.

## group_mapped

One tile per cooperative group (typically a warp of 32 threads).

```cpp
using setup_t = schedule::setup<
    schedule::algorithms_t::group_mapped,
    BLOCK_SIZE, 32, index_t, offset_t>;
```

The group collaborates to process all atoms in a tile, with each thread taking a strided share. Effective when individual tiles are large enough to occupy a full warp.

## work_oriented

Distributes the total atom count evenly across all threads.

```cpp
using setup_t = schedule::setup<
    schedule::algorithms_t::work_oriented,
    128, 1, index_t, offset_t>;
```

Each thread processes a contiguous range of atoms, handling complete tiles within its range and using atomic operations for tiles that span thread boundaries. Good for skewed distributions where a few tiles dominate.

## merge_path_flat

Optimal merge-based partitioning in O(tiles + atoms) work.

```cpp
using setup_t = schedule::setup<
    schedule::algorithms_t::merge_path_flat,
    128, 4, index_t, offset_t>;
```

Uses a diagonal search on the merge-path of tiles and atoms to find the exact partition point for each thread block. Requires a preprocessing step (`preprocess_t`) to compute per-block starting coordinates. The most balanced of all schedules, but also the most complex.

## Choosing a Schedule

| Schedule | Best for | Overhead |
| --- | --- | --- |
| thread_mapped | Uniform tile sizes | Minimal |
| group_mapped | Large tiles, high degree | Low |
| work_oriented | Skewed distributions | Low |
| merge_path_flat | Any distribution (optimal) | Preprocessing step |

For most workloads, start with `thread_mapped` and move to `merge_path_flat` if you observe load imbalance.
