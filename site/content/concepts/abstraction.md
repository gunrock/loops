---
title: Abstraction
description: The tile-atom abstraction that powers loops' format-generic load balancing.
---

# The Tile-Atom Abstraction

## The Core Idea

Every irregular workload can be described as a set of **tiles** and **atoms**:

- **Tile:** A logical grouping of work (a row in CSR, a column in CSC, a block-row in BCSR).
- **Atom:** The smallest unit of processing (a single nonzero element in a sparse matrix).

A **schedule** creates a mapping *M* from GPU processor IDs to balanced groups of tiles and atoms. The user writes their computation against this mapping using range-based loops.

## Three Domains

The abstraction separates every irregular computation into three independent concerns:

### 1. Data

Choose your sparse format. Each format has a corresponding **layout view** that describes how tiles and atoms are organized:

| Format | Container | Layout | Tile | Atom |
| --- | --- | --- | --- | --- |
| CSR | `csr_t` | `layout::csr` | row | nonzero |
| CSC | `csc_t` | `layout::csc` | column | nonzero |
| COO | `coo_t` | `layout::coo` | nonzero | nonzero |
| ELL | `ell_t` | `layout::ell` | row | bucketed nz |
| BCSR | `bcsr_t` | `layout::bcsr` | block-row | R x C block |
| DIA | `dia_t` | `layout::dia` | row | diagonal cell |

### 2. Schedule

Choose a load-balancing strategy. All four strategies consume workloads through the same layout contract:

- **thread_mapped** — one tile per thread, simplest possible mapping
- **group_mapped** — one tile per warp/group for high-degree tiles
- **work_oriented** — distribute atoms evenly across threads
- **merge_path** — optimal O(n+m) merge-based partitioning

### 3. Computation

Write your kernel logic once. It works unchanged across all schedules and all formats:

```cpp
for (auto row : config.tiles()) {
  type_t sum = 0;
  for (auto nz : config.atoms(row)) {
    sum += values[nz] * x[indices[nz]];
  }
  y[row] = sum;
}
```

## The Layout Contract

Any struct that exposes these methods can drive any schedule:

```cpp
struct my_layout {
  tile_id_t num_tiles() const;
  atom_id_t num_atoms() const;
  atom_id_t tile_begin(tile_id_t t) const;
  atom_id_t tile_end(tile_id_t t) const;
  atom_id_t tile_size(tile_id_t t) const;
  tile_end_iterator_t tile_end_iter() const;
};
```

## Formal Notation

Given a sparse-irregular problem *S* made of tiles *T*, where tile *T_i* is a collection of atoms: the scheduler creates a map *M = { P_id, T_i ... T_j }* from processor IDs to groups of tiles. The scheduler function *L(S) = { M_0, ..., M_m }* produces the complete balanced mapping.
