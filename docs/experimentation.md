[README](/README.md) > **Experimentation**

# Experimentation

If CUDA and CMake are already set up, follow the [Getting Started](/README.md#getting-started) instructions in the top-level README. If you'd rather work from a container, the project ships a Docker setup; instructions live in [`/docker`](../docker).

## Sanity Check

After a successful build, run any of the SpMV examples on the bundled chesapeake matrix:

```bash
./build/release-native/bin/loops.spmv.merge_path \
    -m datasets/chesapeake/chesapeake.mtx --validate -v
```

You should see something close to:

```text
Elapsed (ms):   0.0XX
Matrix:         chesapeake.mtx
Dimensions:     39 x 39 (340)
Errors:         0
```

`Errors: 0` is the only number that needs to match exactly — the elapsed time naturally varies with hardware. Repeat with the other example binaries (`thread_mapped`, `group_mapped`, `work_oriented`, `merge_path`, `ell_thread_mapped`, `ell_merge_path`, `custom_layout`, `flat_partitioned`) to confirm every schedule and layout combination is healthy on your GPU.

## What to do next

- For larger inputs and full benchmarks, see [Datasets](datasets.md) for how to fetch SuiteSparse.
- To re-run the paper's experiments, see [Reproducing Results](reproducing-results.md).
- To explore the abstraction, see [Abstraction](abstraction.md), [Background](background.md), and [Load-Balancing API](loadbalancing_api.md).
