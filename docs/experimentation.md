[README](/README.md) > **Experimentation**

# Experimentation

If CUDA and CMake are already set up, follow the [Getting Started](/README.md#getting-started) instructions in the top-level README. If you'd rather work from a container, the project ships a Docker setup; instructions live in [`/docker`](../docker).

## Precisions

Every SpMV example is built twice, once per element type:

- `loops.spmv.<kernel>.f32` — `float`
- `loops.spmv.<kernel>.f64` — `double`

The CMake plumbing lives in [`examples/spmv/CMakeLists.txt`](../examples/spmv/CMakeLists.txt) and resolves `using type_t = LOOPS_VALUE_T;` per-target, so adding a new precision (e.g. `__half`) is a one-line edit to the `LOOPS_SPMV_PRECISIONS` list.

## Sanity Check

After a successful build, run any of the SpMV examples on the bundled chesapeake matrix:

```bash
./build/release-native/bin/loops.spmv.merge_path.f32 \
    -m datasets/chesapeake/chesapeake.mtx --validate -v
```

You should see something close to:

```text
Elapsed (ms):   0.0XX
Matrix:         chesapeake.mtx
Dimensions:     39 x 39 (340)
Errors:         0
```

`Errors: 0` is the only number that needs to match exactly — the elapsed time naturally varies with hardware. Repeat with the other example binaries (`thread_mapped`, `group_mapped`, `work_oriented`, `merge_path`, `ell_thread_mapped`, `ell_merge_path`, `custom_layout`, `flat_partitioned`, `coo_thread_mapped`, `csc_thread_mapped`, `bcsr_thread_mapped`, `dia_thread_mapped`) and/or the `.f64` variant to confirm every schedule x layout x precision combination is healthy on your GPU.

## Rigorous Validation

`--validate` only counts naive `|y_gpu - y_ref| > tolerance` mismatches. On large or ill-conditioned matrices (`cant`, `scircuit`, hub-heavy graph matrices) float32 SpMV legitimately accumulates round-off larger than any fixed tolerance — a non-zero `Errors` count there is **not** a bug, just float arithmetic.

To distinguish "real bug" from "expected float round-off", pass `--rigorous`:

```bash
./build/release-native/bin/loops.spmv.thread_mapped.f32 \
    -m datasets/cant/cant.mtx --rigorous
```

This recomputes the reference with double-precision accumulation and compares the GPU output against a per-row [Wilkinson](https://en.wikipedia.org/wiki/James_H._Wilkinson) bound `K * nnz_row * eps * row_L1`. The output adds:

```text
WilkinsonK:           8
NaiveMismatches:      7
F32BaselineOverruns:  7
GPUOverruns:          0
MaxAbsError:          0.025
MaxRelError:          0.0024
Verdict:              NOT_A_BUG
```

`GPUOverruns == 0` means every row's disagreement against the f64 reference is bounded by what valid float32 summation can produce — i.e. the kernel is correct, even if the naive `--validate` count is non-zero.

## What to do next

- For larger inputs and full benchmarks, see [Datasets](datasets.md) for how to fetch SuiteSparse.
- To re-run the paper's experiments, see [Reproducing Results](reproducing-results.md).
- To explore the abstraction, see [Abstraction](abstraction.md), [Background](background.md), and [Load-Balancing API](loadbalancing_api.md).
