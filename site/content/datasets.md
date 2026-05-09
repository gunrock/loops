---
title: Datasets
description: How to fetch and prepare the SuiteSparse Matrix Collection for benchmarking.
---

# Datasets

`loops` ships with a single bundled matrix, [`datasets/chesapeake/chesapeake.mtx`](../datasets/chesapeake), so all examples can be run as smoke tests right out of the box. For benchmarking and the paper's experiments we use the SuiteSparse Matrix Collection.[^1]

## Downloading SuiteSparse

The full collection is large enough that a download takes a long time and a lot of disk; we recommend running the command below inside a `tmux` session.

```bash
wget --recursive --no-parent --force-directories -l inf -X RB,mat \
     --accept "*.tar.gz" \
     "https://suitesparse-collection-website.herokuapp.com/"
```

| Flag | Purpose |
|---|---|
| `--recursive`         | Recursively follow links |
| `--no-parent`         | Don't fetch links above the starting URL |
| `-l inf`              | No depth limit |
| `-X RB,mat`           | Skip the `RB` and `mat` subdirs (we only want MatrixMarket) |
| `--accept "*.tar.gz"` | Only download tarballs |
| `--force-directories` | Preserve the original site hierarchy |

Total downloaded size: ~887 GB (uncompressed + compressed).

## Uncompressing

After the wget completes, decompress every tarball in place:

```bash
find . -name '*.tar.gz' -execdir tar -xzvf '{}' \;
```

## Picking a subset

The list of matrices used by the paper's runs lives in [`datasets/suitesparse.txt`](../datasets/suitesparse.txt). When invoking [`scripts/run.sh`](../scripts/run.sh) you can either:

- point `DATASET_FILES_NAME` at this list to reproduce the paper's experiment set, or
- supply your own newline-separated list to run on a custom subset.

[^1]: Timothy A. Davis and Yifan Hu. 2011. The University of Florida Sparse Matrix Collection. *ACM Transactions on Mathematical Software* 38, 1, Article 1 (December 2011), 25 pages. DOI: <https://doi.org/10.1145/2049662.2049663>.
