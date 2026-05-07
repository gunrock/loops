[README](/README.md) > **Reproducing Results**

# Reproducing Results

> Pre-generated CSVs and the `performance_evaluation.ipynb` notebook used to recreate every labeled figure in the paper live in [`plots/`](../plots/).

## End-to-end run

1. **Update the dataset path** in [`scripts/run.sh`](../scripts/run.sh):
   - Set `DATASET_DIR` to the directory containing the `MM` subdirectory (which itself holds per-matrix subdirectories of `.mtx` files). See [Datasets](datasets.md) for how to obtain it.
   - Optionally change `DATASET_FILES_NAME`; it defaults to [`datasets/suitesparse.txt`](../datasets/suitesparse.txt), the paper's matrix list.

2. **Kick off the run** from the `scripts/` directory:

   ```bash
   cd scripts && ./run.sh
   ```

   A complete run can take up to **~3 days** — it sweeps the entire SuiteSparse collection four times, once per algorithm. The dominant cost is `.mtx` parsing from disk, not GPU time.

3. **Expect occasional matrix-level failures.** Some files in SuiteSparse are mislabeled or otherwise malformed, and a few matrices are larger than the GPU can hold. These produce runtime exceptions or `offset_t` overflows and can be safely ignored — they don't affect the rest of the sweep.

4. **Run on a smaller subset** by editing the stop condition in [`scripts/run.sh#L22`](../scripts/run.sh#L22) (it caps at the first 10 matrices by default), or removing the conditional in [L22–L26](../scripts/run.sh#L22-L26) to process every `.mtx` file under `DATASET_DIR`.

## Output

Each algorithm writes a CSV alongside `run.sh`. Drop these into `plots/data/` to replace the bundled results, then open `plots/performance_evaluation.ipynb` (Jupyter — install with `pip install jupyter` or [following the official instructions](https://jupyter.org/install)) and re-run all cells to regenerate the figures.

A representative line of CSV output:

```csv
kernel,dataset,rows,cols,nnzs,elapsed
merge-path,144,144649,144649,2148786,0.0720215
merge-path,08blocks,300,300,592,0.0170898
merge-path,1138_bus,1138,1138,4054,0.0200195
```
