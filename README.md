# 🐧 `loops`: Expressing Parallel Irregular Computations

[![ubuntu-20.04](https://github.com/gunrock/loops/actions/workflows/ubuntu-20.04.yml/badge.svg)](https://github.com/gunrock/loops/actions/workflows/ubuntu-20.04.yml) [![ubuntu-22.04](https://github.com/gunrock/loops/actions/workflows/ubuntu-22.04.yml/badge.svg)](https://github.com/gunrock/loops/actions/workflows/ubuntu-22.04.yml) [![windows-2019](https://github.com/gunrock/loops/actions/workflows/windows-2019.yml/badge.svg)](https://github.com/gunrock/loops/actions/workflows/windows-2019.yml) [![windows-2022](https://github.com/gunrock/loops/actions/workflows/windows-2022.yml/badge.svg)](https://github.com/gunrock/loops/actions/workflows/windows-2022.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7465053.svg)](https://doi.org/10.5281/zenodo.7465053)

We propose an open-source GPU load-balancing framework for applications that exhibit irregular parallelism. The set of applications and algorithms we consider are fundamental to computing tasks ranging from sparse machine learning, large numerical simulations, and on through to graph analytics. The underlying data and data structures that drive these applications present access patterns that naturally don't map well to the GPU's architecture that is designed with dense and regular patterns in mind. 

Prior to the work we present and propose here, the only way to unleash the GPU's full power on these problems has been to workload balance through tightly coupled load-balancing techniques. Our proposed load-balancing abstraction decouples load balancing from work processing and aims to support both static and dynamic schedules with a programmable interface to implement new load-balancing schedules in the future. 

With our open-source framework, we hope to not only improve programmers' productivity when developing irregular-parallel algorithms on the GPU but also improve the overall performance characteristics for such applications by allowing a quick path to experimentation with a variety of existing load-balancing techniques. Consequently, we also hope that by separating the concerns of load-balancing from work processing within our abstraction, managing and extending existing code to future architectures becomes easier.

## Requirements
- **OS:** Ubuntu 18.04, 20.04, Windows
- **Hardware:** NVIDIA GPU (Volta or newer)
- **Software:** CUDA 11.7 or above and cmake 3.20.1 or above.
- **CUDA Architecture:** SM 70 or above (see [GPUs supported](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)), this is specified using cmake's command: `-DCMAKE_CUDA_ARCHITECTURES=70`. Alternatively, set the CUDA architecture version in the `CMakeLists.txt` file directly: [CMakeLists.txt#72](https://github.com/gunrock/loops/blob/main/CMakeLists.txt#L72).

## Getting Started
Before building `loops` make sure you have CUDA Toolkit and cmake installed on your system, and exported in `PATH` of your system. Other external dependencies such as `NVIDIA/thrust`, `NVIDIA/cub`, etc. are automatically fetched using cmake.

```bash
git clone https://github.com/gunrock/loops.git
cd loops
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=70 .. # Volta = 70, Turing = 75, ...
make -j$(nproc)
bin/loops.spmv.merge_path -m ../datasets/chesapeake/chesapeake.mtx
```

### Building Specific Algorithms

```bash
make loops.spmv.<algorithm>
```
Replaced the `<algorithm>` with one of the following algorithm names to build a specific SpMV algorithm instead of all of them:
- `original`
- `thread_mapped`
- `group_mapped`
- `work_oriented`
- `merge_path`

An example of the above: `make loops.spmv.merge_path`.

## Datasets

To download the SuiteSparse Matrix Collection[^1], simply run the following command. We recommend using a `tmux` session, because downloading the entire collection can take a significant time. Uncompress the dataset by running the following command in the dataset's directory `find . -name '*.tar.gz' -execdir tar -xzvf '{}' \;
` The total downloaded size of the dataset is nontrivial: uncompressed + compressed = 887GB.
```bash
wget --recursive --no-parent --force-directories -l inf -X RB,mat \ 
--accept "*.tar.gz" "https://suitesparse-collection-website.herokuapp.com/"
```

- `--recursive` recursively download
- `--no-parent` prevent wget from starting to fetch links in the parent of the website
- `--l inf` keep downloading for an infinite level
- `-X RB,mat` ignore subdirectories RB and mat, since I am only downloading matrix market MM, you can choose to download any of the others or remove this entirely to download all formats
- `--accept` accept the following extension only
- `--force-directories` create a hierarchy of directories, even if one would not have been created otherwise

[^1]: Timothy A. Davis and Yifan Hu. 2011. The University of Florida Sparse Matrix Collection. ACM Transactions on Mathematical Software 38, 1, Article 1 (December 2011), 25 pages. DOI: https://doi.org/10.1145/2049662.2049663

## Experimentation
If CUDA and cmake are already setup, follow the [Getting Started](#getting-started) instructions. Or, you may prefer to set up the entire project using docker, and for that we have provided a docker file and instructions on how to use it in [/docker](https://github.com/gunrock/loops/tree/main/docker) directory.

### Sanity Check
Run the following command in the cmake's `build` folder:
```bash
bin/loops.spmv.merge_path -m ../datasets/chesapeake/chesapeake.mtx \
 --validate -v
```
You should approximately see the following output:
```bash
~/loops/build$ bin/loops.spmv.merge_path \
 -m ../datasets/chesapeake/chesapeake.mtx --validate -v
Elapsed (ms):   0.063328
Matrix:         chesapeake.mtx
Dimensions:     39 x 39 (340)
Errors:         0
```
## Reproducing Results
> Find pre-generated results in [plots/](https://github.com/gunrock/loops/blob/main/plots/) directory along with `performance_evaluation.ipynb` notebook to recreate the plots (labeled figures) found in the paper.

1. In the run script, update the `DATASET_DIR` to point to the path of all the downloaded datasets (set to the path of the directory containing `MM` directory, and inside the `MM` it has subdirectories with `.mtx` files): [scripts/run.sh](https://github.com/gunrock/loops/blob/main/scripts/run.sh). Additionally, you may change the path to `DATASET_FILES_NAME` containing the list of all the datasets (default points to [datasets/suitesparse.txt](https://github.com/gunrock/loops/blob/main/datasets/suitesparse.txt)).
2. Fire up the complete run using `run.sh` found in `scripts` directory, `cd scripts && ./run.sh`, note one complete run can take up to 3 days (goes over the entire suitesparse matrix collection dataset four times with four different algorithms, the main bottleneck is loading files from disk.)
3. **Warning!** Some runs on the matrices are expected to fail as they are not in proper MatrixMarket Format although labeled as `.mtx`. These matrices and the ones that do not fit on the GPU will result in runtime exceptions or `offset_t` type overflow and can be safely ignored.
4. To run *N* number of datasets simply adjust the stop condition here (default set to `10`): [scripts/run.sh#L22](https://github.com/gunrock/loops/blob/main/scripts/run.sh#L22), or remove this if-condition entirely to run on all available `.mtx` files: [scripts/run.sh#L22-L26](https://github.com/gunrock/loops/blob/main/scripts/run.sh#L22-L26).

Expected output from the above runs are `csv` files in the same directory as the `run.sh`, these can replace the existing `csv` files within `plots/data`, and a [python jupyter notebook](https://jupyter.org/install) can be fired up to evaluate the results. Python notebook includes instructions on generating plots. See sample output of one of the `csv` files below:

```csv
kernel,dataset,rows,cols,nnzs,elapsed
merge-path,144,144649,144649,2148786,0.0720215
merge-path,08blocks,300,300,592,0.0170898
merge-path,1138_bus,1138,1138,4054,0.0200195
```

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
