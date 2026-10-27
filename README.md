# cuPC, fork for CITS

This fork of [cuPC](https://github.com/LIS-Laboratory/cupc) (Zarebavani et al. 2020) is the GPU backend of [CITS](https://github.com/abbasilab/cits) from version 1.10.0. It keeps cuPC's original `Skeleton()` function and adds two changes.

1. **64-bit combination counters.** The original kernels count conditioning sets in 32-bit integers. The count overflows once a variable has many neighbours (from degree 1,627 at level 3 down to 31 at level 14). Some tests were then skipped, and edges that should have been removed survived.
2. **No cap on the conditioning-set size.** The original code stops at level 14. A new generic kernel tests any level, so the PC-stable search runs until no variable has enough neighbours. It computes partial correlations by an incremental Cholesky and applies cuPC's exact Fisher-z rule near the threshold. Every kernel launch is checked, and every removed edge is logged with its separating set.

The new kernel rebuilt 229 skeletons of a large calcium-imaging study exactly, including 1,690 edges removed above level 14. It ran about 460 times faster than the original kernels with a CPU continuation above level 14.

## Build

```
git clone https://github.com/biswasr/cupc
cd cupc
nvcc -O3 --shared -Xcompiler -fPIC -o Skeleton.so cuPC-S.cu
```

The build needs the CUDA toolkit and takes about 30 s. CITS finds `Skeleton.so` through `CUPC_DIR`, `cits.set_cupc_dir()` or `~/repos/cupc` (see the CITS README). It reads `cupc_cits_version()` to check that the library comes from this fork.

## C interface

`Skeleton()` is unchanged, so cuPC's R wrapper (`cuPC.R`) works as before. It gains the 64-bit fix but keeps the level-14 cap. The `ext_*` functions in `ext_levels.cuh` run one level per call, and CITS calls them from Python.

- `ext_create(C, n, has_old)` copies the correlation matrix to the GPU and returns a handle.
- `ext_level0(h, th0)` runs level 0 (marginal tests) as `Skeleton()` does.
- `ext_scan(h, rowsize, nprime)` compacts the adjacency and returns the row sizes and their maximum.
- `ext_level_new(h, l, th, ...)` tests every size-l subset with the generic kernel and logs each removal (X, Y, separating set, partial correlation).
- `ext_level_old(h, l, th)` runs the original kernel of level l from 1 to 14, for comparisons.
- `ext_get_G(h, G)` and `ext_set_G(h, G)` copy the adjacency matrix out and in.
- `ext_destroy(h)` frees the GPU memory.

Each returns 0 on success and a negative code on a CUDA error.

## License

GNU GPL version 3, as the original cuPC. The changes are listed at the top of `cuPC-S.cu` and `cuPC-S.h`. `ext_levels.cuh` is a new file.

---

*The original cuPC README follows.*

# cuPC
cuPC is a CUDA-based parallel implementation of PC-stable algorithm for causal structure learning on GPU. The main highlights of cuPC are as follows:
* Easy usage.
* Compatible with [pcalg](https://cran.r-project.org/web/packages/pcalg/index.html) software.
* 100X to 10,000X speedup compared to serial implementation on CPU.


# Installation
#### CUDA toolkit
To install CUDA toolkit please use [this link](https://developer.nvidia.com/cuda-downloads).

#### R
```
sudo echo "deb http://cran.rstudio.com/bin/linux/ubuntu xenial/" | sudo tee -a /etc/apt/sources.list
gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
gpg -a --export E084DAB9 | sudo apt-key add -

sudo apt update
sudo apt install r-base r-base-dev
```

#### Linux dependencies
```
sudo apt install libv8-3.14-dev
sudo apt install libcurl4-openssl-dev
sudo apt install libgmp3-dev
```

#### R dependencies
First, enter R by executing the following command:
```
sudo -i R
```

Now inside the R environment, run the following commands:

```
install.packages("tictoc")
source("http://bioconductor.org/biocLite.R")
biocLite(c("graph", "RBGL", "Rgraphviz"))
install.packages("pcalg")
```

#### Compile and execute

* Execute "nvcc -O3 --shared -Xcompiler -fPIC -o Skeleton.so cuPC-S.cu" to compile .cu files
* A test example exists in use_cuPC.R
* Data_generator.R create gaussian-distributed data

# Publication

Behrooz Zarebavani, Foad Jafarinejad, Matin Hashemi, Saber Salehkaleybar, [cuPC: CUDA-based Parallel PC Algorithm for Causal Structure Learning on GPU](https://ieeexplore.ieee.org/document/8823064), IEEE Transactions on Parallel and Distributed Systems (TPDS), Vol. 31, No. 3, March 2020.

#### Abstract
The main goal in many fields in the empirical sciences is to discover causal relationships among a set of variables from observational data. PC algorithm is one of the promising solutions to learn underlying causal structure by performing a number of conditional independence tests. In this paper, we propose a novel GPU-based parallel algorithm, called cuPC, to execute an order-independent version of PC. The proposed solution has two variants, cuPC-E and cuPC-S, which parallelize PC in two different ways for multivariate normal distribution. Experimental results show the scalability of the proposed algorithms with respect to the number of variables, the number of samples, and different graph densities. For instance, in one of the most challenging datasets, the runtime is reduced from more than 11 hours to about 4 seconds. On average, cuPC-E and cuPC-S achieve 500 X and 1300 X speedup, respectively, compared to serial implementation on CPU.

#### Original version of cuPC
The original source code which was employed in the above published article is available at our lab webpage [here](http://lis.ee.sharif.edu/pub/cupc/).

#### Citation
Please cite cuPC in your publications if it helps your research:
```
@article{cupc,
author = {Behrooz Zarebavani and Foad Jafarinejad and Matin Hashemi and Saber Salehkaleybar},
title = {{cuPC}: CUDA-based Parallel PC Algorithm for Causal Structure Learning on GPU},
journal = {IEEE Transactions on Parallel and Distributed Systems (TPDS)},
year = {2020},
volume = {31},
number = {3},
pages = {530 - 542}
} 
```
