// Copyright (C) 2026 Rahul Biswas. New file in the CITS fork of cuPC (not in the original cuPC).
// Licensed under the GNU GPL version 3, the license of cuPC (see LICENSE).
//
// ext_levels.cuh  (2026-10-19)
//
// Extended conditioning levels for cuPC (Zarebavani et al.), appended to a verbatim copy of the
// patched 64-bit source (cuPC-S64.cu -> cuPC-S-ext.cu). Nothing above this include is changed
// except the header name.
//
// 1. A stateful per-level host API (ext_*), so that a Python driver runs one level per call:
//    level 0 and levels 1-14 with the ORIGINAL kernels (same launch configuration and order as
//    Skeleton()), and any level l >= 1 with the new generic kernel cal_Indep_ext.
// 2. cal_Indep_ext: PC-stable Fisher-z tests at an arbitrary level l (no compile-time cap).
//    - Work unit = task (row X, contiguous range of lexicographic ranks of size-l subsets of the
//      start-of-level adjacency adj(X)). One warp per task; tasks are pulled from an atomic counter.
//    - Statistic: incremental Cholesky. For the current subset S = (s_0 < ... < s_{l-1}) and every
//      slot a in adj(X) plus X itself, b_a = L^{-1} C[S, a] (L L^T = C[S, S]) together with the
//      running sums ||b_a||^2 and b_X . b_a at every depth. Then for Y in adj(X) \ S:
//        H00 = C_XX - ||b_X||^2,  H11 = C_YY - ||b_Y||^2,  H01 = C_XY - b_X . b_Y,
//        rho = H01 / sqrt(H00 H11)   (the partial correlation of X and Y given S).
//    - Decision: rho^2 against tanh(th)^2 with a relative safety band of 1e-9; inside the band the
//      cuPC formula Z = |0.5 log|(1+rho)/(1-rho)|| < th decides. Degenerate H (non-positive or
//      non-finite diagonal) keeps the edge (as the CPU continuation does) and is counted.
//    - Removal: atomicCAS on the canonical upper-triangle entry G[min][max] (exactly one remover),
//      then the mirror entry, the per-row present-edge counters, and one log record
//      (X, Y, S as node ids, rho).
//    - A row is abandoned only when its present-edge counter reaches 0.

#include <stdint.h>
#include <stdio.h>
#include <math.h>

#define EXT_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) {                    \
    fprintf(stderr, "[cupc_ext] CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__,   \
            __LINE__); return -1; } } while (0)

struct ExtHandle {
    int n;
    int nprime;      // max row size from the last ext_scan
    int has_old;     // buffers for the original kernels (mutex, SepSet) allocated
    double *C;
    int *G;
    int *GPrime;
    int *mutex;
    int *nprime_d;
    double *pMax;
    int *SepSet;
};

enum { EXT_ST_LEAVES = 0, EXT_ST_TESTS, EXT_ST_BAND, EXT_ST_NONPOS, EXT_ST_DEGEN, EXT_ST_SKIPPED,
       EXT_ST_REMOVED, EXT_ST_TASKS, EXT_NSTATS };

struct ExtWarpWS {
    double *B;            // [l][MP]     b_a[k]
    double *NRM;          // [l+1][MP]   sum_{j<k} b_a[j]^2
    double *DOT;          // [l+1][MP]   sum_{j<k} b_X[j] b_a[j]
    double *CX;           // [MP]        C[X, node[a]]
    double *CD;           // [MP]        C[node[a], node[a]] or 1.0
    int *node;            // [MP]        node id of slot a (slot m = X, -1 beyond)
    int *Spos;            // [l+1]       positions (slots) of the current subset
    unsigned char *inS;   // [MP]        1 if slot a is in the current subset
};

__host__ __device__ inline size_t ext_ws_bytes(int l, int MP)
{
    size_t d = (size_t)l * MP + 2 * (size_t)(l + 1) * MP + 2 * (size_t)MP;
    size_t bytes = d * sizeof(double);
    bytes += (size_t)MP * sizeof(int);
    bytes += (size_t)(l + 1) * sizeof(int);
    bytes += (size_t)MP;
    bytes = (bytes + 15) & ~(size_t)15;
    return bytes;
}

__device__ __forceinline__ void ext_carve(char *base, int l, int MP, ExtWarpWS &w)
{
    w.B = (double *)base;
    w.NRM = w.B + (size_t)l * MP;
    w.DOT = w.NRM + (size_t)(l + 1) * MP;
    w.CX = w.DOT + (size_t)(l + 1) * MP;
    w.CD = w.CX + MP;
    w.node = (int *)(w.CD + MP);
    w.Spos = w.node + MP;
    w.inS = (unsigned char *)(w.Spos + (l + 1));
}

// Append slot zs at depth k (the prefix S[0..k-1] is already factorised).
__device__ __forceinline__ void ext_append(const int k, const int zs, const int lane, const int MP,
                                           const int m, const int n, const double *__restrict__ C,
                                           ExtWarpWS &w, unsigned long long &n_nonpos)
{
    const int zn = w.node[zs];
    const double piv = w.CD[zs] - w.NRM[(size_t)k * MP + zs];
    if (lane == 0 && !(piv > 0.0)) n_nonpos++;
    const double invd = 1.0 / sqrt(piv);
    const double *__restrict__ Crow = C + (size_t)zn * n;
    const double *Bz = w.B + zs;
    for (int a = lane; a <= m; a += 32) {
        double t = 0.0;
        for (int j = 0; j < k; j++) t = fma(Bz[(size_t)j * MP], w.B[(size_t)j * MP + a], t);
        const double b = (__ldg(Crow + w.node[a]) - t) * invd;
        w.B[(size_t)k * MP + a] = b;
        w.NRM[(size_t)(k + 1) * MP + a] = fma(b, b, w.NRM[(size_t)k * MP + a]);
    }
    __syncwarp();
    const double bXk = w.B[(size_t)k * MP + m];
    for (int a = lane; a <= m; a += 32)
        w.DOT[(size_t)(k + 1) * MP + a] = fma(bXk, w.B[(size_t)k * MP + a], w.DOT[(size_t)k * MP + a]);
    __syncwarp();
}

// Tests at a leaf (full subset of size l).
__device__ __forceinline__ void ext_leaf(const int X, const int l, const int lane, const int MP,
                                         const int m, const int n, int *G, ExtWarpWS &w,
                                         const double th, const double t2lo, const double t2hi,
                                         int *row_left, int *log_xy, int *log_S, double *log_rho,
                                         const long long cap, unsigned long long *log_count,
                                         unsigned long long &n_test, unsigned long long &n_band,
                                         unsigned long long &n_degen, unsigned long long &n_rm)
{
    const double h00 = w.CD[m] - w.NRM[(size_t)l * MP + m];
    for (int a = lane; a < m; a += 32) {
        if (w.inS[a]) continue;
        const int Y = w.node[a];
        if (__ldcg(G + (size_t)X * n + Y) == 0) continue;
        n_test++;
        const double h11 = w.CD[a] - w.NRM[(size_t)l * MP + a];
        const double h01 = w.CX[a] - w.DOT[(size_t)l * MP + a];
        bool indep;
        if (!(h00 > 0.0) || !(h11 > 0.0) || !isfinite(h00) || !isfinite(h11) || !isfinite(h01)) {
            n_degen++;
            indep = false;
        } else {
            const double p = h00 * h11;
            const double q = h01 * h01;
            if (q < t2lo * p) {
                indep = true;
            } else if (q > t2hi * p) {
                indep = false;
            } else {
                n_band++;
                const double rho = h01 / sqrt(fabs(h00 * h11));
                const double Z = fabs(0.5 * log(fabs((1.0 + rho) / (1.0 - rho))));
                indep = (Z < th);
            }
        }
        if (indep) {
            const int lo = min(X, Y), hi = max(X, Y);
            if (atomicCAS(G + (size_t)lo * n + hi, 1, 0) == 1) {
                atomicExch(G + (size_t)hi * n + lo, 0);
                atomicSub(row_left + X, 1);
                atomicSub(row_left + Y, 1);
                n_rm++;
                const unsigned long long s = atomicAdd(log_count, 1ULL);
                if (s < (unsigned long long)cap) {
                    log_xy[2 * s] = X;
                    log_xy[2 * s + 1] = Y;
                    for (int q2 = 0; q2 < l; q2++) log_S[(size_t)s * l + q2] = w.node[w.Spos[q2]];
                    log_rho[s] = h01 / sqrt(h00 * h11);
                }
            }
        }
    }
    __syncwarp();
}

template <bool SHARED>
__global__ void __launch_bounds__(256)
cal_Indep_ext(const double *__restrict__ C, int *G, const int *__restrict__ GPrime, const int n,
              const int l, const double th, const double t2lo, const double t2hi, const int use_cdiag,
              const int *__restrict__ task_row, const long long *__restrict__ task_start,
              const long long *__restrict__ task_count, const long long ntasks,
              unsigned long long *task_counter, const unsigned long long *__restrict__ binom,
              const int bt_cols, const int MP, const size_t ws_bytes, char *gws, int *row_left,
              int *log_xy, int *log_S, double *log_rho, const long long cap,
              unsigned long long *log_count, unsigned long long *stats, int *err)
{
    extern __shared__ __align__(16) unsigned char ext_smem[];
    const unsigned FULL = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int wib = threadIdx.x >> 5;
    char *base = SHARED ? (char *)ext_smem + (size_t)wib * ws_bytes
                        : gws + ((size_t)blockIdx.x * (blockDim.x >> 5) + wib) * ws_bytes;
    ExtWarpWS w;
    ext_carve(base, l, MP, w);
    unsigned long long n_leaf = 0, n_test = 0, n_band = 0, n_nonpos = 0, n_degen = 0, n_skip = 0,
                       n_rm = 0, n_task = 0;

    while (true) {
        long long t = 0;
        if (lane == 0) t = (long long)atomicAdd(task_counter, 1ULL);
        t = __shfl_sync(FULL, t, 0);
        if (t >= ntasks) break;
        const int X = task_row[t];
        const long long r0 = task_start[t];
        const long long cnt = task_count[t];
        int left = 0;
        if (lane == 0) left = __ldcg(row_left + X);
        left = __shfl_sync(FULL, left, 0);
        if (left <= 0) {
            if (lane == 0) n_skip++;
            continue;
        }
        if (lane == 0) n_task++;
        const int m = GPrime[(size_t)X * n + n - 1];
        for (int a = lane; a < MP; a += 32) {
            const int nd = (a < m) ? GPrime[(size_t)X * n + a] : ((a == m) ? X : -1);
            w.node[a] = nd;
            w.inS[a] = 0;
            w.NRM[a] = 0.0;
            w.DOT[a] = 0.0;
            if (nd >= 0) {
                w.CX[a] = C[(size_t)X * n + nd];
                w.CD[a] = use_cdiag ? C[(size_t)nd * n + nd] : 1.0;
            } else {
                w.CX[a] = 0.0;
                w.CD[a] = 1.0;
            }
        }
        __syncwarp();
        if (lane == 0) {
            // unrank r0 (0-based, lexicographic over positions 0..m-1) with the exact table
            long long r = r0;
            int x = 0, bad = 0;
            for (int i = 0; i < l && !bad; i++) {
                while (true) {
                    if (x >= m) { bad = 1; break; }
                    const unsigned long long c = binom[(size_t)(m - x - 1) * bt_cols + (l - i - 1)];
                    if ((unsigned long long)r < c) break;
                    r -= (long long)c;
                    x++;
                }
                if (!bad) { w.Spos[i] = x; x++; }
            }
            if (bad) {
                atomicExch(err, 1);
                for (int i = 0; i < l; i++) w.Spos[i] = i;
            }
            for (int i = 0; i < l; i++) w.inS[w.Spos[i]] = 1;
        }
        __syncwarp();
        for (int k = 0; k < l; k++) ext_append(k, w.Spos[k], lane, MP, m, n, C, w, n_nonpos);
        for (long long i = 0;; i++) {
            ext_leaf(X, l, lane, MP, m, n, G, w, th, t2lo, t2hi, row_left, log_xy, log_S, log_rho,
                     cap, log_count, n_test, n_band, n_degen, n_rm);
            if (lane == 0) n_leaf++;
            if (i + 1 >= cnt) break;
            if ((i & 63) == 63) {
                int lf = 0;
                if (lane == 0) lf = __ldcg(row_left + X);
                lf = __shfl_sync(FULL, lf, 0);
                if (lf <= 0) break;
            }
            int j = 0;
            if (lane == 0) {
                j = l - 1;
                while (j >= 0 && w.Spos[j] == m - l + j) j--;
                if (j < 0) {
                    atomicExch(err, 2);
                } else {
                    for (int q2 = j; q2 < l; q2++) w.inS[w.Spos[q2]] = 0;
                    w.Spos[j]++;
                    for (int q2 = j + 1; q2 < l; q2++) w.Spos[q2] = w.Spos[q2 - 1] + 1;
                    for (int q2 = j; q2 < l; q2++) w.inS[w.Spos[q2]] = 1;
                }
            }
            j = __shfl_sync(FULL, j, 0);
            __syncwarp();
            if (j < 0) break;
            for (int k = j; k < l; k++) ext_append(k, w.Spos[k], lane, MP, m, n, C, w, n_nonpos);
        }
    }
    if (n_leaf) atomicAdd(stats + EXT_ST_LEAVES, n_leaf);
    if (n_test) atomicAdd(stats + EXT_ST_TESTS, n_test);
    if (n_band) atomicAdd(stats + EXT_ST_BAND, n_band);
    if (n_nonpos) atomicAdd(stats + EXT_ST_NONPOS, n_nonpos);
    if (n_degen) atomicAdd(stats + EXT_ST_DEGEN, n_degen);
    if (n_skip) atomicAdd(stats + EXT_ST_SKIPPED, n_skip);
    if (n_rm) atomicAdd(stats + EXT_ST_REMOVED, n_rm);
    if (n_task) atomicAdd(stats + EXT_ST_TASKS, n_task);
}

// ============================================================================ host API

extern "C" void *ext_create(const double *C, int n, int has_old)
{
    ExtHandle *h = (ExtHandle *)calloc(1, sizeof(ExtHandle));
    if (!h) return NULL;
    h->n = n;
    h->has_old = has_old;
    size_t nn = (size_t)n * n;
    cudaError_t e = cudaSuccess;
    if (e == cudaSuccess) e = cudaMalloc((void **)&h->C, nn * sizeof(double));
    if (e == cudaSuccess) e = cudaMalloc((void **)&h->G, nn * sizeof(int));
    if (e == cudaSuccess) e = cudaMalloc((void **)&h->GPrime, nn * sizeof(int));
    if (e == cudaSuccess) e = cudaMalloc((void **)&h->nprime_d, sizeof(int));
    if (e == cudaSuccess) e = cudaMalloc((void **)&h->pMax, nn * sizeof(double));
    if (has_old) {
        if (e == cudaSuccess) e = cudaMalloc((void **)&h->mutex, nn * sizeof(int));
        if (e == cudaSuccess) e = cudaMalloc((void **)&h->SepSet, nn * ML * sizeof(int));
        if (e == cudaSuccess) e = cudaMemset(h->mutex, 0, nn * sizeof(int));
    }
    if (e == cudaSuccess) e = cudaMemcpy(h->C, C, nn * sizeof(double), cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemset(h->G, 0, nn * sizeof(int));
    if (e == cudaSuccess) e = cudaMemset(h->GPrime, 0, nn * sizeof(int));
    if (e == cudaSuccess) e = cudaMemset(h->pMax, 0, nn * sizeof(double));
    if (e != cudaSuccess) {
        fprintf(stderr, "[cupc_ext] ext_create failed: %s\n", cudaGetErrorString(e));
        cudaFree(h->C); cudaFree(h->G); cudaFree(h->GPrime); cudaFree(h->nprime_d);
        cudaFree(h->pMax); cudaFree(h->mutex); cudaFree(h->SepSet);
        free(h);
        return NULL;
    }
    return (void *)h;
}

extern "C" void ext_destroy(void *hv)
{
    ExtHandle *h = (ExtHandle *)hv;
    if (!h) return;
    cudaFree(h->C); cudaFree(h->G); cudaFree(h->GPrime); cudaFree(h->nprime_d);
    cudaFree(h->pMax); cudaFree(h->mutex); cudaFree(h->SepSet);
    free(h);
}

// Level 0 exactly as in Skeleton().
extern "C" int ext_level0(void *hv, double th0)
{
    ExtHandle *h = (ExtHandle *)hv;
    int n = h->n;
    dim3 BLOCKS_PER_GRID, THREADS_PER_BLOCK;
    if ((n * n) < 1024) {
        BLOCKS_PER_GRID = dim3(1, 1, 1);
        THREADS_PER_BLOCK = dim3(32, 32, 1);
    } else {
        BLOCKS_PER_GRID = dim3(ceil(((double)(n)) / 32.0), ceil(((double)(n)) / 32.0), 1);
        THREADS_PER_BLOCK = dim3(32, 32, 1);
    }
    cal_Indepl0<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(h->C, h->G, th0, h->pMax, n);
    EXT_CHECK(cudaGetLastError());
    EXT_CHECK(cudaDeviceSynchronize());
    if (h->has_old) {
        BLOCKS_PER_GRID = dim3(n * n, 1, 1);
        THREADS_PER_BLOCK = dim3(ML, 1, 1);
        SepSet_initialize<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(h->SepSet, n);
        EXT_CHECK(cudaGetLastError());
        EXT_CHECK(cudaDeviceSynchronize());
    }
    return 0;
}

// Scan/compact exactly as in Skeleton(); returns the max row size, fills rowsize_out[n].
extern "C" int ext_scan(void *hv, int *rowsize_out, int *nprime_out)
{
    ExtHandle *h = (ExtHandle *)hv;
    int n = h->n;
    EXT_CHECK(cudaMemset(h->nprime_d, 0, 1 * sizeof(int)));
    dim3 BLOCKS_PER_GRID = dim3(1, n, 1);
    dim3 THREADS_PER_BLOCK = dim3(1024, 1, 1);
    scan_compact<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, n * sizeof(int)>>>(h->GPrime, h->G, n, h->nprime_d);
    EXT_CHECK(cudaGetLastError());
    EXT_CHECK(cudaDeviceSynchronize());
    int nprime = 0;
    EXT_CHECK(cudaMemcpy(&nprime, h->nprime_d, sizeof(int), cudaMemcpyDeviceToHost));
    h->nprime = nprime;
    *nprime_out = nprime;
    if (rowsize_out)
        EXT_CHECK(cudaMemcpy2D(rowsize_out, sizeof(int), h->GPrime + (n - 1), (size_t)n * sizeof(int),
                               sizeof(int), n, cudaMemcpyDeviceToHost));
    return 0;
}

// Original kernel for level l in 1..14, same launch configuration as Skeleton().
extern "C" int ext_level_old(void *hv, int l, double th)
{
    ExtHandle *h = (ExtHandle *)hv;
    int n = h->n;
    int nprime = h->nprime;
    if (!h->has_old || l < 1 || l > ML) return -2;
    dim3 B, T;
    size_t sh = nprime * sizeof(int);
    switch (l) {
    case 1: B = dim3(NumOfBlockForEachNodeL1, n, 1); T = dim3(ParGivenL1, 1, 1);
        cal_Indepl1<<<B, T, sh>>>(h->C, h->G, h->GPrime, h->mutex, h->SepSet, h->pMax, th, n); break;
    case 2: B = dim3(NumOfBlockForEachNodeL2, n, 1); T = dim3(ParGivenL2, 1, 1);
        cal_Indepl2<<<B, T, sh>>>(h->C, h->G, h->GPrime, h->mutex, h->SepSet, h->pMax, n, th); break;
    case 3: B = dim3(NumOfBlockForEachNodeL3, n, 1); T = dim3(ParGivenL3, 1, 1);
        cal_Indepl3<<<B, T, sh>>>(h->C, h->G, h->GPrime, h->mutex, h->SepSet, h->pMax, n, th); break;
    case 4: B = dim3(NumOfBlockForEachNodeL4, n, 1); T = dim3(ParGivenL4, 1, 1);
        cal_Indepl4<<<B, T, sh>>>(h->C, h->G, h->GPrime, h->mutex, h->SepSet, h->pMax, n, th); break;
    case 5: B = dim3(NumOfBlockForEachNodeL5, n, 1); T = dim3(ParGivenL5, 1, 1);
        cal_Indepl5<<<B, T, sh>>>(h->C, h->G, h->GPrime, h->mutex, h->SepSet, h->pMax, n, th); break;
    case 6: B = dim3(NumOfBlockForEachNodeL6, n, 1); T = dim3(ParGivenL6, 1, 1);
        cal_Indepl6<<<B, T, sh>>>(h->C, h->G, h->GPrime, h->mutex, h->SepSet, h->pMax, n, th); break;
    case 7: B = dim3(NumOfBlockForEachNodeL7, n, 1); T = dim3(ParGivenL7, 1, 1);
        cal_Indepl7<<<B, T, sh>>>(h->C, h->G, h->GPrime, h->mutex, h->SepSet, h->pMax, n, th); break;
    case 8: B = dim3(NumOfBlockForEachNodeL8, n, 1); T = dim3(ParGivenL8, 1, 1);
        cal_Indepl8<<<B, T, sh>>>(h->C, h->G, h->GPrime, h->mutex, h->SepSet, h->pMax, n, th); break;
    case 9: B = dim3(NumOfBlockForEachNodeL9, n, 1); T = dim3(ParGivenL9, 1, 1);
        cal_Indepl9<<<B, T, sh>>>(h->C, h->G, h->GPrime, h->mutex, h->SepSet, h->pMax, n, th); break;
    case 10: B = dim3(NumOfBlockForEachNodeL10, n, 1); T = dim3(ParGivenL10, 1, 1);
        cal_Indepl10<<<B, T, sh>>>(h->C, h->G, h->GPrime, h->mutex, h->SepSet, h->pMax, n, th); break;
    case 11: B = dim3(NumOfBlockForEachNodeL11, n, 1); T = dim3(ParGivenL11, 1, 1);
        cal_Indepl11<<<B, T, sh>>>(h->C, h->G, h->GPrime, h->mutex, h->SepSet, h->pMax, n, th); break;
    case 12: B = dim3(NumOfBlockForEachNodeL12, n, 1); T = dim3(ParGivenL12, 1, 1);
        cal_Indepl12<<<B, T, sh>>>(h->C, h->G, h->GPrime, h->mutex, h->SepSet, h->pMax, n, th); break;
    case 13: B = dim3(NumOfBlockForEachNodeL13, n, 1); T = dim3(ParGivenL13, 1, 1);
        cal_Indepl13<<<B, T, sh>>>(h->C, h->G, h->GPrime, h->mutex, h->SepSet, h->pMax, n, th); break;
    case 14: B = dim3(NumOfBlockForEachNodeL14, n, 1); T = dim3(ParGivenL14, 1, 1);
        cal_Indepl14<<<B, T, sh>>>(h->C, h->G, h->GPrime, h->mutex, h->SepSet, h->pMax, n, th); break;
    }
    EXT_CHECK(cudaGetLastError());
    EXT_CHECK(cudaDeviceSynchronize());
    return 0;
}

extern "C" int ext_get_G(void *hv, int *G_out)
{
    ExtHandle *h = (ExtHandle *)hv;
    size_t nn = (size_t)h->n * h->n;
    EXT_CHECK(cudaMemcpy(G_out, h->G, nn * sizeof(int), cudaMemcpyDeviceToHost));
    return 0;
}

// Replace the device graph (used for per-level A/B comparisons). Also clears the original
// kernels' removal mutex, which only gates removals of edges that are present.
extern "C" int ext_set_G(void *hv, const int *G_in)
{
    ExtHandle *h = (ExtHandle *)hv;
    size_t nn = (size_t)h->n * h->n;
    EXT_CHECK(cudaMemcpy(h->G, G_in, nn * sizeof(int), cudaMemcpyHostToDevice));
    if (h->has_old) EXT_CHECK(cudaMemset(h->mutex, 0, nn * sizeof(int)));
    return 0;
}

// New kernel for level l. Tasks, the binomial table and the start-of-level row sizes come from
// the host. Logs are written to caller buffers with capacity cap (= edges at the level start).
// cfg_out[0..5] = warps per block, blocks, shared(1)/global(0) workspace, MP, workspace bytes, err.
extern "C" int ext_level_new(void *hv, int l, double th, double t2lo, double t2hi, int use_cdiag,
                             const int *task_row, const long long *task_start,
                             const long long *task_count, long long ntasks,
                             const unsigned long long *binom, int bt_rows, int bt_cols, int mmax,
                             const int *row_left_init, int *log_xy, int *log_S, double *log_rho,
                             long long cap, long long *nlog_out, unsigned long long *stats_out,
                             int *cfg_out, double *ms_out, int force_global, int wpb_req)
{
    ExtHandle *h = (ExtHandle *)hv;
    int n = h->n;
    if (l < 1 || ntasks <= 0 || mmax < l + 1 || bt_rows < mmax + 1 || bt_cols < l + 1) return -2;
    const int MP = 32 * ((mmax + 1 + 31) / 32);
    const size_t wsb = ext_ws_bytes(l, MP);
    int dev = 0;
    EXT_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    EXT_CHECK(cudaGetDeviceProperties(&prop, dev));
    const size_t optin = prop.sharedMemPerBlockOptin;
    int wpb = wpb_req > 0 ? wpb_req : 8;
    if (wpb > 8) wpb = 8;
    const bool use_shared = (!force_global) && (wsb <= optin);
    if (use_shared) {
        while (wpb > 1 && (size_t)wpb * wsb > optin) wpb--;
    }
    const int threads = 32 * wpb;
    const size_t smem = use_shared ? (size_t)wpb * wsb : 0;

    int *d_task_row = NULL, *d_row_left = NULL, *d_log_xy = NULL, *d_log_S = NULL, *d_err = NULL;
    long long *d_task_start = NULL, *d_task_count = NULL;
    unsigned long long *d_binom = NULL, *d_cnt = NULL;
    double *d_log_rho = NULL;
    char *d_gws = NULL;
    int blocks = 0, err_h = 0;
    float ms = 0.f;
    unsigned long long cnt_h[2 + EXT_NSTATS];
    cudaEvent_t e0 = NULL, e1 = NULL;
    int rc = 0;

#define EXT_TRY(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) {                      \
    fprintf(stderr, "[cupc_ext] CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__,   \
            __LINE__); rc = -1; goto cleanup; } } while (0)

    EXT_TRY(cudaEventCreate(&e0));
    EXT_TRY(cudaEventCreate(&e1));
    EXT_TRY(cudaMalloc((void **)&d_task_row, (size_t)ntasks * sizeof(int)));
    EXT_TRY(cudaMalloc((void **)&d_task_start, (size_t)ntasks * sizeof(long long)));
    EXT_TRY(cudaMalloc((void **)&d_task_count, (size_t)ntasks * sizeof(long long)));
    EXT_TRY(cudaMalloc((void **)&d_binom, (size_t)bt_rows * bt_cols * sizeof(unsigned long long)));
    EXT_TRY(cudaMalloc((void **)&d_row_left, (size_t)n * sizeof(int)));
    EXT_TRY(cudaMalloc((void **)&d_log_xy, (size_t)cap * 2 * sizeof(int)));
    EXT_TRY(cudaMalloc((void **)&d_log_S, (size_t)cap * l * sizeof(int)));
    EXT_TRY(cudaMalloc((void **)&d_log_rho, (size_t)cap * sizeof(double)));
    EXT_TRY(cudaMalloc((void **)&d_cnt, (2 + EXT_NSTATS) * sizeof(unsigned long long)));
    EXT_TRY(cudaMalloc((void **)&d_err, sizeof(int)));
    EXT_TRY(cudaMemcpy(d_task_row, task_row, (size_t)ntasks * sizeof(int), cudaMemcpyHostToDevice));
    EXT_TRY(cudaMemcpy(d_task_start, task_start, (size_t)ntasks * sizeof(long long), cudaMemcpyHostToDevice));
    EXT_TRY(cudaMemcpy(d_task_count, task_count, (size_t)ntasks * sizeof(long long), cudaMemcpyHostToDevice));
    EXT_TRY(cudaMemcpy(d_binom, binom, (size_t)bt_rows * bt_cols * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    EXT_TRY(cudaMemcpy(d_row_left, row_left_init, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    EXT_TRY(cudaMemset(d_cnt, 0, (2 + EXT_NSTATS) * sizeof(unsigned long long)));
    EXT_TRY(cudaMemset(d_err, 0, sizeof(int)));

    if (use_shared) {
        EXT_TRY(cudaFuncSetAttribute(cal_Indep_ext<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        int nb = 0;
        EXT_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, cal_Indep_ext<true>, threads, smem));
        if (nb < 1) nb = 1;
        blocks = nb * prop.multiProcessorCount;
    } else {
        int nb = 0;
        EXT_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, cal_Indep_ext<false>, threads, 0));
        if (nb < 1) nb = 1;
        blocks = nb * prop.multiProcessorCount;
        EXT_TRY(cudaMalloc((void **)&d_gws, (size_t)blocks * wpb * wsb));
    }

    EXT_TRY(cudaEventRecord(e0));
    if (use_shared) {
        cal_Indep_ext<true><<<blocks, threads, smem>>>(
            h->C, h->G, h->GPrime, n, l, th, t2lo, t2hi, use_cdiag, d_task_row, d_task_start,
            d_task_count, ntasks, d_cnt, d_binom, bt_cols, MP, wsb, NULL, d_row_left, d_log_xy,
            d_log_S, d_log_rho, cap, d_cnt + 1, d_cnt + 2, d_err);
    } else {
        cal_Indep_ext<false><<<blocks, threads, 0>>>(
            h->C, h->G, h->GPrime, n, l, th, t2lo, t2hi, use_cdiag, d_task_row, d_task_start,
            d_task_count, ntasks, d_cnt, d_binom, bt_cols, MP, wsb, d_gws, d_row_left, d_log_xy,
            d_log_S, d_log_rho, cap, d_cnt + 1, d_cnt + 2, d_err);
    }
    EXT_TRY(cudaGetLastError());
    EXT_TRY(cudaEventRecord(e1));
    EXT_TRY(cudaEventSynchronize(e1));
    EXT_TRY(cudaDeviceSynchronize());
    EXT_TRY(cudaEventElapsedTime(&ms, e0, e1));

    EXT_TRY(cudaMemcpy(cnt_h, d_cnt, (2 + EXT_NSTATS) * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    EXT_TRY(cudaMemcpy(&err_h, d_err, sizeof(int), cudaMemcpyDeviceToHost));
    {
        long long nlog = (long long)cnt_h[1];
        long long ncopy = nlog < cap ? nlog : cap;
        if (ncopy > 0) {
            EXT_TRY(cudaMemcpy(log_xy, d_log_xy, (size_t)ncopy * 2 * sizeof(int), cudaMemcpyDeviceToHost));
            EXT_TRY(cudaMemcpy(log_S, d_log_S, (size_t)ncopy * l * sizeof(int), cudaMemcpyDeviceToHost));
            EXT_TRY(cudaMemcpy(log_rho, d_log_rho, (size_t)ncopy * sizeof(double), cudaMemcpyDeviceToHost));
        }
        *nlog_out = nlog;
        for (int i = 0; i < EXT_NSTATS; i++) stats_out[i] = cnt_h[2 + i];
    }
    *ms_out = (double)ms;
    cfg_out[0] = wpb;
    cfg_out[1] = blocks;
    cfg_out[2] = use_shared ? 1 : 0;
    cfg_out[3] = MP;
    cfg_out[4] = (int)wsb;
    cfg_out[5] = err_h;
    if (err_h) rc = -3;

cleanup:
    cudaFree(d_task_row); cudaFree(d_task_start); cudaFree(d_task_count); cudaFree(d_binom);
    cudaFree(d_row_left); cudaFree(d_log_xy); cudaFree(d_log_S); cudaFree(d_log_rho);
    cudaFree(d_cnt); cudaFree(d_err); cudaFree(d_gws);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
#undef EXT_TRY
    return rc;
}
