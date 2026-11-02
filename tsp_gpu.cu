// GPU brute-force solver for the Traveling Salesman Problem.
//
// Each thread decodes a flat index t in [0, (n-1)!) into a permutation of
// cities 1..n-1 (factorial / Lehmer number system), keeps city 0 fixed as the
// origin, evaluates the Hamiltonian cycle cost (n edges) and tracks the
// minimum. A grid-stride loop lets a fixed-size grid cover the whole search
// space, so any (n-1)! fits regardless of grid dimensions.
//
// Global best cost is reduced with a single atomicMin per thread; the thread
// that first lowers the global best records its cycle so we recover the route
// without materialising every path.

#include "tsp_gpu.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stdint.h>

// Pass 1: reduce every cycle cost to the global minimum via atomicMin.
__global__ void tsp_pass1(int n, const int32_t *__restrict__ d_costs,
                          uint64_t total, int32_t *best_cost) {
    extern __shared__ int32_t s_cost[];  // n*n matrix, cached in shared memory
    const int nn = n * n;
    for (int i = threadIdx.x; i < nn; i += blockDim.x) {
        s_cost[i] = d_costs[i];
    }
    __syncthreads();
    const int32_t *C = s_cost;

    uint64_t fact[20];
    fact[0] = 1;
    for (int k = 1; k < n; k++) {
        fact[k] = fact[k - 1] * (uint64_t)k;
    }

    int32_t local_best = INT32_MAX;
    const uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
    for (uint64_t t = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
         t < total; t += stride) {
        int32_t perm[20];
        int used[20] = {0};
        uint64_t tt = t;
        for (int p = 0; p < n - 1; p++) {
            uint64_t f = fact[n - 2 - p];
            uint64_t d = (tt / f) % (uint64_t)(n - 1 - p);
            int chosen = -1;
            int count = 0;
            for (int c = 1; c <= n - 1; c++) {
                if (used[c]) continue;
                if ((uint64_t)count == d) {
                    chosen = c;
                    used[c] = 1;
                    break;
                }
                count++;
            }
            perm[p] = chosen;
        }
        int32_t cost = 0;
        int32_t prev = 0;
        for (int i = 0; i < n - 1; i++) {
            int32_t cur = perm[i];
            cost += C[prev * n + cur];
            prev = cur;
        }
        cost += C[prev * n + 0];
        if (cost < local_best) local_best = cost;
    }
    if (local_best < INT32_MAX) atomicMin(best_cost, local_best);
}

// Pass 2: re-enumerate and store the cycle of the first thread that hits the
// global minimum. atomicCAS on a one-shot flag makes the capture race-free, so
// the stored route always corresponds to best_cost and is a valid permutation.
__global__ void tsp_pass2(int n, const int32_t *__restrict__ d_costs,
                          uint64_t total, int32_t best_cost, int32_t *best_route,
                          uint64_t *best_t) {
    extern __shared__ int32_t s_cost[];
    const int nn = n * n;
    for (int i = threadIdx.x; i < nn; i += blockDim.x) {
        s_cost[i] = d_costs[i];
    }
    __syncthreads();
    const int32_t *C = s_cost;

    uint64_t fact[20];
    fact[0] = 1;
    for (int k = 1; k < n; k++) {
        fact[k] = fact[k - 1] * (uint64_t)k;
    }

    const uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
    for (uint64_t t = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
         t < total; t += stride) {
        int32_t perm[20];
        int used[20] = {0};
        uint64_t tt = t;
        for (int p = 0; p < n - 1; p++) {
            uint64_t f = fact[n - 2 - p];
            uint64_t d = (tt / f) % (uint64_t)(n - 1 - p);
            int chosen = -1;
            int count = 0;
            for (int c = 1; c <= n - 1; c++) {
                if (used[c]) continue;
                if ((uint64_t)count == d) {
                    chosen = c;
                    used[c] = 1;
                    break;
                }
                count++;
            }
            perm[p] = chosen;
        }
        int32_t cost = 0;
        int32_t prev = 0;
        for (int i = 0; i < n - 1; i++) {
            int32_t cur = perm[i];
            cost += C[prev * n + cur];
            prev = cur;
        }
        cost += C[prev * n + 0];
        // Tie-break on the smallest enumeration index so the captured cycle is
        // deterministic across runs even when several cycles share the minimum.
        uint64_t prev_t = atomicMin((unsigned long long *)best_t, t);
        if (cost <= best_cost && prev_t > t) {
            best_route[0] = 0;
            for (int i = 0; i < n - 1; i++) best_route[i + 1] = perm[i];
            best_route[n] = 0;
        }
    }
}

extern "C" int tsp_gpu_run(int n, const int32_t *costs_host,
                           int32_t *best_cost_out, int32_t *best_route_out) {
    uint64_t total = 1;
    for (int i = 1; i <= n - 1; i++) {
        total *= (uint64_t)i;  // (n-1)!
    }

    int32_t *d_costs = nullptr;
    int32_t *d_best = nullptr;
    int32_t *d_route = nullptr;
    uint64_t *d_best_t = nullptr;

    if (cudaMalloc(&d_costs, (size_t)n * n * sizeof(int32_t)) != cudaSuccess) {
        return 1;
    }
    if (cudaMalloc(&d_best, sizeof(int32_t)) != cudaSuccess) {
        cudaFree(d_costs);
        return 1;
    }
    if (cudaMalloc(&d_route, ((size_t)n + 1) * sizeof(int32_t)) != cudaSuccess) {
        cudaFree(d_costs);
        cudaFree(d_best);
        return 1;
    }
    if (cudaMalloc(&d_best_t, sizeof(uint64_t)) != cudaSuccess) {
        cudaFree(d_costs);
        cudaFree(d_best);
        cudaFree(d_route);
        return 1;
    }

    cudaMemcpy(d_costs, costs_host, (size_t)n * n * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    int32_t init = INT32_MAX;
    cudaMemcpy(d_best, &init, sizeof(int32_t), cudaMemcpyHostToDevice);
    uint64_t best_t = UINT64_MAX;
    cudaMemcpy(d_best_t, &best_t, sizeof(uint64_t), cudaMemcpyHostToDevice);

    const int blockSize = 256;
    uint64_t needed = (total + blockSize - 1) / blockSize;
    int grid = (int)(needed < 2048 ? needed : 2048);  // saturate, don't explode
    if (grid < 1) grid = 1;
    size_t shmem = (size_t)n * n * sizeof(int32_t);

    // Pass 1: global minimum cost.
    tsp_pass1<<<grid, blockSize, shmem>>>(n, d_costs, total, d_best);
    // Pass 2: capture the lowest-index optimal cycle (deterministic).
    tsp_pass2<<<grid, blockSize, shmem>>>(n, d_costs, total, init, d_route, d_best_t);

    cudaMemcpy(best_cost_out, d_best, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_route_out, d_route, ((size_t)n + 1) * sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(d_costs);
    cudaFree(d_best);
    cudaFree(d_route);
    return 0;
}
