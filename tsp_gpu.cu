// Busca exaustiva em duas passagens: mínimo e índice da primeira rota ótima.
#include "tsp_gpu.h"
#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>

__device__ int32_t route_cost(int n, int start, const int32_t *costs, uint64_t t,
                             unsigned long long *edges) {
    uint64_t fact[20];
    fact[0] = 1;
    for (int i = 1; i < n; i++) fact[i] = fact[i - 1] * (uint64_t)i;
    bool used[20] = {false};
    int prev = start;
    int32_t cost = 0;
    for (int p = 0; p < n - 1; p++) {
        uint64_t f = fact[n - 2 - p];
        uint64_t digit = (t / f) % (uint64_t)(n - 1 - p);
        int chosen = -1;
        for (int city = 0; city < n; city++) {
            if (city != start && !used[city]) {
                if (digit == 0) { chosen = city; break; }
                digit--;
            }
        }
        used[chosen] = true;
        cost += costs[prev * n + chosen];
        if (edges) ++*edges;
        prev = chosen;
    }
    if (edges) ++*edges;
    return cost + costs[prev * n + start];
}

__global__ void pass_min(int n, int start, const int32_t *costs, uint64_t total,
                         int32_t *best, unsigned long long *routes,
                         unsigned long long *edges) {
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    unsigned long long count = 0;
    unsigned long long edge_count = 0;
    int32_t local_best = INT32_MAX;
    for (uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         t < total; t += stride) {
        int32_t cost = route_cost(n, start, costs, t, &edge_count);
        if (cost < local_best) local_best = cost;
        count++;
    }
    if (count) {
        atomicMin(best, local_best);
        atomicAdd(routes, count);
        atomicAdd(edges, edge_count);
    }
}

__global__ void pass_index(int n, int start, const int32_t *costs, uint64_t total,
                           const int32_t *best, unsigned long long *best_index,
                           unsigned long long *routes, unsigned long long *edges) {
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    unsigned long long count = 0, edge_count = 0;
    for (uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         t < total; t += stride) {
        if (route_cost(n, start, costs, t, &edge_count) == *best)
            atomicMin(best_index, (unsigned long long)t);
        count++;
    }
    if (count) {
        atomicAdd(routes, count);
        atomicAdd(edges, edge_count);
    }
}

extern "C" int tsp_gpu_run(int n, int start, const int32_t *host, int32_t *best_out,
                           int32_t *route, uint64_t *routes_out, uint64_t *edges_out) {
    if (n < 1 || n > 18 || start < 0 || start >= n) return 1;
    uint64_t total = 1;
    for (int i = 1; i < n; i++) total *= (uint64_t)i;
    int32_t *costs = nullptr, *best = nullptr;
    unsigned long long *index = nullptr, *routes = nullptr, *edges = nullptr;
    auto cleanup = [&]() { cudaFree(costs); cudaFree(best); cudaFree(index); cudaFree(routes); cudaFree(edges); };
    auto check = [&](cudaError_t error) { if (error != cudaSuccess) { cleanup(); return false; } return true; };
    if (!check(cudaMalloc(&costs, (size_t)n * n * sizeof(int32_t)))) return 1;
    if (!check(cudaMalloc(&best, sizeof(*best)))) return 1;
    if (!check(cudaMalloc(&index, sizeof(*index)))) return 1;
    if (!check(cudaMalloc(&routes, sizeof(*routes)))) return 1;
    if (!check(cudaMalloc(&edges, sizeof(*edges)))) return 1;
    int32_t initial = INT32_MAX;
    unsigned long long initial_index = ULLONG_MAX, zero = 0, host_index, host_routes, host_edges;
    if (!check(cudaMemcpy(costs, host, (size_t)n * n * sizeof(int32_t), cudaMemcpyHostToDevice))) return 1;
    if (!check(cudaMemcpy(best, &initial, sizeof(initial), cudaMemcpyHostToDevice))) return 1;
    if (!check(cudaMemcpy(index, &initial_index, sizeof(initial_index), cudaMemcpyHostToDevice))) return 1;
    if (!check(cudaMemcpy(routes, &zero, sizeof(zero), cudaMemcpyHostToDevice))) return 1;
    if (!check(cudaMemcpy(edges, &zero, sizeof(zero), cudaMemcpyHostToDevice))) return 1;
    int grid = (int)((total + 255) / 256 < 2048 ? (total + 255) / 256 : 2048);
    pass_min<<<grid, 256>>>(n, start, costs, total, best, routes, edges);
    if (!check(cudaGetLastError())) return 1;
    pass_index<<<grid, 256>>>(n, start, costs, total, best, index, routes, edges);
    if (!check(cudaGetLastError())) return 1;
    if (!check(cudaMemcpy(best_out, best, sizeof(*best_out), cudaMemcpyDeviceToHost))) return 1;
    if (!check(cudaMemcpy(&host_index, index, sizeof(host_index), cudaMemcpyDeviceToHost))) return 1;
    if (!check(cudaMemcpy(&host_routes, routes, sizeof(host_routes), cudaMemcpyDeviceToHost))) return 1;
    if (!check(cudaMemcpy(&host_edges, edges, sizeof(host_edges), cudaMemcpyDeviceToHost))) return 1;
    cleanup();
    if (host_index >= total) return 1;
    *routes_out = host_routes;
    *edges_out = host_edges;
    uint64_t fact[20]; fact[0] = 1;
    for (int i = 1; i < n; i++) fact[i] = fact[i - 1] * (uint64_t)i;
    bool used[20] = {false};
    route[0] = route[n] = start;
    for (int p = 0; p < n - 1; p++) {
        uint64_t digit = (host_index / fact[n - 2 - p]) % (uint64_t)(n - 1 - p);
        for (int city = 0; city < n; city++) {
            if (city != start && !used[city]) {
                if (digit == 0) { route[p + 1] = city; used[city] = true; break; }
                digit--;
            }
        }
    }
    return 0;
}
