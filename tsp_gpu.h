#ifndef TSP_GPU_H
#define TSP_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Brute-force TSP solver running on the GPU.
 *
 * Costs are a symmetric n x n int32 matrix stored row-major on the host.
 * City 0 is fixed as the origin; the kernel enumerates every (n-1)!
 * permutation of cities 1..n-1, evaluates each Hamiltonian cycle (n edges)
 * and keeps the minimum-cost cycle.
 *
 * On return:
 *   *best_cost_out  - minimum total cost found (written by the kernel)
 *   *best_route_out - the optimal cycle as n+1 zero-indexed city ids,
 *                     reconstructed on the host from the winning index
 *   *routes_out, *edges_out - observed counts during the first pass
 *
 * Returns 0 on success, non-zero on a CUDA error.
 */
int tsp_gpu_run(int n, const int32_t *costs_host,
                int32_t *best_cost_out, int32_t *best_route_out,
                uint64_t *routes_out, uint64_t *edges_out);

#ifdef __cplusplus
}
#endif

#endif /* TSP_GPU_H */
