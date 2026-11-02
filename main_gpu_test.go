//go:build gpu

package main

import "testing"

func TestSolveTSPGPU(t *testing.T) {
	costs := [][]int64{{0, 10, 15, 20}, {10, 0, 35, 25}, {15, 35, 0, 30}, {20, 25, 30, 0}}
	for run := 0; run < 10; run++ {
		r := solveTSPGPU(len(costs), costs)
		if r.BestCost != 80 || r.RoutesEvaluated != 6 || r.EdgesEvaluated != 24 {
			t.Fatalf("resultado incorreto: %+v", r)
		}
		if len(r.BestRoute) != 5 || r.BestRoute[0] != 0 || r.BestRoute[4] != 0 {
			t.Fatalf("rota inválida: %v", r.BestRoute)
		}
		seen := make(map[int]bool)
		var cost int64
		for i := 0; i < 4; i++ {
			city, next := r.BestRoute[i], r.BestRoute[i+1]
			if city < 0 || city >= 4 || next < 0 || next >= 4 || seen[city] {
				t.Fatalf("rota inválida: %v", r.BestRoute)
			}
			seen[city] = true
			cost += costs[city][next]
		}
		if cost != r.BestCost {
			t.Fatalf("custo da rota %d difere do mínimo %d", cost, r.BestCost)
		}
	}
}
