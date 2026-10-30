package main

import "testing"

func TestSolveTSP(t *testing.T) {
	costs := [][]int64{
		{0, 10, 15, 20},
		{10, 0, 35, 25},
		{15, 35, 0, 30},
		{20, 25, 30, 0},
	}
	result, err := solveTSP(costs, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.BestCost != 80 {
		t.Errorf("custo mínimo = %d; esperado 80", result.BestCost)
	}
	if result.RoutesEvaluated != 6 {
		t.Errorf("rotas avaliadas = %d; esperado 6", result.RoutesEvaluated)
	}
	if result.EdgesEvaluated != 24 {
		t.Errorf("arestas avaliadas = %d; esperado 24", result.EdgesEvaluated)
	}
	if len(result.BestRoute) != 5 || result.BestRoute[0] != 0 || result.BestRoute[4] != 0 {
		t.Errorf("rota não inicia e termina na cidade 1: %v", result.BestRoute)
	}
}

func TestSolveTSPRejectsNonSquareMatrix(t *testing.T) {
	if _, err := solveTSP([][]int64{{0, 1}, {1}}, nil); err == nil {
		t.Fatal("esperava erro para matriz não quadrada")
	}
}
