// Programa de força bruta para o Problema do Caixeiro Viajante (TSP).
// A cidade 0 é fixada como origem e as demais cidades são permutadas.
// São examinadas (n-1)! rotas; cada rota avalia n arestas: Θ(n!).
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type Result struct {
	N               int
	BestRoute       []int
	BestCost        int64
	RoutesEvaluated uint64
	EdgesEvaluated  uint64
	Duration        time.Duration
}

func main() {
	minN := flag.Int("min-n", 3, "menor quantidade de cidades do benchmark")
	maxN := flag.Int("max-n", 10, "maior quantidade de cidades do benchmark (máximo suportado: 18)")
	seed := flag.Int64("seed", 42, "semente para gerar custos reproduzíveis")
	outDir := flag.String("out", "resultados", "diretório dos arquivos de saída")
	verbose := flag.Bool("verbose", false, "registra o custo de cada rota no arquivo de log")
	flag.Parse()

	if *minN < 1 || *maxN < *minN || *maxN > 18 {
		log.Fatalf("intervalo inválido: use 1 <= min-n <= max-n <= 18")
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatalf("não foi possível criar %q: %v", *outDir, err)
	}

	logFile, err := os.Create(filepath.Join(*outDir, "execucao.log"))
	if err != nil {
		log.Fatalf("não foi possível criar o log: %v", err)
	}
	defer logFile.Close()
	logger := log.New(io.MultiWriter(os.Stdout, logFile), "", log.LstdFlags)

	csvFile, err := os.Create(filepath.Join(*outDir, "resultados.csv"))
	if err != nil {
		log.Fatalf("não foi possível criar o CSV: %v", err)
	}
	defer csvFile.Close()
	csvWriter := csv.NewWriter(csvFile)
	if err := csvWriter.Write([]string{"cidades", "rotas_teoricas", "rotas_avaliadas", "arestas_avaliadas", "custo_minimo", "rota_otima", "tempo_ns", "tempo_ms"}); err != nil {
		log.Fatalf("erro ao escrever cabeçalho CSV: %v", err)
	}

	logger.Printf("TSP por força bruta | origem fixa: cidade 1 | semente: %d", *seed)
	logger.Printf("Resultados serão gravados em %s", *outDir)
	weights := randomMatrix(*maxN, *seed)
	for n := *minN; n <= *maxN; n++ {
		matrix := make([][]int64, n)
		for i := range matrix {
			matrix[i] = weights[i][:n]
		}
		var routeLogger func([]int, int64)
		if *verbose {
			routeLogger = func(route []int, cost int64) {
				logger.Printf("rota %s | custo=%d", formatRoute(route), cost)
			}
		}

		result, err := solveTSP(matrix, routeLogger)
		if err != nil {
			log.Fatalf("erro ao resolver para n=%d: %v", n, err)
		}
		expected := factorial(uint64(n - 1))
		logger.Printf("n=%d | rotas=%d (teórico=%d) | arestas=%d | custo mínimo=%d | rota=%s | tempo=%s",
			result.N, result.RoutesEvaluated, expected, result.EdgesEvaluated,
			result.BestCost, formatRoute(result.BestRoute), result.Duration)
		if result.RoutesEvaluated != expected {
			logger.Fatalf("contagem incorreta de rotas para n=%d: esperado %d, obtido %d", n, expected, result.RoutesEvaluated)
		}
		row := []string{
			strconv.Itoa(n), strconv.FormatUint(expected, 10),
			strconv.FormatUint(result.RoutesEvaluated, 10), strconv.FormatUint(result.EdgesEvaluated, 10),
			strconv.FormatInt(result.BestCost, 10), formatRoute(result.BestRoute),
			strconv.FormatInt(result.Duration.Nanoseconds(), 10),
			strconv.FormatFloat(float64(result.Duration)/float64(time.Millisecond), 'f', 6, 64),
		}
		if err := csvWriter.Write(row); err != nil {
			log.Fatalf("erro ao escrever resultado CSV: %v", err)
		}
		csvWriter.Flush()
		if err := csvWriter.Error(); err != nil {
			log.Fatalf("erro ao salvar resultado CSV: %v", err)
		}
	}
	logger.Printf("Benchmark concluído. Arquivos: %s e %s", filepath.Join(*outDir, "execucao.log"), filepath.Join(*outDir, "resultados.csv"))
}

// solveTSP enumera todas as permutações das cidades 1..n-1, mantendo a cidade 0 fixa.
func solveTSP(costs [][]int64, onRoute func([]int, int64)) (Result, error) {
	n := len(costs)
	if n == 0 {
		return Result{}, fmt.Errorf("a matriz de custos não pode ser vazia")
	}
	for i := range costs {
		if len(costs[i]) != n {
			return Result{}, fmt.Errorf("a matriz deve ser quadrada: linha %d tem %d colunas, esperado %d", i, len(costs[i]), n)
		}
	}

	permutation := make([]int, n-1)
	for i := range permutation {
		permutation[i] = i + 1
	}
	result := Result{N: n, BestCost: math.MaxInt64}
	started := time.Now()

	var enumerate func(int)
	enumerate = func(position int) {
		if position == len(permutation) {
			// Percorre a rota v1 -> ... -> v1: exatamente n arestas.
			route := make([]int, 0, n+1)
			route = append(route, 0)
			route = append(route, permutation...)
			route = append(route, 0)
			var routeCost int64
			for i := 0; i < n; i++ {
				routeCost += costs[route[i]][route[i+1]]
			}
			result.RoutesEvaluated++
			result.EdgesEvaluated += uint64(n)
			if routeCost < result.BestCost {
				result.BestCost = routeCost
				result.BestRoute = append([]int(nil), route...)
			}
			if onRoute != nil {
				onRoute(route, routeCost)
			}
			return
		}
		for i := position; i < len(permutation); i++ {
			permutation[position], permutation[i] = permutation[i], permutation[position]
			enumerate(position + 1)
			permutation[position], permutation[i] = permutation[i], permutation[position]
		}
	}
	enumerate(0)
	result.Duration = time.Since(started)
	return result, nil
}

func randomMatrix(n int, seed int64) [][]int64 {
	rng := rand.New(rand.NewSource(seed))
	matrix := make([][]int64, n)
	for i := range matrix {
		matrix[i] = make([]int64, n)
	}
	// Custos simétricos positivos: grafo completo não direcionado.
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			weight := int64(rng.Intn(100) + 1)
			matrix[i][j], matrix[j][i] = weight, weight
		}
	}
	return matrix
}

func factorial(n uint64) uint64 {
	value := uint64(1)
	for i := uint64(2); i <= n; i++ {
		value *= i
	}
	return value
}

func formatRoute(route []int) string {
	labels := make([]string, len(route))
	for i, city := range route {
		labels[i] = strconv.Itoa(city + 1)
	}
	return strings.Join(labels, " -> ")
}
