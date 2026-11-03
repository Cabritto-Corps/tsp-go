//go:build gpu

// GPU build of the TSP brute-force benchmark.
//
// Identical search and data generation to main.go, but the (n-1)! enumeration
// runs on the GPU (see tsp_gpu.cu). Inputs use the same seeded random matrix as
// the CPU path, so results.csv and results_gpu.csv are directly comparable.
//
// Build/run with the "gpu" tag, e.g.:
//
//	make gpu          # build ./.build/gpu/tsp-gpu
//	make gurun        # build and run
//	make libtsp_gpu.a  # build the CUDA archive first
//	go run -tags gpu .
//
// (main.go carries the complementary //go:build !gpu constraint, so the two
// main() functions never coexist in one build.)
package main

/*
#cgo gpu LDFLAGS: -L/opt/cuda/lib64 -lstdc++ -lcudart -L${SRCDIR}/.build/gpu -ltsp_gpu
#include <stdint.h>
#include "tsp_gpu.h"
*/
import "C"

import (
	"encoding/csv"
	"flag"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
	"unsafe"
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
	randomStart := flag.Bool("random-start", false, "sorteia a cidade inicial para cada quantidade de cidades")
	allStarts := flag.Bool("all-starts", false, "enumera as rotas para todas as cidades iniciais")
	flag.Bool("unique-output", true, "mantido por compatibilidade; os arquivos sempre têm nomes únicos")
	flag.Parse()
	if *randomStart && *allStarts {
		log.Fatal("use apenas um de -random-start e -all-starts")
	}

	if *minN < 1 || *maxN < *minN || *maxN > 18 {
		log.Fatalf("intervalo inválido: use 1 <= min-n <= max-n <= 18")
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatalf("não foi possível criar %q: %v", *outDir, err)
	}

	logPrefix, csvPrefix := "execucao_gpu_*.log", "results_gpu_*.csv"
	if *allStarts {
		logPrefix, csvPrefix = "execucao_sem_fixar_gpu_*.log", "sem_fixar_gpu_*.csv"
	}
	logFile, err := os.CreateTemp(*outDir, logPrefix)
	if err != nil {
		log.Fatalf("não foi possível criar o log: %v", err)
	}
	defer logFile.Close()
	logPath := logFile.Name()
	logger := log.New(io.MultiWriter(os.Stdout, logFile), "", log.LstdFlags)

	csvFile, err := os.CreateTemp(*outDir, csvPrefix)
	if err != nil {
		log.Fatalf("não foi possível criar o CSV: %v", err)
	}
	defer csvFile.Close()
	csvPath := csvFile.Name()
	csvWriter := csv.NewWriter(csvFile)
	if err := csvWriter.Write([]string{"cidades", "rotas_teoricas", "rotas_avaliadas", "arestas_avaliadas", "custo_minimo", "rota_otima", "tempo_ns", "tempo_ms"}); err != nil {
		log.Fatalf("erro ao escrever cabeçalho CSV: %v", err)
	}

	logger.Printf("TSP por força bruta (GPU) | todas as origens: %t | origem aleatória: %t | semente dos custos: %d | max-n: %d", *allStarts, *randomStart, *seed, *maxN)
	logger.Printf("Resultados serão gravados em %s", *outDir)
	logger.Print("As contagens incluem as duas passagens; cada rota é reavaliada para recuperar a rota ótima.")
	if *verbose {
		logger.Print("A listagem detalhada é reproduzida no CPU e não entra no tempo medido da GPU.")
	}

	// Same seeded matrix and top-left submatrix slicing as main.go so the
	// GPU and CPU solve identical instances.
	weights := randomMatrix(*maxN, *seed)
	startRNG := rand.New(rand.NewSource(time.Now().UnixNano()))
	for n := *minN; n <= *maxN; n++ {
		matrix := make([][]int64, n)
		for i := range matrix {
			matrix[i] = weights[i][:n]
		}
		start := 0
		if *randomStart {
			start = startRNG.Intn(n)
		}
		if *verbose {
			if *allStarts {
				for city := 0; city < n; city++ {
					logRoutes(matrix, city, logger)
				}
			} else {
				logRoutes(matrix, start, logger)
			}
		}

		result := solveTSPGPU(n, start, matrix)
		if *allStarts {
			logger.Printf("n=%d | origem 1/%d concluída", n, n)
			for city := 1; city < n; city++ {
				next := solveTSPGPU(n, city, matrix)
				result.Duration += next.Duration
				result.RoutesEvaluated += next.RoutesEvaluated
				result.EdgesEvaluated += next.EdgesEvaluated
				if next.BestCost < result.BestCost {
					result.BestCost, result.BestRoute = next.BestCost, next.BestRoute
				}
				logger.Printf("n=%d | origem %d/%d concluída", n, city+1, n)
			}
		}
		expected := factorial(uint64(n - 1))
		if *allStarts {
			expected *= uint64(n)
		}
		logger.Printf("n=%d | avaliações de rotas=%d (teórico em duas passagens=%d) | arestas=%d | custo mínimo=%d | rota=%s | tempo=%s",
			result.N, result.RoutesEvaluated, 2*expected, result.EdgesEvaluated,
			result.BestCost, formatRoute(result.BestRoute), result.Duration)
		if result.RoutesEvaluated != 2*expected || result.EdgesEvaluated != 2*expected*uint64(n) {
			log.Fatalf("contagens incorretas para n=%d: rotas=%d, arestas=%d", n, result.RoutesEvaluated, result.EdgesEvaluated)
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
	logger.Printf("Benchmark (GPU) concluído. Arquivos: %s e %s", logPath, csvPath)
}

// solveTSPGPU runs the brute-force search on the GPU via the CGO entry point.
func solveTSPGPU(n, start int, costs [][]int64) Result {
	dim := n * n
	host := make([]C.int32_t, dim)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			host[i*n+j] = C.int32_t(costs[i][j])
		}
	}
	hostPtr := unsafe.Pointer(&host[0])

	bestCost := C.int32_t(math.MaxInt32)
	bestRoute := make([]C.int32_t, n+1)
	var routes, edges C.uint64_t

	started := time.Now()
	if rc := C.tsp_gpu_run(C.int(n), C.int(start), (*C.int32_t)(hostPtr), &bestCost, (*C.int32_t)(unsafe.Pointer(&bestRoute[0])), &routes, &edges); rc != 0 {
		log.Fatalf("gpu run falhou para n=%d (rc=%d)", n, int(rc))
	}
	duration := time.Since(started)

	route := make([]int, n+1)
	for i := 0; i <= n; i++ {
		route[i] = int(bestRoute[i])
	}
	return Result{
		N:               n,
		BestRoute:       route,
		BestCost:        int64(bestCost),
		RoutesEvaluated: uint64(routes),
		EdgesEvaluated:  uint64(edges),
		Duration:        duration,
	}
}

func logRoutes(costs [][]int64, start int, logger *log.Logger) {
	n := len(costs)
	permutation := make([]int, 0, n-1)
	for city := 0; city < n; city++ {
		if city != start {
			permutation = append(permutation, city)
		}
	}

	var enumerate func(int)
	enumerate = func(position int) {
		if position == len(permutation) {
			route := make([]int, 0, n+1)
			route = append(route, start)
			route = append(route, permutation...)
			route = append(route, start)
			var cost int64
			for i := 0; i < n; i++ {
				cost += costs[route[i]][route[i+1]]
			}
			logger.Printf("rota %s | custo=%d", formatRoute(route), cost)
			return
		}
		for i := position; i < len(permutation); i++ {
			permutation[position], permutation[i] = permutation[i], permutation[position]
			enumerate(position + 1)
			permutation[position], permutation[i] = permutation[i], permutation[position]
		}
	}
	enumerate(0)
}

// randomMatrix builds a symmetric n x n matrix of positive weights (a complete
// undirected graph). Replicates main.go exactly for reproducible instances.
func randomMatrix(n int, seed int64) [][]int64 {
	rng := rand.New(rand.NewSource(seed))
	matrix := make([][]int64, n)
	for i := range matrix {
		matrix[i] = make([]int64, n)
	}
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
