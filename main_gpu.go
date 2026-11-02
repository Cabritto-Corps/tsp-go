//go:build gpu

// GPU build of the TSP brute-force benchmark.
//
// Identical search and data generation to main.go, but the (n-1)! enumeration
// runs on the GPU (see tsp_gpu.cu). Inputs use the same seeded random matrix as
// the CPU path, so results.csv and results_gpu.csv are directly comparable.
//
// Build/run with the "gpu" tag, e.g.:
//
//	make gpu          # build ./tsp-gpu
//	make gurun        # build and run
//	go run -tags gpu .
//
// (main.go carries the complementary //go:build !gpu constraint, so the two
// main() functions never coexist in one build.)
package main

/*
#cgo gpu LDFLAGS: -L/opt/cuda/lib64 -lstdc++ -lcudart -L${SRCDIR} -ltsp_gpu
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
	randomStart := flag.Bool("random-start", false, "sorteia a cidade inicial para cada quantidade de cidades")
	flag.Bool("unique-output", true, "mantido por compatibilidade; os arquivos sempre têm nomes únicos")
	flag.Parse()

	if *minN < 1 || *maxN < *minN || *maxN > 18 {
		log.Fatalf("intervalo inválido: use 1 <= min-n <= max-n <= 18")
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatalf("não foi possível criar %q: %v", *outDir, err)
	}

	logFile, err := os.CreateTemp(*outDir, "execucao_gpu_*.log")
	if err != nil {
		log.Fatalf("não foi possível criar o log: %v", err)
	}
	defer logFile.Close()
	logPath := logFile.Name()
	logger := log.New(io.MultiWriter(os.Stdout, logFile), "", log.LstdFlags)

	csvFile, err := os.CreateTemp(*outDir, "results_gpu_*.csv")
	if err != nil {
		log.Fatalf("não foi possível criar o CSV: %v", err)
	}
	defer csvFile.Close()
	csvPath := csvFile.Name()
	csvWriter := csv.NewWriter(csvFile)
	if err := csvWriter.Write([]string{"cidades", "rotas_teoricas", "rotas_avaliadas", "arestas_avaliadas", "custo_minimo", "rota_otima", "tempo_ns", "tempo_ms"}); err != nil {
		log.Fatalf("erro ao escrever cabeçalho CSV: %v", err)
	}

	logger.Printf("TSP por força bruta (GPU) | origem aleatória: %t | semente dos custos: %d", *randomStart, *seed)
	logger.Printf("Resultados serão gravados em %s", *outDir)
	logger.Print("As contagens representam a primeira passagem; uma segunda passagem reavalia as rotas para recuperar a rota ótima.")

	// Same seeded matrix and top-left submatrix slicing as main.go so the
	// GPU and CPU solve identical instances.
	weights := randomMatrix(*maxN, *seed)
	startRNG := rand.New(rand.NewSource(time.Now().UnixNano()))
	for n := *minN; n <= *maxN; n++ {
		cities := make([]int, n)
		for i := range cities {
			cities[i] = i
		}
		if *randomStart {
			start := startRNG.Intn(n)
			cities[0], cities[start] = cities[start], cities[0]
		}
		matrix := make([][]int64, n)
		for i := range matrix {
			matrix[i] = make([]int64, n)
			for j := range matrix[i] {
				matrix[i][j] = weights[cities[i]][cities[j]]
			}
		}

		result := solveTSPGPU(n, matrix)
		for i, city := range result.BestRoute {
			result.BestRoute[i] = cities[city]
		}
		expected := factorial(uint64(n - 1))
		logger.Printf("n=%d | rotas=%d (teórico=%d) | arestas=%d | custo mínimo=%d | rota=%s | tempo=%s",
			result.N, result.RoutesEvaluated, expected, result.EdgesEvaluated,
			result.BestCost, formatRoute(result.BestRoute), result.Duration)
		if result.RoutesEvaluated != expected {
			log.Fatalf("contagem incorreta de rotas para n=%d: esperado %d, obtido %d", n, expected, result.RoutesEvaluated)
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
func solveTSPGPU(n int, costs [][]int64) Result {
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
	if rc := C.tsp_gpu_run(C.int(n), (*C.int32_t)(hostPtr), &bestCost, (*C.int32_t)(unsafe.Pointer(&bestRoute[0])), &routes, &edges); rc != 0 {
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
