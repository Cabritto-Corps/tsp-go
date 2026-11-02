# GPU build for the TSP brute-force benchmark.
#
# The CUDA code (tsp_gpu.cu) is compiled to a static archive with nvcc, then
# linked into the Go binary through CGO. The "gpu" tag selects main_gpu.go and
# (via the complementary !gpu tag on main.go) excludes the CPU main().

NVCC      ?= nvcc
ARCH      ?= sm_120          # RTX 5060 Ti (Blackwell)
CGO_TAGS  := gpu

BUILD := tsp-gpu

.PHONY: all gpu gurun clean

all: gpu

# Compile the CUDA kernel + C wrapper into a static archive. Link flags live in
# the #cgo directive of main_gpu.go (which expands ${SRCDIR} correctly).
libtsp_gpu.a: tsp_gpu.cu tsp_gpu.h
	$(NVCC) -arch=$(ARCH) -O3 -Xcompiler -fPIC -c tsp_gpu.cu -o tsp_gpu.o
	ar r "$@" tsp_gpu.o

gpu: libtsp_gpu.a
	go build -tags $(CGO_TAGS) -o $(BUILD) .

gurun: libtsp_gpu.a
	go run -tags $(CGO_TAGS) .

# CPU benchmark (unchanged main.go).
run:
	go run .

clean:
	rm -f tsp_gpu.o libtsp_gpu.a $(BUILD)
	rm -f resultados/execucao_gpu.log resultados/results_gpu.csv
