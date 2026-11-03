# GPU build for the TSP brute-force benchmark.
#
# The CUDA code (tsp_gpu.cu) is compiled to a static archive with nvcc, then
# linked into the Go binary through CGO. The "gpu" tag selects main_gpu.go and
# (via the complementary !gpu tag on main.go) excludes the CPU main().

NVCC      ?= nvcc
AR        ?= ar
ARCH      ?= sm_120          # RTX 5060 Ti (Blackwell)
NVCCFLAGS ?= -O3 -Xcompiler -fPIC
CGO_TAGS  := gpu

GPU_BUILD := .build/gpu
GPU_OBJECT := $(GPU_BUILD)/tsp_gpu.o
GPU_LIBRARY := $(GPU_BUILD)/libtsp_gpu.a
GPU_CONFIG := $(GPU_BUILD)/build-config
BUILD := $(GPU_BUILD)/tsp-gpu

.PHONY: all gpu gurun gurun-random run run-random grafico grafico-mediana grafico-reducao clean FORCE libtsp_gpu.a

all: gpu

# Keep generated files out of the source tree and rebuild when compiler settings change.
$(GPU_CONFIG): FORCE
	@mkdir -p $(GPU_BUILD)
	@printf '%s\n' '$(NVCC) -arch=$(ARCH) $(NVCCFLAGS)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm $@.tmp; else mv $@.tmp $@; fi

$(GPU_OBJECT): tsp_gpu.cu tsp_gpu.h $(GPU_CONFIG)
	$(NVCC) -arch=$(ARCH) $(NVCCFLAGS) -c tsp_gpu.cu -o $@

$(GPU_LIBRARY): $(GPU_OBJECT)
	$(AR) rcs $@ $<

# Compatibility target used by all GPU build and run commands.
libtsp_gpu.a: $(GPU_LIBRARY)

gpu: libtsp_gpu.a
	go build -tags $(CGO_TAGS) -o $(BUILD) .

gurun: libtsp_gpu.a
	go run -tags $(CGO_TAGS) . $(ARGS)

# Enumera todas as origens, sem fixação global.
gurun-random: libtsp_gpu.a
	go run -tags $(CGO_TAGS) . -all-starts $(ARGS)

# CPU benchmark (unchanged main.go).
run:
	go run . $(ARGS)

# Enumera todas as origens, sem fixação global.
run-random:
	go run . -all-starts $(ARGS)

# Exemplo: make grafico RESULTADO=resultados_123.csv
grafico:
	@test -n "$(RESULTADO)" || { echo 'Informe o CSV: make grafico RESULTADO="arquivo.csv"'; exit 1; }
	python3 grafico.py "resultados/$(RESULTADO)"

# Agrega todos os CSVs do dispositivo e plota a mediana de tempo por n.
# Exemplos: make grafico-mediana DISPOSITIVO=cpu|gpu
grafico-mediana:
	@test "$(DISPOSITIVO)" = cpu -o "$(DISPOSITIVO)" = gpu || { echo 'Use DISPOSITIVO=cpu ou DISPOSITIVO=gpu'; exit 1; }
	python3 grafico_mediana.py "$(DISPOSITIVO)"

# Comparação teórica entre fixar a origem e enumerar todas as permutações.
grafico-reducao:
	python3 grafico_reducao.py

clean:
	rm -rf $(GPU_BUILD)

FORCE:
