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

.PHONY: all gpu gurun gurun-random run run-random grafico clean FORCE libtsp_gpu.a

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

# Origem sorteada e arquivos exclusivos da GPU em resultados/.
gurun-random: libtsp_gpu.a
	go run -tags $(CGO_TAGS) . -random-start -unique-output $(ARGS)

# CPU benchmark (unchanged main.go).
run:
	go run . $(ARGS)

# Origem sorteada e arquivos exclusivos em resultados/.
run-random:
	go run . -random-start -unique-output $(ARGS)

# Exemplo: make grafico RESULTADO=resultados_123.csv
grafico:
	@test -n "$(RESULTADO)" || { echo 'Informe o CSV: make grafico RESULTADO="arquivo.csv"'; exit 1; }
	python3 grafico.py "resultados/$(RESULTADO)"

clean:
	rm -rf $(GPU_BUILD)

FORCE:
