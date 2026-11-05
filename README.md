# TSP por força bruta

Projeto em Go que resolve o Problema do Caixeiro Viajante por busca exaustiva. Inclui uma versão para CPU, uma versão opcional para GPU com CUDA e scripts para gerar gráficos dos resultados.

## Requisitos

- Go 1.27.3 ou superior.
- Python 3 e as dependências listadas em `requirements.txt` (somente para gerar gráficos).
- Para a versão GPU: CUDA Toolkit, `nvcc`, compilador C++ e uma GPU NVIDIA compatível.

## Executar na CPU

Na raiz do repositório:

```bash
go run .
```

O programa testa de 3 a 10 cidades por padrão e salva um CSV e um log em `resultados/`. Para escolher o intervalo:

```bash
go run . -min-n 3 -max-n 8
```

Também é possível usar `make run` no lugar de `go run .`.

## Executar na GPU (opcional)

Com CUDA instalado (incluindo as bibliotecas em `/opt/cuda/lib64`), compile e execute:

```bash
make gpu
make gurun
```

A configuração padrão do Makefile usa `nvcc` e `sm_120`. Para ajustar a arquitetura da GPU, informe `ARCH`, por exemplo: `make gpu ARCH=sm_89`.

## Gráficos (opcional)

Instale as dependências Python e gere um gráfico a partir de um CSV:

```bash
python3 -m pip install -r requirements.txt
make grafico RESULTADO=nome-do-arquivo.csv
```

Os gráficos e tabelas gerados ficam em `grafricos/`. Para gerar um gráfico com a mediana das execuções, use `make grafico-mediana DISPOSITIVO=cpu` ou `DISPOSITIVO=gpu`.

## Testes

```bash
go test ./...
```

## Estrutura

- `main.go`, `main_test.go`: implementação e testes da CPU.
- `main_gpu.go`, `main_gpu_test.go`, `tsp_gpu.cu`, `tsp_gpu.h`: implementação CUDA.
- `scripts/`: scripts Python para geração de gráficos e tabelas.
- `resultados/`, `grafricos/`: resultados e gráficos do projeto.
- `relatorio.tex`, `amostras.tex`: arquivos do relatório.
