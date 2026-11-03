#!/usr/bin/env python3
"""Gera gráfico e tabela a partir da mediana de todas as execuções CPU ou GPU."""

import argparse
import csv
import glob
import math
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dispositivo", choices=("cpu", "gpu"))
    parser.add_argument("--resultados", default="resultados", help="diretório com CSVs brutos")
    parser.add_argument("--out", default="grafricos", help="diretório de saída")
    args = parser.parse_args()

    pattern = "results_gpu*.csv" if args.dispositivo == "gpu" else "resultados*.csv"
    paths = sorted(glob.glob(os.path.join(args.resultados, pattern)))
    if not paths:
        parser.error(f"nenhum CSV encontrado para {args.dispositivo}: {args.resultados}/{pattern}")

    samples = {}
    for path in paths:
        try:
            with open(path, encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                required = {"cidades", "rotas_teoricas", "rotas_avaliadas", "arestas_avaliadas", "tempo_ns"}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    print(f"Ignorando CSV incompatível: {path}")
                    continue
                for row in reader:
                    n = int(row["cidades"])
                    samples.setdefault(n, {"times": [], "routes": set(), "edges": set(), "theoretical": set()})
                    samples[n]["times"].append(int(row["tempo_ns"]) / 1_000_000)
                    routes = int(row["rotas_avaliadas"])
                    edges = int(row["arestas_avaliadas"])
                    # CSVs antigos da GPU contam uma passagem; os novos contam duas.
                    if args.dispositivo == "gpu" and routes == 2 * math.factorial(n - 1) and edges == 2 * math.factorial(n):
                        routes //= 2
                        edges //= 2
                    samples[n]["routes"].add(routes)
                    samples[n]["edges"].add(edges)
                    samples[n]["theoretical"].add(int(row["rotas_teoricas"]))
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"erro lendo {path}: {exc}")

    rows = []
    for n, values in sorted(samples.items()):
        if len(values["routes"]) != 1 or len(values["edges"]) != 1 or len(values["theoretical"]) != 1:
            parser.error(f"contagens divergentes entre execuções para n={n}")
        rows.append({
            "n": n,
            "factorial": math.factorial(n),
            "theoretical": values["theoretical"].pop(),
            "routes": values["routes"].pop(),
            "edges": values["edges"].pop(),
            "median_ms": statistics.median(values["times"]),
            "samples": len(values["times"]),
        })
    if not rows:
        parser.error("nenhuma medição válida encontrada")

    c = sum(math.factorial(row["n"]) * row["median_ms"] for row in rows) / sum(
        math.factorial(row["n"]) ** 2 for row in rows
    )
    os.makedirs(args.out, exist_ok=True)
    label = args.dispositivo.upper()
    table_path = os.path.join(args.out, f"tabela_{label}_mediana.csv")
    with open(table_path, "w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["cidades", "n_fatorial", "rotas_teoricas", "rotas_avaliadas", "arestas_avaliadas", "amostras", "mediana_ms", "ajuste_ms"])
        for row in rows:
            writer.writerow([row["n"], row["factorial"], row["theoretical"], row["routes"], row["edges"], row["samples"], f'{row["median_ms"]:.9f}', f'{c * row["factorial"]:.9f}'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ns = [row["n"] for row in rows]
    ax.plot(ns, [row["median_ms"] for row in rows], "o-", label="Mediana do tempo")
    ax.plot(ns, [c * row["factorial"] for row in rows], "o--", label="c · n!")
    if any(row["median_ms"] > 0 for row in rows):
        ax.set_yscale("log")
    ax.set_title(f"TSP: mediana do tempo ({label})\nAjuste por mínimos quadrados: c = {c:.6g} ms/n!")
    ax.set_xlabel("Número de cidades (n)")
    ax.set_ylabel("Tempo (ms, escala logarítmica)")
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    graph_path = os.path.join(args.out, f"Grafico_{label}_mediana.png")
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)
    print(f"Execuções lidas: {len(paths)} CSV(s)")
    print(f"Gráfico: {graph_path}\nTabela: {table_path}\nConstante c: {c:.9g} ms/n!")


if __name__ == "__main__":
    main()
