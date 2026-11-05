#!/usr/bin/env python3
"""Compara o espaço de busca com origem fixa e sem fixar a origem."""

import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-n", type=int, default=3)
    parser.add_argument("--max-n", type=int, default=17)
    parser.add_argument("--out", default="grafricos")
    args = parser.parse_args()
    if args.min_n < 2 or args.max_n < args.min_n:
        parser.error("use 2 <= min-n <= max-n")

    rows = []
    for n in range(args.min_n, args.max_n + 1):
        routes_fixed = math.factorial(n - 1)
        routes_unfixed = math.factorial(n)
        rows.append({
            "n": n,
            "routes_fixed": routes_fixed,
            "routes_unfixed": routes_unfixed,
            "edges_fixed": n * routes_fixed,
            "edges_unfixed": n * routes_unfixed,
        })

    os.makedirs(args.out, exist_ok=True)
    table_path = os.path.join(args.out, "tabela_reducao_origem.csv")
    with open(table_path, "w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    ns = [row["n"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(ns, [row["routes_fixed"] for row in rows], "o-", label=r"Origem fixa: $(n-1)!$")
    axes[0].plot(ns, [row["routes_unfixed"] for row in rows], "o--", label=r"Sem fixar: $n!$")
    axes[0].set_title("Rotas enumeradas")
    axes[1].plot(ns, [row["edges_fixed"] for row in rows], "o-", label=r"Origem fixa: $n!$")
    axes[1].plot(ns, [row["edges_unfixed"] for row in rows], "o--", label=r"Sem fixar: $n\cdot n!$")
    axes[1].set_title("Avaliações de arestas")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("Número de cidades (n)")
        ax.set_ylabel("Quantidade (escala logarítmica)")
        ax.set_xticks(ns)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("Efeito de fixar a cidade inicial")
    fig.tight_layout()
    graph_path = os.path.join(args.out, "Reducao_busca.png")
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)
    print(f"Gráfico: {graph_path}\nTabela: {table_path}")


if __name__ == "__main__":
    main()
