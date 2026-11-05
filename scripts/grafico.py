#!/usr/bin/env python3
"""Gera gráfico PNG e tabela CSV a partir do resultado de um benchmark TSP."""

import argparse
import csv
import math
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def output_file(directory, prefix, suffix):
    return tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=directory,
        prefix=prefix, suffix=suffix, delete=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resultado", help="CSV produzido pelo benchmark")
    parser.add_argument("--out", default="grafricos", help="diretório de saída (padrão: grafricos)")
    args = parser.parse_args()

    with open(args.resultado, encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"cidades", "rotas_teoricas", "rotas_avaliadas", "arestas_avaliadas", "tempo_ns"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            parser.error("CSV inválido: faltam colunas do benchmark")
        try:
            rows = sorted(({
                "n": int(row["cidades"]),
                "teoricas": int(row["rotas_teoricas"]),
                "rotas": int(row["rotas_avaliadas"]),
                "arestas": int(row["arestas_avaliadas"]),
                "tempo": int(row["tempo_ns"]) / 1_000_000,
            } for row in reader), key=lambda row: row["n"])
        except (ValueError, KeyError) as exc:
            parser.error(f"dados inválidos no CSV: {exc}")
    if not rows or any(row["n"] < 1 or row["n"] > 18 or row["tempo"] < 0 for row in rows):
        parser.error("o CSV deve conter medições válidas com 1 <= cidades <= 18")

    # Ajuste por mínimos quadrados, passando pela origem: c = Σ(x*y)/Σ(x²).
    factors = [math.factorial(row["n"]) for row in rows]
    c = sum(x * row["tempo"] for x, row in zip(factors, rows)) / sum(x * x for x in factors)
    for row, factor in zip(rows, factors):
        row["fatorial"] = factor
        row["estimado"] = c * factor

    os.makedirs(args.out, exist_ok=True)
    with output_file(args.out, "tabela_", ".csv") as table:
        writer = csv.writer(table)
        writer.writerow(["cidades", "n_fatorial", "rotas_teoricas", "rotas_avaliadas", "arestas_avaliadas", "tempo_medido_ms", "tempo_teorico_ms"])
        for row in rows:
            writer.writerow([row["n"], row["fatorial"], row["teoricas"], row["rotas"], row["arestas"], f'{row["tempo"]:.9f}', f'{row["estimado"]:.9f}'])

    fig, ax = plt.subplots(figsize=(10, 6))
    cities = [row["n"] for row in rows]
    ax.plot(cities, [row["tempo"] for row in rows], "o-", label="Tempo medido")
    ax.plot(cities, [row["estimado"] for row in rows], "o--", label="c · n!")
    if any(row["tempo"] > 0 for row in rows):
        ax.set_yscale("log")
        if any(row["tempo"] == 0 for row in rows):
            print("Aviso: tempos iguais a zero não aparecem na escala logarítmica.")
    ax.set_title(f"TSP: tempo medido e crescimento fatorial\nAjuste por mínimos quadrados: c = {c:.6g} ms/n!")
    ax.set_xlabel("Número de cidades (n)")
    ax.set_ylabel("Tempo (ms, escala logarítmica)" if any(row["tempo"] > 0 for row in rows) else "Tempo (ms)")
    ax.set_xticks(sorted(set(cities)))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    with output_file(args.out, "grafico_", ".png") as graph:
        graph_path = graph.name
    try:
        fig.savefig(graph_path, dpi=150)
    finally:
        plt.close(fig)
    print(f"Gráfico: {graph.name}\nTabela: {table.name}\nConstante c: {c:.9g} ms/n!")


if __name__ == "__main__":
    main()
