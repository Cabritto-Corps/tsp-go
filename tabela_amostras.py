#!/usr/bin/env python3
"""Gera tabelas LaTeX com todas as medições individuais dos CSVs existentes."""

import csv
import glob
import math
import statistics


def numero(valor):
    return f"{valor:.6f}".replace(".", ",")


def main():
    linhas = ["% Gerado por tabela_amostras.py; tempos em milissegundos."]
    for dispositivo, padrao in (("CPU, origem fixa", "resultados/resultados*.csv"),
                                ("GPU, origem fixa", "resultados/results_gpu*.csv"),
                                ("CPU, todas as origens", "resultados/sem_fixar_cpu_*.csv"),
                                ("GPU, todas as origens", "resultados/sem_fixar_gpu_*.csv")):
        arquivos = sorted(glob.glob(padrao))
        amostras = {}
        contagens = {}
        for indice, caminho in enumerate(arquivos, 1):
            with open(caminho, encoding="utf-8", newline="") as fonte:
                for registro in csv.DictReader(fonte):
                    n = int(registro["cidades"])
                    tempo = int(registro["tempo_ns"]) / 1_000_000
                    if n < 1 or tempo < 0 or not math.isfinite(tempo):
                        raise ValueError(f"Medição inválida em {caminho}: {registro}")
                    rotas = int(registro["rotas_avaliadas"])
                    arestas = int(registro["arestas_avaliadas"])
                    esperado = math.factorial(n) if "todas as origens" in dispositivo else math.factorial(n - 1)
                    passagens = 2 if dispositivo == "GPU, todas as origens" else 1
                    if dispositivo == "GPU, origem fixa" and rotas == 2 * esperado:
                        passagens = 2
                    if rotas != passagens * esperado or arestas != passagens * esperado * n:
                        raise ValueError(f"Contagens inesperadas em {caminho}, n={n}")
                    amostras.setdefault(n, []).append((indice, tempo))
                    contagens.setdefault(n, set()).add((rotas, arestas))
        if not amostras:
            continue
        linhas += [
            rf"\subsection{{Medições individuais: {dispositivo}}}",
            "As letras identificam os arquivos de origem; tempos em milissegundos. Mínimo, mediana e máximo estão arredondados a três algarismos significativos.",
            r"\begin{itemize}",
        ]
        for indice, caminho in enumerate(arquivos, 1):
            nome = caminho.replace("_", r"\_")
            linhas.append(rf"\item {chr(64 + indice)}: \texttt{{{nome}}}")
        linhas += [
            r"\end{itemize}",
            r"\begin{small}",
            r"\begin{longtable}{r p{5.2cm} r r r}",
            r"\toprule",
            r"$n$ & Amostras (arquivo: tempo) & Mín. & Mediana & Máx. \\",
            r"\midrule\endfirsthead",
            r"\toprule $n$ & Amostras (arquivo: tempo) & Mín. & Mediana & Máx. \\",
            r"\midrule\endhead",
        ]
        for n, valores in sorted(amostras.items()):
            tempos = [valor for _, valor in valores]
            detalhes = "; ".join(f"{chr(64 + indice)}: {numero(valor)}" for indice, valor in valores)
            linhas.append(rf"{n} & {detalhes} & {min(tempos):.3g} & {statistics.median(tempos):.3g} & {max(tempos):.3g} \\")
        linhas += [r"\bottomrule", r"\end{longtable}", r"\end{small}"]
        linhas += [r"\noindent Contagens efetivamente registradas nos CSVs (rotas e arestas):",
                   r"\begin{longtable}{rrr}", r"\toprule",
                   r"$n$ & Avaliações de rotas & Avaliações de arestas \\",
                   r"\midrule\endfirsthead", r"\toprule",
                   r"$n$ & Avaliações de rotas & Avaliações de arestas \\",
                   r"\midrule\endhead"]
        for n, pares in sorted(contagens.items()):
            for rotas, arestas in sorted(pares):
                linhas.append(rf"{n} & {rotas} & {arestas} \\")
        linhas += [r"\bottomrule", r"\end{longtable}"]
    with open("amostras.tex", "w", encoding="utf-8") as destino:
        destino.write("\n".join(linhas) + "\n")


if __name__ == "__main__":
    main()
