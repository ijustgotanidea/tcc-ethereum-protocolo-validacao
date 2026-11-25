# -*- coding: utf-8 -*-
"""Figura 1 do Capítulo 4: a AUC-PR de cada execução do MVP, por algoritmo e
protocolo.

Lê Caio/mvp/execucoes.csv (protocolo A) e Caio/mvp/temporal/execucoes.csv
(protocolo B), gravados por tools/run_mvp_experiment.py, e não roda nada de novo.
Grava Caio/mvp/figura_auc_pr_execucoes.png, que build_postagem_docx.py VIII
insere no documento. Tons de cinza, para sair legível impresso.

Dependências: pandas, matplotlib.

    python tools/build_figura_mvp.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MVP = ROOT / "Caio" / "mvp"
SAIDA = MVP / "figura_auc_pr_execucoes.png"
LARGURA_CM, ALTURA_CM = 16.0, 8.5  # 16 cm: a largura de postagem_content.FIGURA1

PROTOCOLOS = [
    ("Protocolo A (divisão aleatória)", MVP / "execucoes.csv",
     dict(marker="o", facecolor="black", edgecolor="black")),
    ("Protocolo B (divisão temporal)", MVP / "temporal" / "execucoes.csv",
     dict(marker="s", facecolor="white", edgecolor="0.35")),
]


def main():
    plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    dados = [(rotulo, pd.read_csv(csv), estilo) for rotulo, csv, estilo in PROTOCOLOS]
    a = dados[0][1]
    a = a[a["algoritmo"] != "Classe majoritária"]
    # de cima para baixo, na ordem do protocolo A
    ordem = a.groupby("algoritmo")["auc_pr"].mean().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(LARGURA_CM / 2.54, ALTURA_CM / 2.54))
    for k, (rotulo, ex, estilo) in enumerate(dados):
        desloc = -0.17 if k == 0 else 0.17
        for i, alg in enumerate(ordem):
            v = ex.loc[ex["algoritmo"] == alg, "auc_pr"].to_numpy()
            ax.scatter(v, [i + desloc] * len(v), s=16, linewidths=0.8, zorder=3,
                       label=rotulo if i == 0 else None, **estilo)
            ax.plot([v.mean()] * 2, [i + desloc - 0.13, i + desloc + 0.13],
                    color="black", linewidth=1.2, zorder=4)

    ax.set_yticks(range(len(ordem)))
    ax.set_yticklabels(ordem)
    ax.invert_yaxis()
    menor = min(ex.loc[ex["algoritmo"] != "Classe majoritária", "auc_pr"].min()
                for _, ex, _ in dados)
    ax.set_xlim(np.floor(menor / 0.05) * 0.05, 1.0)  # múltiplo de 0,05 abaixo do menor ponto
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    ax.set_xlabel("AUC-PR no conjunto de teste (cada ponto é uma semente; o traço é a média)")
    ax.grid(axis="x", color="0.85", linewidth=0.6, zorder=0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    # abaixo do eixo, para não cobrir pontos qualquer que seja a faixa dos dados
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
              frameon=False)
    fig.tight_layout()
    fig.savefig(SAIDA, dpi=300)
    print(SAIDA)


if __name__ == "__main__":
    main()
