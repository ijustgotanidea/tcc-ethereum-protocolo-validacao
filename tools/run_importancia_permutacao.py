# -*- coding: utf-8 -*-
"""Objetivo específico 6 do TCC: quais atributos sustentam as diferenças entre
os algoritmos. Importância por permutação (BREIMAN, 2001) no conjunto de teste,
medida pela queda da AUC-PR, nos dois protocolos do MVP.

Não escolhe nada de novo. Para cada protocolo, semente e algoritmo, refaz o
modelo com os hiperparâmetros que tools/run_mvp_experiment.py escolheu
(execucoes.csv), no mesmo treino, e confere que a AUC-PR de teste é a mesma do
registro. Depois embaralha cada atributo 5 vezes no teste; a importância é a
queda média da AUC-PR. O SMOTE fica fora da previsão, porque o Pipeline do
imbalanced-learn só o aplica no ajuste.

Saídas em Caio/mvp/:
- importancia_permutacao.csv   uma linha por protocolo, semente, algoritmo e
                               atributo: queda média e desvio nas 5 repetições
- importancia_permutacao.json  média das 10 sementes por protocolo e algoritmo,
                               posições, comparação entre protocolos e a
                               conferência da AUC-PR refeita

Dependências: as de tools/run_mvp_experiment.py, que precisa ter rodado nos dois
protocolos antes.

    python tools/run_importancia_permutacao.py [--erc20-vazio] [--so-analise]

--so-analise refaz só o JSON a partir do CSV já gravado. --erc20-vazio lê e grava
em Caio/mvp/erc20-vazio/, as rodadas de 2026-11-14, antes da correção dos vazios
ERC-20 (ver run_mvp_experiment.py).
"""
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mvp_experiment as M  # noqa: E402

REPETICOES = 5  # o padrão do scikit-learn; as 10 sementes já repetem o cálculo
TOPO = 5
ERC20_VAZIO = "--erc20-vazio" in sys.argv
RAIZ = M.SAIDA / "erc20-vazio" if ERC20_VAZIO else M.SAIDA
PROTOCOLOS = {"A": RAIZ, "B": RAIZ / "temporal"}
CSV = RAIZ / "importancia_permutacao.csv"
JSON = RAIZ / "importancia_permutacao.json"


def hiperparametros(texto):
    """O JSON de execucoes.csv volta com listas onde o grid tinha tuplas."""
    return {f"modelo__{k}": tuple(v) if isinstance(v, list) else v
            for k, v in json.loads(texto).items()}


def calcular():
    X, y, atributos, enderecos = M.base_de_trabalho(not ERC20_VAZIO)
    itr_b, ite_b, _ = M.divisao_temporal(M.instantes(enderecos), enderecos, y)
    linhas = []
    for protocolo, pasta in PROTOCOLOS.items():
        ex = pd.read_csv(pasta / "execucoes.csv")
        for s in M.SEMENTES:
            if protocolo == "A":
                itr, ite = train_test_split(np.arange(len(y)), test_size=M.TESTE,
                                            stratify=y, random_state=s)
            else:
                itr, ite = itr_b, ite_b
            for nome, (fabrica, _) in M.ALGORITMOS.items():
                t0 = time.perf_counter()
                reg = ex[(ex["semente"] == s) & (ex["algoritmo"] == nome)].iloc[0]
                est = M.pipeline(fabrica(s), s)
                est.set_params(**hiperparametros(reg["hiperparametros"]))
                est.fit(X[itr], y[itr])
                auc = average_precision_score(y[ite], M.escore(est, X[ite]))
                r = permutation_importance(
                    est, X[ite], y[ite], scoring="average_precision",
                    n_repeats=REPETICOES, random_state=s, n_jobs=-1)
                for j, a in enumerate(atributos):
                    linhas.append({
                        "protocolo": protocolo, "semente": s, "algoritmo": nome,
                        "atributo": a, "queda_media": r.importances_mean[j],
                        "queda_dp": r.importances_std[j],
                        "auc_pr_refeita": auc, "auc_pr_registrada": reg["auc_pr"]})
                print(f"{protocolo} semente {s} {nome:20s} AUC-PR {auc:.4f} "
                      f"(registro {reg['auc_pr']:.4f}) "
                      f"{time.perf_counter() - t0:5.1f}s", flush=True)
    return pd.DataFrame(linhas)


def analisar(df):
    algoritmos = list(M.ALGORITMOS)
    conferencia = (df.groupby(["protocolo", "semente", "algoritmo"])
                   [["auc_pr_refeita", "auc_pr_registrada"]].first())
    dif = (conferencia["auc_pr_refeita"] - conferencia["auc_pr_registrada"]).abs()

    media = (df.groupby(["protocolo", "algoritmo", "atributo"])["queda_media"]
             .agg(["mean", "std"]).reset_index())
    resumo = {"gerado_em": datetime.now().isoformat(timespec="seconds"),
              "repeticoes": REPETICOES, "sementes": M.SEMENTES,
              "conferencia_auc_pr_max_dif": float(dif.max()),
              "por_protocolo": {}}
    for p in PROTOCOLOS:
        m = media[media["protocolo"] == p]
        por_alg = {}
        for alg in algoritmos:
            a = m[m["algoritmo"] == alg].sort_values("mean", ascending=False)
            por_alg[alg] = {
                "importancia": dict(zip(a["atributo"], a["mean"])),
                "dp_entre_sementes": dict(zip(a["atributo"], a["std"])),
                "posicao": {at: i + 1 for i, at in enumerate(a["atributo"])},
                "topo": list(a["atributo"][:TOPO])}
        # média dos seis algoritmos, atributo a atributo
        geral = (m.groupby("atributo")["mean"].mean()
                 .sort_values(ascending=False))
        resumo["por_protocolo"][p] = {
            "por_algoritmo": por_alg,
            "media_dos_algoritmos": geral.to_dict(),
            "topo_media": list(geral.index[:TOPO])}

    # os atributos mais importantes mudam entre os protocolos?
    comparacao = {}
    for alg in algoritmos:
        ia = resumo["por_protocolo"]["A"]["por_algoritmo"][alg]["importancia"]
        ib = resumo["por_protocolo"]["B"]["por_algoritmo"][alg]["importancia"]
        at = sorted(ia)
        rho = spearmanr([ia[a] for a in at], [ib[a] for a in at]).statistic
        ta = set(resumo["por_protocolo"]["A"]["por_algoritmo"][alg]["topo"])
        tb = set(resumo["por_protocolo"]["B"]["por_algoritmo"][alg]["topo"])
        comparacao[alg] = {"spearman_a_b": float(rho),
                           "topo_em_comum": len(ta & tb)}
    resumo["comparacao_a_b"] = comparacao
    return resumo


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    if "--so-analise" in sys.argv:
        df = pd.read_csv(CSV)
    else:
        t0 = time.perf_counter()
        df = calcular()
        df.to_csv(CSV, index=False)
        print(f"tempo total: {time.perf_counter() - t0:.0f}s")
    resumo = analisar(df)
    JSON.write_text(json.dumps(resumo, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("conferência da AUC-PR, maior diferença:",
          resumo["conferencia_auc_pr_max_dif"])
    for p in PROTOCOLOS:
        print(p, resumo["por_protocolo"][p]["topo_media"])
    print(json.dumps(resumo["comparacao_a_b"], ensure_ascii=False, indent=1))
