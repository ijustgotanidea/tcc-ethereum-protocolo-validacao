# -*- coding: utf-8 -*-
"""MVP do TCC (Capítulo 4): os dois protocolos, com os seis algoritmos.

Protocolo A, divisão aleatória: estratificada 80/20, repetida com 10 sementes.
Protocolo B, divisão temporal dentro de cada classe (--temporal): em cada classe,
as contas ordenadas pelo instante da primeira transação
(tools/collect_first_tx.py); as 80% mais antigas treinam, as 20% mais recentes
testam, e os hiperparâmetros são escolhidos nas 20% mais recentes do treino de
cada classe. A divisão é uma só; as 10 sementes mudam o SMOTE e os modelos.

Por que dentro de cada classe (2026-11-14): no conjunto do Kaggle a data separa
as classes quase sozinha. As lícitas começam até 2017-06-13 e as ilícitas de
2017-03-15 a 2020-05-14, e a data sozinha dá AUC-ROC de 0,998. Um corte global
80/20 deixava 5 ilícitas em 7.430 contas de treino e 88,9% de ilícitas no teste.

Os dois rodam sobre a base de trabalho da §3.2 (mesmas regras de limpeza de
tools/build_seminario2_deck.py).

Igual para os seis algoritmos e nos dois protocolos, como a §3.4 do Seminário II
promete:
- imputação pela mediana, transformação logarítmica com sinal, normalização e
  SMOTE ajustados só no treino, dentro de um Pipeline do imbalanced-learn, então
  cada dobra da validação refaz o SMOTE;
- seis configurações de hiperparâmetros por algoritmo, escolhidas pela AUC-PR em
  validação dentro do treino (A: 3 dobras estratificadas; B: os 20% mais
  recentes do treino);
- as mesmas sementes.

A transformação logarítmica entrou depois da primeira rodada (2026-11-14): 32 dos
38 atributos têm assimetria acima de 10 (mediana 47,4), e sem ela a regressão
logística e o SVM ficaram abaixo do piso de acurácia. Ela não tem parâmetro
ajustado, então não vaza o teste.

Os vazios dos 16 atributos de tokens ERC-20 viram zero antes da divisão
(2026-11-25). Eles aparecem em 549 contas da base, todas ilícitas, e nenhuma
conta ilícita com dados registra zero transações ERC-20, contra 57,4% das lícitas:
o arquivo marca a falta de atividade com zero nas lícitas e com vazio nas
ilícitas. Antes da correção, a imputação pela mediana do treino decidia se esse
sinal chegava ao modelo (mediana 1 em 4 sementes de A e no treino de B). Depois
dela não sobra nenhum valor ausente, nem linha repetida; a imputação fica só como
proteção.

Saídas em Caio/mvp/ (A) e Caio/mvp/temporal/ (B):
- execucoes.csv  um registro por execução (entrada e saída), o instrumento da §3.3
- predicoes.csv  rótulo, classe prevista e escore de cada conta de teste
- resumo.json    médias, ordens, τ de Kendall, McNemar com Holm, bootstrap; em B,
                 também os períodos de cada parte e a comparação com A
- ambiente.json  versões e máquina

Dependências: pandas, scikit-learn, imbalanced-learn, xgboost, lightgbm, scipy,
statsmodels.

    python tools/run_mvp_experiment.py [--temporal | --sem-log] [--erc20-vazio] [--so-analise]

--so-analise refaz só resumo.json a partir dos CSVs já gravados. Em B, a
comparação lê os CSVs de A, então A tem de ter rodado antes.

--erc20-vazio reproduz as rodadas de 2026-11-14, com os vazios ERC-20 deixados
para a mediana, em Caio/mvp/erc20-vazio/. A primeira rodada, sem o log, é
--erc20-vazio --sem-log, em Caio/mvp/erc20-vazio/sem-log/.
"""

import itertools
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from scipy.stats import kendalltau
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, PredefinedSplit,
                                     StratifiedKFold, train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "raw/tcc/dados/transaction_dataset.csv"
DATAS = ROOT / "raw/tcc/dados/primeira_transacao_etherscan.csv"
SAIDA = ROOT / "Caio/mvp"

SEMENTES = list(range(10))
TESTE = 0.20
DOBRAS = 3
BOOTSTRAP = 1000
ALFA = 0.05

# Seis configurações por algoritmo: o mesmo orçamento de ajuste para todos.
ALGORITMOS = {
    "Regressão logística": (
        lambda s: LogisticRegression(max_iter=5000, random_state=s),
        {"C": [0.01, 0.1, 1, 10, 100, 1000]}),
    "Floresta aleatória": (
        lambda s: RandomForestClassifier(n_estimators=300, random_state=s,
                                         n_jobs=1),
        {"max_depth": [None, 10, 20], "min_samples_leaf": [1, 5]}),
    "XGBoost": (
        lambda s: XGBClassifier(n_estimators=300, random_state=s, n_jobs=1,
                                eval_metric="logloss"),
        {"max_depth": [3, 6, 9], "learning_rate": [0.05, 0.2]}),
    "LightGBM": (
        lambda s: LGBMClassifier(n_estimators=300, random_state=s, n_jobs=1,
                                 verbose=-1),
        {"num_leaves": [15, 31, 63], "learning_rate": [0.05, 0.2]}),
    "SVM": (
        lambda s: SVC(kernel="rbf", random_state=s),
        {"C": [0.1, 1, 10], "gamma": ["scale", 0.01]}),
    "MLP": (
        lambda s: MLPClassifier(max_iter=500, early_stopping=True,
                                random_state=s),
        {"hidden_layer_sizes": [(64,), (128,), (128, 64)],
         "alpha": [1e-4, 1e-2]}),
}


# ------------------------------------------------------------------ dados
def base_de_trabalho(erc20_zero=True):
    """As regras de limpeza da §3.2, idênticas às do deck do Seminário II, mais
    os vazios ERC-20 como zero (erc20_zero=False reproduz 2026-11-14)."""
    df = pd.read_csv(CSV)
    df.columns = [c.strip() for c in df.columns]
    ids = ["Unnamed: 0", "Index", "Address", "FLAG"]
    feats = [c for c in df.columns if c not in ids]
    num = df[feats].select_dtypes("number").columns.tolist()
    atributos = [c for c in num if df[c].dropna().nunique() > 1]
    addr = df["Address"].str.strip().str.lower()
    malformado = ~addr.str.fullmatch(r"0x[0-9a-f]{40}")
    repetido = df.duplicated(subset=feats)
    assert (malformado & ~repetido).sum() == 0
    d = df.loc[~repetido]
    d = d.loc[~d["Address"].str.lower().duplicated()]
    assert not d.duplicated(subset=atributos).any()
    X = d[atributos].to_numpy(dtype=float)
    if erc20_zero:
        erc20 = [i for i, a in enumerate(atributos) if a.startswith("ERC20")
                 or a == "Total ERC20 tnxs"]
        X[:, erc20] = np.nan_to_num(X[:, erc20], nan=0.0)
    y = d["FLAG"].to_numpy(dtype=int)
    enderecos = d["Address"].str.strip().str.lower().to_numpy()
    return X, y, atributos, enderecos


def instantes(enderecos):
    """Instante (Unix, s) da primeira transação de cada conta; NaN sem data."""
    d = pd.read_csv(DATAS)
    d = d[d["status"] == "ok"].set_index("address")["primeira_ts"]
    return pd.Series(enderecos).map(d).to_numpy(dtype=float)


def divisao_temporal(ts, enderecos, y):
    """Divisão temporal dentro de cada classe. Devolve os índices de treino e
    de teste e o vetor de dobras da validação (0 nas 20% mais recentes do
    treino de cada classe, -1 no resto). Empate de instante desfeito pelo
    endereço."""
    itr, ite, dobra = [], [], []
    for c in (0, 1):
        ix = np.flatnonzero(y == c)
        ix = ix[np.lexsort((enderecos[ix], ts[ix]))]
        n_te = int(np.ceil(TESTE * len(ix)))
        tr = ix[:-n_te]
        n_va = int(np.ceil(TESTE * len(tr)))
        itr.append(tr)
        ite.append(ix[-n_te:])
        dobra.append(np.r_[np.full(len(tr) - n_va, -1), np.zeros(n_va, int)])
    return np.concatenate(itr), np.concatenate(ite), np.concatenate(dobra)


def log_com_sinal(Z):
    """sinal(x)·log(1 + |x|): comprime as caudas longas e preserva o zero."""
    return np.sign(Z) * np.log1p(np.abs(Z))


def pipeline(modelo, semente, log=True):
    passos = [("imputacao", SimpleImputer(strategy="median"))]
    if log:
        passos.append(("log", FunctionTransformer(log_com_sinal)))
    passos += [("normalizacao", StandardScaler()),
               ("smote", SMOTE(random_state=semente)),
               ("modelo", modelo)]
    return Pipeline(passos)


def escore(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def metricas(y, pred, sc):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "acuracia": accuracy_score(y, pred),
        "precisao": precision_score(y, pred, zero_division=0),
        "revocacao": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "auc_pr": average_precision_score(y, sc),
        "auc_roc": roc_auc_score(y, sc),
    }


# ------------------------------------------------------------------ execução
def executar(X, y, log=True, temporal=None):
    """temporal: None para o protocolo A, ou (instantes, endereços) para o B."""
    registros, predicoes = [], []
    if temporal is not None:
        itr_b, ite_b, dobra_b = divisao_temporal(*temporal, y)
    for s in SEMENTES:
        if temporal is None:
            itr, ite = train_test_split(np.arange(len(y)), test_size=TESTE,
                                        stratify=y, random_state=s)
            cv = StratifiedKFold(DOBRAS, shuffle=True, random_state=s)
            protocolo = "A-aleatoria"
        else:
            itr, ite = itr_b, ite_b
            cv = PredefinedSplit(dobra_b)
            protocolo = "B-temporal"
        Xtr, Xte, ytr, yte = X[itr], X[ite], y[itr], y[ite]
        base = {"protocolo": protocolo, "semente": s,
                "n_treino": len(itr), "n_teste": len(ite),
                "pos_treino": int(ytr.sum()), "pos_teste": int(yte.sum())}

        piso = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
        registros.append({**base, "algoritmo": "Classe majoritária",
                          "hiperparametros": "{}", "auc_pr_validacao": None,
                          "tempo_s": 0.0,
                          **metricas(yte, piso.predict(Xte),
                                     piso.predict_proba(Xte)[:, 1])})

        pred_s = {"semente": s, "indice": ite, "y": yte}
        for nome, (fabrica, grade) in ALGORITMOS.items():
            t0 = time.perf_counter()
            busca = GridSearchCV(
                pipeline(fabrica(s), s, log),
                {f"modelo__{k}": v for k, v in grade.items()},
                scoring="average_precision", cv=cv, n_jobs=-1, refit=True)
            busca.fit(Xtr, ytr)
            dt = time.perf_counter() - t0
            pred, sc = busca.predict(Xte), escore(busca, Xte)
            hp = {k.removeprefix("modelo__"): v
                  for k, v in busca.best_params_.items()}
            registros.append({**base, "algoritmo": nome,
                              "hiperparametros": json.dumps(hp, default=str),
                              "auc_pr_validacao": busca.best_score_,
                              "tempo_s": dt, **metricas(yte, pred, sc)})
            pred_s[f"pred|{nome}"], pred_s[f"escore|{nome}"] = pred, sc
            print(f"semente {s} {nome:20s} AUC-PR "
                  f"{registros[-1]['auc_pr']:.4f} acc "
                  f"{registros[-1]['acuracia']:.4f} {dt:6.1f}s", flush=True)
        predicoes.append(pd.DataFrame(pred_s))
    return pd.DataFrame(registros), pd.concat(predicoes, ignore_index=True)


# ------------------------------------------------------------------ análise
def ordem(valores):
    """Posição de cada algoritmo, 1 = melhor."""
    return {n: i + 1 for i, n in enumerate(
        sorted(valores, key=lambda n: -valores[n]))}


def pares_invertidos(a, b):
    nomes = list(a)
    return [(i, j) for i, j in itertools.combinations(nomes, 2)
            if (a[i] - a[j]) * (b[i] - b[j]) < 0]


def analisar(ex, pr):
    nomes = list(ALGORITMOS)
    mod = ex[ex["algoritmo"].isin(nomes)]
    cols = ["acuracia", "precisao", "revocacao", "f1", "auc_pr", "auc_roc",
            "tempo_s"]
    media = mod.groupby("algoritmo")[cols].mean().loc[nomes]
    dp = mod.groupby("algoritmo")[cols].std(ddof=1).loc[nomes]
    piso = ex[ex["algoritmo"] == "Classe majoritária"][cols].mean()

    acc, apr, f1 = (media[c].to_dict() for c in ("acuracia", "auc_pr", "f1"))
    tau_media = kendalltau([acc[n] for n in nomes], [apr[n] for n in nomes])

    rng = np.random.default_rng(2026)
    por_semente, mcn, boot = [], {}, {}
    for s, g in pr.groupby("semente"):
        m = mod[mod["semente"] == s].set_index("algoritmo")
        a_s, p_s = m["acuracia"].to_dict(), m["auc_pr"].to_dict()
        t = kendalltau([a_s[n] for n in nomes], [p_s[n] for n in nomes])
        por_semente.append({"semente": int(s), "tau": t.statistic,
                            "invertidos": len(pares_invertidos(a_s, p_s)),
                            "ordem_acuracia": ordem(a_s),
                            "ordem_auc_pr": ordem(p_s)})

        y = g["y"].to_numpy()
        # McNemar exato por par, Holm sobre os 15 pares (DIETTERICH, 1998)
        pv, pares = [], list(itertools.combinations(nomes, 2))
        for i, j in pares:
            ci = g[f"pred|{i}"].to_numpy() == y
            cj = g[f"pred|{j}"].to_numpy() == y
            tab = [[int((ci & cj).sum()), int((ci & ~cj).sum())],
                   [int((~ci & cj).sum()), int((~ci & ~cj).sum())]]
            pv.append(mcnemar(tab, exact=True).pvalue)
        rej, padj, _, _ = multipletests(pv, alpha=ALFA, method="holm")
        for (i, j), r, p in zip(pares, rej, padj):
            mcn.setdefault(f"{i} × {j}", []).append(
                {"semente": int(s), "p_holm": p, "significativo": bool(r)})

        # bootstrap pareado da AUC-PR no teste: mesmo reamostra para os seis
        sc = {n: g[f"escore|{n}"].to_numpy() for n in nomes}
        amostras = {n: [] for n in nomes}
        n_te = len(y)
        for _ in range(BOOTSTRAP):
            k = rng.integers(0, n_te, n_te)
            for n in nomes:
                amostras[n].append(average_precision_score(y[k], sc[n][k]))
        amostras = {n: np.array(v) for n, v in amostras.items()}
        for n in nomes:
            lo, hi = np.percentile(amostras[n], [2.5, 97.5])
            boot.setdefault("ic95_auc_pr", {}).setdefault(n, []).append(
                [lo, hi])
        for i, j in pares:
            d = amostras[i] - amostras[j]
            lo, hi = np.percentile(d, [2.5, 97.5])
            boot.setdefault("dif_auc_pr", {}).setdefault(f"{i} × {j}", []).append(
                {"semente": int(s), "ic95": [lo, hi],
                 "separa": bool(lo > 0 or hi < 0)})

        # regra de decisão da §3.4: uma inversão só conta se as duas
        # diferenças passarem no teste (McNemar na acurácia, bootstrap na AUC-PR)
        lista = []
        for i, j in pares_invertidos(a_s, p_s):
            k = f"{i} × {j}"
            lista.append({"par": k,
                          "mcnemar": mcn[k][-1]["significativo"],
                          "bootstrap": boot["dif_auc_pr"][k][-1]["separa"]})
        por_semente[-1]["pares_invertidos"] = lista
        por_semente[-1]["invertidos_significativos"] = sum(
            p["mcnemar"] and p["bootstrap"] for p in lista)

    inv = pares_invertidos(acc, apr)
    resumo = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "sementes": SEMENTES, "teste": TESTE, "dobras": DOBRAS,
        "bootstrap": BOOTSTRAP, "alfa": ALFA,
        "n_treino": int(ex["n_treino"].iloc[0]),
        "n_teste": int(ex["n_teste"].iloc[0]),
        "pos_teste": int(ex["pos_teste"].iloc[0]),
        "piso": piso.to_dict(),
        "media": media.to_dict(orient="index"),
        "dp": dp.to_dict(orient="index"),
        "ordem_acuracia": ordem(acc), "ordem_auc_pr": ordem(apr),
        "ordem_f1": ordem(f1),
        "tau_acuracia_auc_pr": tau_media.statistic,
        "tau_p": tau_media.pvalue,
        "invertidos_medias": [list(p) for p in inv],
        "por_semente": por_semente,
        "mcnemar": mcn,
        "bootstrap_auc_pr": boot,
        "hiperparametros": mod.groupby("algoritmo")["hiperparametros"]
                              .agg(lambda v: v.value_counts().to_dict())
                              .to_dict(),
    }
    return resumo


def comparar(ex_a, res_a, ex_b, res_b):
    """A ordem do protocolo A contra a do B: a Questão de Pesquisa.

    Nas médias: τ de Kendall e pares invertidos, por AUC-PR (a métrica que
    define a ordem) e por acurácia. Por semente, o A da semente s contra o B da
    mesma semente. Regra da §3.4 (Capítulo 3): uma troca só conta se, nos dois
    protocolos, o par passar nos dois testes, McNemar com Holm e o intervalo
    de 95% da diferença de AUC-PR sem o zero."""
    nomes = list(ALGORITMOS)
    saida = {}
    for met in ("auc_pr", "acuracia"):
        ma = {n: res_a["media"][n][met] for n in nomes}
        mb = {n: res_b["media"][n][met] for n in nomes}
        tau = kendalltau([ma[n] for n in nomes], [mb[n] for n in nomes])

        def passa(res, par, s):
            mc = next(r["significativo"] for r in res["mcnemar"][par]
                      if r["semente"] == s)
            bs = next(r["separa"] for r in
                      res["bootstrap_auc_pr"]["dif_auc_pr"][par]
                      if r["semente"] == s)
            return bool(mc and bs)

        por_semente = []
        for s in SEMENTES:
            va = (ex_a[ex_a["semente"] == s].set_index("algoritmo")[met]
                  .to_dict())
            vb = (ex_b[ex_b["semente"] == s].set_index("algoritmo")[met]
                  .to_dict())
            va = {n: va[n] for n in nomes}
            vb = {n: vb[n] for n in nomes}
            lista = []
            for i, j in pares_invertidos(va, vb):
                par = f"{i} × {j}"
                lista.append({"par": par, "teste_a": passa(res_a, par, s),
                              "teste_b": passa(res_b, par, s)})
            por_semente.append({
                "semente": s,
                "tau": kendalltau([va[n] for n in nomes],
                                  [vb[n] for n in nomes]).statistic,
                "pares_invertidos": lista,
                "invertidos_significativos": sum(
                    p["teste_a"] and p["teste_b"] for p in lista)})
        saida[met] = {
            "teste": "McNemar com Holm e bootstrap da AUC-PR, nos dois",
            "ordem_a": ordem(ma), "ordem_b": ordem(mb),
            "tau_medias": tau.statistic, "tau_p": tau.pvalue,
            "invertidos_medias": [list(p) for p in pares_invertidos(ma, mb)],
            "por_semente": por_semente,
        }
    return saida


def ambiente():
    import imblearn, lightgbm, scipy, sklearn, statsmodels, xgboost
    return {
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "processador": platform.processor(),
        "nucleos_logicos": os.cpu_count(),
        "pacotes": {m.__name__: m.__version__ for m in (
            np, pd, sklearn, imblearn, xgboost, lightgbm, scipy, statsmodels)},
    }


def utc(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    args = sys.argv[1:]
    log = "--sem-log" not in args
    temporal = "--temporal" in args
    if temporal and not log:
        sys.exit("--temporal roda só com o log com sinal, como o protocolo A.")
    erc20_vazio = "--erc20-vazio" in args
    raiz = SAIDA / "erc20-vazio" if erc20_vazio else SAIDA
    saida = (raiz / "temporal" if temporal
             else raiz if log else raiz / "sem-log")
    X, y, atributos, enderecos = base_de_trabalho(not erc20_vazio)
    print(f"base: {len(y)} contas, {y.sum()} ilícitas, {len(atributos)} "
          f"atributos; log com sinal: {log}; temporal: {temporal}", flush=True)
    if temporal:
        ts = instantes(enderecos)
        if np.isnan(ts).any():
            sys.exit(f"{int(np.isnan(ts).sum())} contas da base sem instante em "
                     f"{DATAS.name}; conclua a coleta antes do protocolo B.")
    saida.mkdir(parents=True, exist_ok=True)
    if "--so-analise" in args:
        # refaz só resumo.json a partir dos CSVs já gravados
        ex = pd.read_csv(saida / "execucoes.csv")
        pr = pd.read_csv(saida / "predicoes.csv")
        anterior = json.loads((saida / "resumo.json").read_text(encoding="utf-8"))
        tempo = anterior["tempo_total_s"]
    else:
        t0 = time.perf_counter()
        ex, pr = executar(X, y, log, (ts, enderecos) if temporal else None)
        ex.insert(1, "log_com_sinal", log)
        ex.to_csv(saida / "execucoes.csv", index=False)
        pr.to_csv(saida / "predicoes.csv", index=False)
        (saida / "ambiente.json").write_text(
            json.dumps(ambiente(), ensure_ascii=False, indent=1),
            encoding="utf-8")
        tempo = time.perf_counter() - t0
    resumo = analisar(ex, pr)
    resumo["log_com_sinal"] = log
    resumo["erc20_zero"] = not erc20_vazio
    resumo["base"] = {"n": int(len(y)), "pos": int(y.sum()),
                      "atributos": len(atributos)}
    resumo["tempo_total_s"] = tempo
    if temporal:
        itr, ite, dobra = divisao_temporal(ts, enderecos, y)

        def parte(ix):
            classes = {}
            for c, nome in ((0, "licitas"), (1, "ilicitas")):
                t = ts[ix][y[ix] == c]
                classes[nome] = {"n": int(len(t)), "de": utc(t.min()),
                                 "ate": utc(t.max())}
            return {"n": int(len(ix)), "taxa": float(y[ix].mean()), **classes}
        resumo["periodos"] = {"treino_ajuste": parte(itr[dobra == -1]),
                              "validacao": parte(itr[dobra == 0]),
                              "treino": parte(itr), "teste": parte(ite)}
        resumo["data_sozinha_auc_roc"] = float(roc_auc_score(y, ts))
        res_a = json.loads((raiz / "resumo.json").read_text(encoding="utf-8"))
        resumo["comparacao_a_b"] = comparar(
            pd.read_csv(raiz / "execucoes.csv"), res_a, ex, resumo)
    (saida / "resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    print(json.dumps({k: resumo[k] for k in (
        "ordem_acuracia", "ordem_auc_pr", "tau_acuracia_auc_pr",
        "invertidos_medias", "tempo_total_s")}, ensure_ascii=False, indent=1))
    if temporal:
        c = resumo["comparacao_a_b"]
        print(json.dumps({"periodos": resumo["periodos"], **{
            m: {k: c[m][k] for k in ("ordem_a", "ordem_b", "tau_medias",
                                     "invertidos_medias")}
            for m in c}}, ensure_ascii=False, indent=1))
