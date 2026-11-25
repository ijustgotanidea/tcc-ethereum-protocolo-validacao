# -*- coding: utf-8 -*-
"""Etapa 2 -- auditoria de integridade do Ethereum Fraud Detection Dataset.

Answers, in order, the questions Etapa 2 exists to settle:

  1. shape and exact column names
  2. class balance and the majority-class baseline
  3. duplicated addresses and fully duplicated rows
  4. TEMPORAL AVAILABILITY -- is there an absolute date anywhere? (the
     project's highest-risk assumption)
  5. missing values
  6. constant / zero-variance columns
  7. columns that leak the label
  8. address format validity

Usage:
    python audit_dataset.py <path-to-csv> [--out report.md]

Only pandas and numpy are required.
"""
import argparse
import re
import sys

import numpy as np
import pandas as pd

ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

# names that would carry an absolute instant rather than a duration
DATE_HINTS = ("date", "time", "timestamp", "block", "first", "last", "seen",
              "created", "epoch")
# these are known DURATIONS in this dataset, not instants
DURATION_MARKERS = ("mins", "minute", "seconds", "diff", "avg min")


def find_col(df, *needles):
    for c in df.columns:
        low = c.strip().lower()
        if all(n in low for n in needles):
            return c
    return None


def audit(path):
    lines = []

    def say(s=""):
        lines.append(s)
        print(s)

    df = pd.read_csv(path)
    say("# Relatorio de integridade -- Ethereum Fraud Detection Dataset")
    say()
    say("Arquivo: `%s`" % path)
    say()

    # ---- 1. shape and columns -----------------------------------------
    say("## 1. Forma e colunas")
    say()
    say("- linhas: **%d**" % len(df))
    say("- colunas: **%d**" % df.shape[1])
    say()
    say("Colunas, na ordem do arquivo:")
    say()
    for i, c in enumerate(df.columns, 1):
        say("%2d. `%s` (%s)" % (i, c, df[c].dtype))
    say()

    # ---- 2. class balance ---------------------------------------------
    say("## 2. Balanceamento das classes")
    say()
    flag = find_col(df, "flag")
    if flag is None:
        say("**Coluna de rotulo nao encontrada.** Nenhuma coluna com 'flag' no nome.")
    else:
        vc = df[flag].value_counts(dropna=False).sort_index()
        total = len(df)
        for k, v in vc.items():
            say("- `%s` = %s: **%d** (%.2f%%)" % (flag, k, v, 100.0 * v / total))
        maj = vc.max()
        say()
        say("Linha de base da classe majoritaria: **%.2f%%** "
            "(um classificador que responde sempre a classe majoritaria)."
            % (100.0 * maj / total))
        say()
        say("Comparar com o valor usado no Capitulo 1: 9.841 contas, "
            "2.179 ilicitas (22,1%), linha de base 77,9%.")
    say()

    # ---- 3. duplicates -------------------------------------------------
    say("## 3. Duplicatas")
    say()
    full_dups = int(df.duplicated().sum())
    say("- linhas integralmente duplicadas: **%d**" % full_dups)

    addr = find_col(df, "address")
    if addr is None:
        say("- **coluna de endereco nao encontrada** (nenhuma com 'address' no nome)")
    else:
        dup_addr = int(df[addr].duplicated().sum())
        n_uniq = int(df[addr].nunique(dropna=False))
        say("- coluna de endereco: `%s`" % addr)
        say("- enderecos distintos: **%d** de %d linhas" % (n_uniq, len(df)))
        say("- linhas com endereco repetido: **%d**" % dup_addr)
        if dup_addr:
            g = df.groupby(addr).size().sort_values(ascending=False)
            g = g[g > 1]
            say("- enderecos que aparecem mais de uma vez: **%d**" % len(g))
            say()
            say("  Dez maiores repeticoes:")
            say()
            for a, n in g.head(10).items():
                extra = ""
                if flag is not None:
                    labs = sorted(df.loc[df[addr] == a, flag].unique().tolist())
                    extra = " | rotulos: %s%s" % (
                        labs, "  <-- ROTULO CONFLITANTE" if len(labs) > 1 else "")
                say("  - `%s` x%d%s" % (a, n, extra))
            say()
            if flag is not None:
                conflict = (df.groupby(addr)[flag].nunique() > 1).sum()
                say("- enderecos repetidos com rotulos divergentes: **%d**"
                    % int(conflict))
                say()
                say("  Cada duplicata de endereco que sobreviva ao "
                    "pre-processamento pode cair em treino e em teste ao mesmo "
                    "tempo. Isso e vazamento independente do protocolo, e "
                    "precisa ser resolvido antes de qualquer divisao.")
    say()

    # ---- 4. temporal availability (the decisive question) --------------
    say("## 4. Disponibilidade temporal")
    say()
    say("Esta e a pergunta que decide a viabilidade da divisao temporal.")
    say()

    cands = []
    for c in df.columns:
        low = c.strip().lower()
        if any(h in low for h in DATE_HINTS):
            is_duration = any(m in low for m in DURATION_MARKERS)
            cands.append((c, is_duration))

    if not cands:
        say("**Nenhuma coluna com nome sugestivo de tempo.**")
    else:
        say("Colunas com nome sugestivo de tempo:")
        say()
        for c, is_dur in cands:
            s = df[c]
            kind = "DURACAO (nome contem mins/diff/avg)" if is_dur else "a verificar"
            say("- `%s` | dtype %s | %s" % (c, s.dtype, kind))
            if pd.api.types.is_numeric_dtype(s):
                say("    min %s | max %s | mediana %s"
                    % (s.min(), s.max(), s.median()))
                # a unix timestamp in seconds sits roughly in this window
                unix_like = s.dropna().between(1.4e9, 1.9e9).mean() if len(s.dropna()) else 0
                if unix_like > 0.9:
                    say("    **>90% dos valores caem na faixa de epoch Unix "
                        "(2014-2030). Candidato real a instante absoluto.**")
                blocky = s.dropna().between(1e6, 3e7).mean() if len(s.dropna()) else 0
                if blocky > 0.9 and "block" in c.lower():
                    say("    **>90% na faixa de numero de bloco Ethereum. "
                        "Convertivel em data via Etherscan.**")
            else:
                say("    exemplos: %s" % s.dropna().head(3).tolist())
                parsed = pd.to_datetime(s, errors="coerce")
                ok = parsed.notna().mean()
                say("    parseavel como data: %.1f%%" % (100 * ok))
                if ok > 0.9:
                    say("    **Data absoluta legivel. min %s | max %s**"
                        % (parsed.min(), parsed.max()))
    say()

    absolute = []
    for c in df.columns:
        low = c.strip().lower()
        if any(m in low for m in DURATION_MARKERS):
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and len(s.dropna()):
            if s.dropna().between(1.4e9, 1.9e9).mean() > 0.9:
                absolute.append((c, "epoch unix"))
        else:
            if pd.to_datetime(s, errors="coerce").notna().mean() > 0.9:
                absolute.append((c, "data legivel"))

    say("### Veredito")
    say()
    if absolute:
        say("**Existe instante absoluto.** Colunas utilizaveis:")
        for c, why in absolute:
            say("- `%s` (%s)" % (c, why))
        say()
        say("A divisao temporal e executavel como planejado. Etapa 3 fixa o "
            "instante de previsao sobre essa coluna.")
    else:
        say("**Nao ha instante absoluto no arquivo.** So existem duracoes e "
            "medias, que ordenam transacoes dentro de uma conta mas nao "
            "ordenam contas entre si.")
        say()
        say("Consequencia: a divisao temporal, como escrita na Questao de "
            "Pesquisa, nao roda direto sobre este arquivo. Os dois caminhos "
            "previstos na secao 1.5 passam a valer:")
        say()
        say("1. recuperar o instante da primeira transacao de cada endereco "
            "pela API publica do Etherscan (9.841 chamadas, cabe em uma tarde "
            "com a chave gratuita e respeitando o limite de 5 req/s);")
        say("2. trocar para Elliptic++ (ELMOUGY; LIU, 2023), que ja traz 49 "
            "passos de tempo.")
        say()
        say("Levar as duas opcoes ao orientador na Etapa 3.")
    say()

    # ---- 5. missing values --------------------------------------------
    say("## 5. Valores ausentes")
    say()
    na = df.isna().sum()
    na = na[na > 0].sort_values(ascending=False)
    if not len(na):
        say("Nenhum valor ausente.")
    else:
        say("| Coluna | Ausentes | % |")
        say("| --- | ---: | ---: |")
        for c, n in na.items():
            say("| `%s` | %d | %.2f%% |" % (c, n, 100.0 * n / len(df)))
    say()

    # ---- 6. constant columns ------------------------------------------
    say("## 6. Colunas constantes ou sem variancia")
    say()
    # NaN is not a second value: a column that is 0 wherever it is filled is
    # still constant, and counting NaN as distinct hid seven such columns.
    const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const:
        say("Descartaveis no pre-processamento (%d), ausentes ignorados:" % len(const))
        for c in const:
            vals = df[c].dropna()
            say("- `%s` (valor unico: %s; ausentes: %d)"
                % (c, vals.iloc[0] if len(vals) else None, int(df[c].isna().sum())))
    else:
        say("Nenhuma.")
    say()

    # ---- 7. label leakage ---------------------------------------------
    say("## 7. Correlacao com o rotulo")
    say()
    if flag is None:
        say("Sem rotulo, nada a checar.")
    else:
        num = df.select_dtypes(include=[np.number])
        if flag in num:
            corr = num.corr(numeric_only=True)[flag].drop(flag).abs()
            corr = corr.sort_values(ascending=False).dropna()
            say("Dez maiores correlacoes absolutas com `%s`:" % flag)
            say()
            for c, v in corr.head(10).items():
                mark = "  <-- suspeito de vazamento" if v > 0.95 else ""
                say("- `%s`: %.4f%s" % (c, v, mark))
            say()
            hi = corr[corr > 0.95]
            if len(hi):
                say("**%d coluna(s) acima de 0,95.** Verificar se e informacao "
                    "derivada do rotulo antes de treinar." % len(hi))
            else:
                say("Nenhuma coluna acima de 0,95. Sem sinal obvio de vazamento "
                    "por atributo.")
    say()

    # ---- 8. address format --------------------------------------------
    say("## 8. Formato dos enderecos")
    say()
    if addr is None:
        say("Sem coluna de endereco.")
    else:
        s = df[addr].astype(str).str.strip()
        bad = ~s.str.match(ADDR_RE)
        say("- fora do formato `0x` + 40 hex: **%d**" % int(bad.sum()))
        if bad.any():
            say("- exemplos: %s" % s[bad].head(5).tolist())
        ws = (df[addr].astype(str) != s).sum()
        say("- com espacos nas bordas: **%d**" % int(ws))
        mixed = s.str.contains("[A-F]", regex=True) & s.str.contains("[a-f]", regex=True)
        say("- em checksum misto (EIP-55): **%d** (normalizar para minusculas "
            "antes de deduplicar)" % int(mixed.sum()))
    say()

    say("---")
    say()
    say("Gerado por `audit_dataset.py`.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    text = audit(a.csv)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print("\n[escrito em %s]" % a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
