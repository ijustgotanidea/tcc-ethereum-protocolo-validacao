# -*- coding: utf-8 -*-
"""Coleta o instante da primeira transação de cada conta (§3.3 do TCC).

Para cada endereço bem formado e distinto de raw/tcc/dados/transaction_dataset.csv,
consulta a API V2 do Etherscan (ETHERSCAN, 2026) em três listas, todas com
sort=asc&page=1&offset=1, então o único registro devolvido é o mais antigo:

- txlist          transações normais
- txlistinternal  transações internas
- tokentx         transferências de tokens ERC-20

O instante da conta é o menor carimbo de tempo de bloco entre as três
(WOOD, 2025). Endereço sem nenhum registro nas três fica com o instante vazio e
status "sem_transacao".

A coleta é retomável: cada endereço concluído vira uma linha no CSV de saída na
hora, e uma nova execução pula os que já estão lá. Endereço que falhou depois das
tentativas fica com status "erro" e é refeito na próxima execução.

A chave vem da variável de ambiente ETHERSCAN_API_KEY e nunca é gravada.

    ETHERSCAN_API_KEY=... python tools/collect_first_tx.py [--rps 3] [--limite N]

Só biblioteca padrão.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRADA = ROOT / "raw/tcc/dados/transaction_dataset.csv"
SAIDA = ROOT / "raw/tcc/dados/primeira_transacao_etherscan.csv"
API = "https://api.etherscan.io/v2/api"
ACOES = ("txlist", "txlistinternal", "tokentx")
CAMPOS = ["address", "ts_txlist", "bloco_txlist", "ts_txlistinternal",
          "bloco_txlistinternal", "ts_tokentx", "bloco_tokentx",
          "primeira_ts", "primeira_utc", "status", "coletado_em"]
TENTATIVAS = 6
# Mensagens do Etherscan que pedem só uma nova tentativa, não um erro definitivo.
TRANSITORIO = re.compile(r"rate limit|max calls|timeout|busy|try again", re.I)


def enderecos():
    """Endereços distintos e bem formados, na ordem do arquivo."""
    vistos, lista = set(), []
    with ENTRADA.open(newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            a = linha["Address"].strip().lower()
            if re.fullmatch(r"0x[0-9a-f]{40}", a) and a not in vistos:
                vistos.add(a)
                lista.append(a)
    return lista


class Cliente:
    def __init__(self, chave, rps):
        self.chave, self.intervalo, self.ultimo = chave, 1.0 / rps, 0.0

    def primeiro(self, acao, endereco):
        """(timestamp, bloco) do registro mais antigo, ou (None, None)."""
        q = urllib.parse.urlencode({
            "chainid": 1, "module": "account", "action": acao,
            "address": endereco, "startblock": 0, "endblock": 99999999,
            "page": 1, "offset": 1, "sort": "asc", "apikey": self.chave})
        espera = 1.0
        for _ in range(TENTATIVAS):
            pausa = self.ultimo + self.intervalo - time.monotonic()
            if pausa > 0:
                time.sleep(pausa)
            self.ultimo = time.monotonic()
            try:
                with urllib.request.urlopen(f"{API}?{q}", timeout=30) as r:
                    d = json.load(r)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                time.sleep(espera)
                espera *= 2
                continue
            res = d.get("result")
            if d.get("status") == "1" and isinstance(res, list) and res:
                return int(res[0]["timeStamp"]), int(res[0]["blockNumber"])
            if isinstance(res, list) and not res:
                return None, None          # "No transactions found"
            texto = f"{d.get('message')} {res}"
            if "invalid api key" in texto.lower():
                sys.exit(f"Etherscan recusou a chave: {texto}")
            if not TRANSITORIO.search(texto):
                raise RuntimeError(f"{acao} {endereco}: {texto}")
            time.sleep(espera)
            espera *= 2
        raise RuntimeError(f"{acao} {endereco}: {TENTATIVAS} tentativas falharam")


def ja_coletados():
    if not SAIDA.exists():
        return set()
    with SAIDA.open(newline="", encoding="utf-8") as f:
        return {l["address"] for l in csv.DictReader(f) if l["status"] != "erro"}


def regravar_sem_erros():
    """Tira as linhas de erro antes de refazê-las, para não duplicar endereço."""
    if not SAIDA.exists():
        return
    with SAIDA.open(newline="", encoding="utf-8") as f:
        linhas = [l for l in csv.DictReader(f) if l["status"] != "erro"]
    with SAIDA.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CAMPOS)
        w.writeheader()
        w.writerows(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", type=float, default=3.0,
                    help="chamadas por segundo; o plano gratuito permite 3")
    ap.add_argument("--limite", type=int, default=None,
                    help="coleta só os N primeiros endereços pendentes")
    args = ap.parse_args()
    chave = os.environ.get("ETHERSCAN_API_KEY")
    if not chave:
        sys.exit("Defina ETHERSCAN_API_KEY.")

    regravar_sem_erros()
    feitos = ja_coletados()
    pendentes = [a for a in enderecos() if a not in feitos]
    if args.limite is not None:
        pendentes = pendentes[:args.limite]
    print(f"{len(feitos)} já coletados, {len(pendentes)} pendentes", flush=True)

    cli = Cliente(chave, args.rps)
    novo = not SAIDA.exists()
    t0 = time.monotonic()
    with SAIDA.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CAMPOS)
        if novo:
            w.writeheader()
        for i, a in enumerate(pendentes, 1):
            linha = {"address": a}
            try:
                for acao in ACOES:
                    ts, bl = cli.primeiro(acao, a)
                    linha[f"ts_{acao}"], linha[f"bloco_{acao}"] = ts, bl
                tss = [linha[f"ts_{x}"] for x in ACOES if linha[f"ts_{x}"]]
                if tss:
                    m = min(tss)
                    linha["primeira_ts"] = m
                    linha["primeira_utc"] = datetime.fromtimestamp(
                        m, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    linha["status"] = "ok"
                else:
                    linha["status"] = "sem_transacao"
            except RuntimeError as e:
                print(f"erro: {e}", flush=True)
                linha["status"] = "erro"
            linha["coletado_em"] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
            w.writerow(linha)
            f.flush()
            if i % 100 == 0 or i == len(pendentes):
                dt = time.monotonic() - t0
                print(f"{i}/{len(pendentes)}  {dt / 60:.1f} min  "
                      f"faltam ~{dt / i * (len(pendentes) - i) / 60:.0f} min",
                      flush=True)


if __name__ == "__main__":
    main()
