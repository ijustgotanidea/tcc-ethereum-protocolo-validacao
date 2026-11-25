# Validação temporal na identificação de contas ilícitas no Ethereum

Experimento do TCC de Caio Busarello Dutra, Engenharia de Software, UNIVILLE. O estudo compara a ordem de desempenho de seis algoritmos em dois protocolos: divisão aleatória estratificada (A) e divisão temporal dentro de cada classe (B). Os dois usam a mesma base de trabalho, com 9.288 contas, e 10 sementes. A AUC-PR da classe ilícita define a ordem.

## Dados de entrada

Os CSVs de entrada ficam em `raw/tcc/dados/`. Eles não são versionados.

1. Baixe a versão 1 do [Ethereum Fraud Detection Dataset, de Vagif Aliyev](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset). Extraia `transaction_dataset.csv` para `raw/tcc/dados/`. O Kaggle informa a licença *Database: Open Database, Contents: Database Contents*. SHA-256 do arquivo usado no estudo: `874e125be2ac6220a92c7dbacf6c6f1766c60e3966473406d6700090831f7d9a`.
2. Para rodar B, é preciso ter `raw/tcc/dados/primeira_transacao_etherscan.csv`. Se já tiver o CSV da coleta original, copie-o para essa pasta. Para gerar outro, defina `ETHERSCAN_API_KEY` com sua própria chave da [API V2 do Etherscan](https://docs.etherscan.io/) e rode `python tools/collect_first_tx.py`. O coletor consulta transações normais, internas e de tokens ERC-20 para cada endereço e guarda a data mais antiga. A coleta original (2026-11-14) levou 220 minutos e encontrou datas para todas as 9.288 contas da base de trabalho. Uma nova coleta gera outro arquivo, ainda que as datas coincidam. O SHA-256 do CSV original é `dc815e1c5650e6ec0e4b69bf1dd26341a59e1fcedbc5763c745ff388357b302f`.

O treino funciona sem acesso à rede quando os dois CSVs estão no lugar. A chave só é necessária na coleta.

## Como reproduzir

Use Python 3.14.3. As versões dos pacotes diretos estão em `requirements.txt`, e o ambiente das execuções originais está em `Caio/mvp/ambiente.json`.

```sh
python -m venv .venv
```

Ative o ambiente com `.venv\Scripts\Activate.ps1` no PowerShell ou `source .venv/bin/activate` no Bash em Linux ou macOS. Na raiz do projeto, rode:

```sh
python -m pip install -r requirements.txt
python tools/audit_ethereum_dataset.py raw/tcc/dados/transaction_dataset.csv --out auditoria.md
python tools/run_mvp_experiment.py
python tools/run_mvp_experiment.py --temporal
python tools/run_importancia_permutacao.py
python tools/build_figura_mvp.py
```

Rode A antes de B: B usa as saídas de A para comparar as ordens. A análise de importância refaz os modelos dos dois protocolos e confere a AUC-PR de cada execução antes de permutar os atributos. O treino e a análise sobrescrevem as saídas em `Caio/mvp/`. Para preservar os resultados da entrega, rode em outra cópia do projeto.

A limpeza padrão preenche com zero os valores vazios de 16 atributos ERC-20. No arquivo de origem, esses vazios aparecem só em contas rotuladas como ilícitas. `--erc20-vazio` nos scripts de treino e de importância reproduz o tratamento anterior, em que os vazios passam pela imputação por mediana no pipeline. Essas saídas vão para `Caio/mvp/erc20-vazio/` e não substituem a rodada corrigida.

## Resultados do Capítulo 4

`Caio/mvp/` guarda o Protocolo A e `Caio/mvp/temporal/` guarda o Protocolo B. Cada pasta tem hiperparâmetros e métricas em `execucoes.csv`, predições de teste em `predicoes.csv` e os testes estatísticos em `resumo.json`. A importância por permutação dos dois protocolos está em `Caio/mvp/importancia_permutacao.csv` e `Caio/mvp/importancia_permutacao.json`. A figura do capítulo está em `Caio/mvp/figura_auc_pr_execucoes.png`.

Na rodada corrigida, a AUC-PR cai de 0,178 a 0,247 entre A e B. Só LightGBM e XGBoost trocam de posição pelas médias, e a troca não atende ao critério estatístico do estudo. Em B, o teste contém contas mais novas do que o treino dentro de cada classe. As duas classes foram coletadas em períodos diferentes, e os atributos de cada conta resumem o histórico completo. Por isso, B não simula uma previsão em um único instante de corte.
