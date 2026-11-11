"""
Stock Intelligence Platform - Financial Health: Coleta de Fundamentos
=========================================================
Complementa dim_empresa com indicadores fundamentalistas via
yf.Ticker().info, para alimentar a página "Financial Health" do
dashboard (múltiplos de valuation, rentabilidade e endividamento).

Gera uma tabela nova, dim_fundamentos, em vez de alterar dim_empresa —
mantém o schema original do case intacto e isola dados que mudam com
frequência diferente (fundamentos trimestrais vs. preço diário).

Uso:
    python 05_coleta_fundamentos.py
"""

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "ABEV3.SA", "MGLU3.SA",
]

PAUSA_ENTRE_REQUISICOES = 1.0
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("coleta_fundamentos")

# Campos do yf.Ticker().info mapeados para nomes de coluna em português.
# Nem todo ticker preenche todos os campos (varia por mercado/setor);
# valores ausentes ficam como NaN em vez de quebrar a coleta.
CAMPOS = {
    "trailingPE": "pl",
    "priceToBook": "pvp",
    "returnOnEquity": "roe",
    "returnOnAssets": "roa",
    "profitMargins": "margem_liquida",
    "operatingMargins": "margem_operacional",
    "debtToEquity": "divida_patrimonio",
    "currentRatio": "liquidez_corrente",
    "dividendYield": "dividend_yield",
    "revenueGrowth": "crescimento_receita",
}


def coletar_fundamentos(tickers: list[str]) -> pd.DataFrame:
    linhas = []

    for ticker in tickers:
        log.info(f"Baixando fundamentos de {ticker}...")
        try:
            info = yf.Ticker(ticker).info
        except Exception as e:
            log.warning(f"Falha ao baixar fundamentos de {ticker}: {e}")
            info = {}

        linha = {"ticker": ticker}
        for campo_origem, campo_destino in CAMPOS.items():
            linha[campo_destino] = info.get(campo_origem, None)
        linhas.append(linha)

        time.sleep(PAUSA_ENTRE_REQUISICOES)

    df = pd.DataFrame(linhas)

    # ROE, margens e crescimento vêm como fração (0.15) — convertendo para %.
    # dividend_yield é EXCEÇÃO: na versão atual do yfinance (1.7.0), o campo
    # `dividendYield` já vem em pontos percentuais (8.96 = 8.96%), não como
    # fração (0.0896) — confirmado inspecionando o retorno bruto da API.
    # Multiplicar por 100 aqui infla o valor 100x (PETR4 apareceria como
    # "896% de dividend yield", o que é absurdo). Se o formato da API mudar
    # de novo no futuro, reconfirme com yf.Ticker(t).info["dividendYield"]
    # antes de mexer nesta lista.
    for col in ["roe", "roa", "margem_liquida", "margem_operacional", "crescimento_receita"]:
        df[col] = df[col] * 100

    return df


def main():
    df = coletar_fundamentos(TICKERS)
    df.to_csv(OUTPUT_DIR / "dim_fundamentos.csv", index=False)

    log.info(f"dim_fundamentos: {len(df)} tickers salvos em output/dim_fundamentos.csv")
    log.info("--- Preview ---")
    print(df.round(2).to_string(index=False))

    faltantes = df.isna().sum()
    if faltantes.sum() > 0:
        log.info("Campos com dados ausentes (comum em alguns tickers/setores):")
        print(faltantes[faltantes > 0].to_string())


if __name__ == "__main__":
    main()
