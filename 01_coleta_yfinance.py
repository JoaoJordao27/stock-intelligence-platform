"""
Stock Intelligence Platform - Etapa 1: Coleta de Dados
=========================================================
Coleta preços históricos e dados cadastrais de empresas via Yahoo Finance
(yfinance) e monta as três tabelas do modelo de dados definido em
schema.sql: fato_preco, dim_empresa e dim_tempo (Dim_Tempo é derivada
diretamente de fato_preco, sem chamada extra à API).

Saída: 3 arquivos CSV em ./output/, prontos para carga no banco SQL
(Etapa 3), com as colunas exatamente no formato do schema.sql.

Uso:
    python 01_coleta_yfinance.py
"""

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------

# Carteira de exemplo: grandes empresas de setores distintos na B3,
# para permitir comparação setorial (uma das análises do case).
TICKERS = [
    "PETR4.SA",  # Petrobras - Energia
    "VALE3.SA",  # Vale - Materiais Básicos
    "ITUB4.SA",  # Itaú Unibanco - Financeiro
    "WEGE3.SA",  # WEG - Bens de Capital
    "ABEV3.SA",  # Ambev - Consumo
    "MGLU3.SA",  # Magazine Luiza - Varejo
]

PERIODO = "2y"          # janela histórica de preços
INTERVALO = "1d"        # granularidade diária
PAUSA_ENTRE_REQUISICOES = 1.0  # segundos, para evitar rate limiting

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("coleta_yfinance")


# ---------------------------------------------------------------------------
# COLETA: FATO_PRECO
# ---------------------------------------------------------------------------

def coletar_precos(tickers: list[str], periodo: str, intervalo: str) -> pd.DataFrame:
    """Baixa OHLCV por ticker e calcula o retorno diário.

    Retorna um DataFrame já no formato de fato_preco:
    data, ticker, abertura, maxima, minima, fechamento, volume, retorno_diario
    """
    linhas = []

    for ticker in tickers:
        log.info(f"Baixando preços de {ticker}...")
        try:
            hist = yf.Ticker(ticker).history(period=periodo, interval=intervalo)
        except Exception as e:
            log.warning(f"Falha ao baixar {ticker}: {e}")
            continue

        if hist.empty:
            log.warning(f"Sem dados retornados para {ticker}. Pulando.")
            continue

        hist = hist.sort_index()
        hist["retorno_diario"] = hist["Close"].pct_change()

        for data, row in hist.iterrows():
            linhas.append(
                {
                    "data": data.date(),
                    "ticker": ticker,
                    "abertura": round(row["Open"], 4),
                    "maxima": round(row["High"], 4),
                    "minima": round(row["Low"], 4),
                    "fechamento": round(row["Close"], 4),
                    "volume": int(row["Volume"]),
                    "retorno_diario": (
                        round(row["retorno_diario"], 6)
                        if pd.notna(row["retorno_diario"])
                        else None
                    ),
                }
            )

        time.sleep(PAUSA_ENTRE_REQUISICOES)

    df = pd.DataFrame(linhas)
    log.info(f"fato_preco: {len(df)} linhas coletadas para {df['ticker'].nunique()} tickers.")
    return df


# ---------------------------------------------------------------------------
# COLETA: DIM_EMPRESA
# ---------------------------------------------------------------------------

def coletar_empresas(tickers: list[str]) -> pd.DataFrame:
    """Consulta metadados cadastrais de cada empresa (setor, país, market cap).

    yfinance nem sempre preenche todos os campos (varia por ticker/mercado),
    por isso cada campo tem um fallback explícito em vez de quebrar o pipeline.
    """
    linhas = []

    for ticker in tickers:
        log.info(f"Baixando cadastro de {ticker}...")
        try:
            info = yf.Ticker(ticker).info
        except Exception as e:
            log.warning(f"Falha ao baixar info de {ticker}: {e}")
            info = {}

        linhas.append(
            {
                "ticker": ticker,
                "empresa": info.get("longName", info.get("shortName", ticker)),
                "setor": info.get("sector", "Não informado"),
                "pais": info.get("country", "Não informado"),
                "market_cap": info.get("marketCap", None),
            }
        )
        time.sleep(PAUSA_ENTRE_REQUISICOES)

    df = pd.DataFrame(linhas)
    log.info(f"dim_empresa: {len(df)} empresas cadastradas.")
    return df


# ---------------------------------------------------------------------------
# DERIVAÇÃO: DIM_TEMPO
# ---------------------------------------------------------------------------

def gerar_dim_tempo(fato_preco: pd.DataFrame) -> pd.DataFrame:
    """Gera a dimensão de tempo a partir das datas únicas já coletadas
    em fato_preco — evita qualquer chamada extra à API.
    """
    datas = pd.to_datetime(fato_preco["data"].unique())
    df = pd.DataFrame({"data": sorted(datas)})
    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    df["trimestre"] = df["data"].dt.quarter
    df["data"] = df["data"].dt.date

    log.info(f"dim_tempo: {len(df)} datas únicas.")
    return df


# ---------------------------------------------------------------------------
# EXECUÇÃO
# ---------------------------------------------------------------------------

def main():
    log.info(f"Iniciando coleta para {len(TICKERS)} tickers | período={PERIODO}")

    fato_preco = coletar_precos(TICKERS, PERIODO, INTERVALO)
    dim_empresa = coletar_empresas(TICKERS)
    dim_tempo = gerar_dim_tempo(fato_preco)

    fato_preco.to_csv(OUTPUT_DIR / "fato_preco.csv", index=False)
    dim_empresa.to_csv(OUTPUT_DIR / "dim_empresa.csv", index=False)
    dim_tempo.to_csv(OUTPUT_DIR / "dim_tempo.csv", index=False)

    log.info("Coleta concluída. Arquivos salvos em ./output/:")
    log.info(" - fato_preco.csv")
    log.info(" - dim_empresa.csv")
    log.info(" - dim_tempo.csv")

    # Checagem rápida de qualidade dos dados
    log.info("--- Resumo de qualidade ---")
    log.info(f"Linhas nulas em fato_preco: {fato_preco.isna().sum().sum()}")
    log.info(f"Tickers sem setor identificado: "
             f"{(dim_empresa['setor'] == 'Não informado').sum()}")
    log.info(f"Intervalo de datas: {fato_preco['data'].min()} a {fato_preco['data'].max()}")


if __name__ == "__main__":
    main()
