"""
Stock Intelligence Platform - Indicadores Técnicos
=========================================================
Calcula médias móveis (20/50/200 dias) e RSI (14 períodos, método de
Wilder) a partir de fato_preco.csv, e salva uma versão enriquecida da
tabela pronta para o dashboard Power BI (aba Performance/Risk Analysis).

Uso:
    python 02_indicadores_tecnicos.py
"""

import pandas as pd

INPUT_PATH = "output/fato_preco.csv"
OUTPUT_PATH = "output/fato_preco_com_indicadores.csv"

JANELAS_MM = [20, 50, 200]
PERIODO_RSI = 14


def calcular_rsi(fechamento: pd.Series, periodo: int = PERIODO_RSI) -> pd.Series:
    """RSI pelo método de Wilder (média móvel exponencial suavizada,
    alpha = 1/periodo), que é o padrão de mercado — mais estável que a
    média móvel simples usada em implementações simplificadas.
    """
    delta = fechamento.diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)

    media_ganho = ganho.ewm(alpha=1 / periodo, min_periods=periodo, adjust=False).mean()
    media_perda = perda.ewm(alpha=1 / periodo, min_periods=periodo, adjust=False).mean()

    rs = media_ganho / media_perda
    return 100 - (100 / (1 + rs))


def enriquecer_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("data").copy()
    for janela in JANELAS_MM:
        g[f"mm{janela}"] = g["fechamento"].rolling(janela).mean()
    g["rsi14"] = calcular_rsi(g["fechamento"])
    return g


def main():
    fp = pd.read_csv(INPUT_PATH, parse_dates=["data"])
    fp = fp.sort_values(["ticker", "data"])

    enriquecido = pd.concat(
        [enriquecer_ticker(g) for _, g in fp.groupby("ticker")]
    )
    enriquecido.to_csv(OUTPUT_PATH, index=False)

    print(f"Indicadores calculados para {enriquecido['ticker'].nunique()} tickers.")
    print(f"Salvo em {OUTPUT_PATH}")

    ultimo = (
        enriquecido.sort_values("data")
        .groupby("ticker")
        .tail(1)[["ticker", "data", "fechamento", "mm20", "mm50", "mm200", "rsi14"]]
    )
    print("\nSnapshot mais recente:")
    print(ultimo.to_string(index=False))


if __name__ == "__main__":
    main()
