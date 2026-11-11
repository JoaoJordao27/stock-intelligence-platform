"""
Stock Intelligence Platform - Sharpe Ratio (benchmark CDI/Selic)
=========================================================
Recalcula o Sharpe Ratio de cada ticker descontando o retorno livre de
risco (CDI/Selic), em vez do Sharpe simplificado (retorno/vol, sem
benchmark) usado na primeira rodada da EDA.

A taxa livre de risco é hardcoded como uma média anualizada do período
coletado, calculada a partir das taxas Selic mensais divulgadas pelo
Banco Central (fonte: bcb.gov.br). Para produção, troque RF_ANUAL por
uma consulta à série histórica 4390 da API do BC (CDI diário) e calcule
o excesso de retorno dia a dia em vez de usar uma média fixa.

Uso:
    python 03_sharpe_real.py
"""

import numpy as np
import pandas as pd

INPUT_PATH = "output/fato_preco.csv"
OUTPUT_PATH = "output/sharpe_real.csv"

# Selic/CDI médio anualizado do período coletado (10/09/2024-10/09/2026),
# calculado a partir das taxas mensais do Banco Central. Ajuste se o
# período de coleta mudar.
RF_ANUAL = 0.1376
DIAS_UTEIS_ANO = 252


def calcular_sharpe(fp: pd.DataFrame, rf_anual: float) -> pd.DataFrame:
    rf_diario = (1 + rf_anual) ** (1 / DIAS_UTEIS_ANO) - 1
    resultado = []

    for ticker, g in fp.groupby("ticker"):
        g = g.sort_values("data")
        ret = g["retorno_diario"].dropna()

        preco_ini = g["fechamento"].iloc[0]
        preco_fim = g["fechamento"].iloc[-1]
        retorno_acum = preco_fim / preco_ini - 1
        dias = len(ret)
        retorno_anualizado = (1 + retorno_acum) ** (DIAS_UTEIS_ANO / dias) - 1

        vol_anual = ret.std() * np.sqrt(DIAS_UTEIS_ANO)

        excesso_diario = ret - rf_diario
        sharpe = (
            (excesso_diario.mean() * DIAS_UTEIS_ANO) / vol_anual
            if vol_anual > 0
            else np.nan
        )

        resultado.append(
            {
                "ticker": ticker,
                "retorno_anualizado_%": round(retorno_anualizado * 100, 2),
                "vol_anualizada_%": round(vol_anual * 100, 2),
                "excesso_sobre_cdi_%": round((retorno_anualizado - rf_anual) * 100, 2),
                "sharpe_real": round(sharpe, 2),
            }
        )

    return pd.DataFrame(resultado).sort_values("sharpe_real", ascending=False)


def main():
    fp = pd.read_csv(INPUT_PATH, parse_dates=["data"])
    df = calcular_sharpe(fp, RF_ANUAL)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"CDI/Selic anualizado usado como benchmark: {RF_ANUAL * 100:.2f}%")
    print(f"Salvo em {OUTPUT_PATH}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
