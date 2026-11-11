# -*- coding: utf-8 -*-
"""
Stock Intelligence Platform - Comparação Sharpe simplificado x Sharpe real
=========================================================
Gráfico comparando, por ticker, o Sharpe "simplificado" da Etapa 2 (retorno
anualizado / volatilidade, sem descontar nada) com o Sharpe "real" da
03_sharpe_real.py (mesmo cálculo, mas descontando o CDI/Selic do período
antes de dividir pela volatilidade) — visualiza o quanto considerar um
benchmark de risco muda o ranking de quem "performou bem".

As duas métricas são recalculadas aqui a partir de fato_preco.csv (mesmas
fórmulas de 03_sharpe_real.py) em vez de importadas — módulos cujo nome
começa com dígito não são importáveis via `import` no Python, e duplicar 3
linhas de fórmula é mais simples/robusto que um import dinâmico por caminho.

Gera: plots/comparacao_sharpe.png

Uso:
    python 04_comparacao_sharpe.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

INPUT_PATH = "output/fato_preco.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Mesma taxa livre de risco e mesma janela de dias úteis de 03_sharpe_real.py
RF_ANUAL = 0.1376
DIAS_UTEIS_ANO = 252

# ---------------------------------------------------------------------------
# Paleta (mesma do 02_eda_precos.py)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"     # Sharpe simplificado (sem benchmark)
ORANGE = "#eb6834"   # Sharpe real (descontando CDI)

mpl.rcParams.update({
    "figure.facecolor": PAGE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": PAGE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK_PRIMARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def calcular_sharpes(fp: pd.DataFrame) -> pd.DataFrame:
    """Recalcula, por ticker, o Sharpe simplificado (sem benchmark) e o
    Sharpe real (descontando RF_ANUAL) — mesmas fórmulas de
    03_sharpe_real.py, só que devolvendo as duas versões lado a lado."""
    rf_diario = (1 + RF_ANUAL) ** (1 / DIAS_UTEIS_ANO) - 1
    linhas = []

    for ticker, g in fp.groupby("ticker"):
        g = g.sort_values("data")
        ret = g["retorno_diario"].dropna()

        preco_ini = g["fechamento"].iloc[0]
        preco_fim = g["fechamento"].iloc[-1]
        retorno_acum = preco_fim / preco_ini - 1
        dias = len(ret)
        retorno_anualizado = (1 + retorno_acum) ** (DIAS_UTEIS_ANO / dias) - 1

        vol_anual = ret.std() * np.sqrt(DIAS_UTEIS_ANO)

        sharpe_simplificado = retorno_anualizado / vol_anual if vol_anual > 0 else np.nan

        excesso_diario = ret - rf_diario
        sharpe_real = (
            (excesso_diario.mean() * DIAS_UTEIS_ANO) / vol_anual
            if vol_anual > 0
            else np.nan
        )

        linhas.append({
            "ticker": ticker,
            "sharpe_simplificado": round(sharpe_simplificado, 3),
            "sharpe_real": round(sharpe_real, 3),
        })

    return pd.DataFrame(linhas)


def main():
    fp = pd.read_csv(INPUT_PATH, parse_dates=["data"])
    df = calcular_sharpes(fp).sort_values("sharpe_real", ascending=False).reset_index(drop=True)

    print("=== SHARPE SIMPLIFICADO x SHARPE REAL (descontando CDI) ===")
    df["diferenca"] = df["sharpe_simplificado"] - df["sharpe_real"]
    print(df.to_string(index=False))

    mudou_de_sinal = df[np.sign(df["sharpe_simplificado"]) != np.sign(df["sharpe_real"])]
    if len(mudou_de_sinal):
        print(f"\nTickers que mudam de sinal (Sharpe positivo -> negativo) ao considerar o CDI: "
              f"{', '.join(mudou_de_sinal['ticker'])}")

    # ==================================================================
    # Gráfico: barras agrupadas, simplificado x real, por ticker
    # ==================================================================
    ordem = df.iloc[::-1]  # maior sharpe_real no topo
    y = np.arange(len(ordem))
    altura = 0.35

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(y + altura / 2, ordem["sharpe_simplificado"], height=altura,
            color=BLUE, edgecolor=PAGE, linewidth=0.5, label="Sharpe simplificado (sem benchmark)")
    ax.barh(y - altura / 2, ordem["sharpe_real"], height=altura,
            color=ORANGE, edgecolor=PAGE, linewidth=0.5, label="Sharpe real (descontando CDI)")

    ax.axvline(0, color=INK_PRIMARY, linewidth=1, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(ordem["ticker"])
    ax.set_xlabel("Sharpe Ratio", color=INK_SECONDARY)
    ax.set_title(f"Sharpe simplificado x Sharpe real (CDI = {RF_ANUAL*100:.2f}% a.a.)",
                 color=INK_PRIMARY, fontsize=13, loc="left")
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    ax.tick_params(colors=INK_MUTED)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    caminho = os.path.join(PLOTS_DIR, "comparacao_sharpe.png")
    fig.savefig(caminho, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nGráfico salvo em: {caminho}")


if __name__ == "__main__":
    main()
