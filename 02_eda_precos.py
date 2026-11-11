# -*- coding: utf-8 -*-
"""
Stock Intelligence Platform - Etapa 2: Análise Exploratória
=========================================================
Explora os dados coletados na Etapa 1 (output/fato_preco.csv,
output/dim_empresa.csv, output/dim_tempo.csv): checagem de consistência,
evolução de preços, distribuição de retornos, risco x retorno por ticker e
correlação entre os retornos diários da carteira.

Gera 4 gráficos em ./plots/:
  - preco_normalizado.png       evolução de preço (rebase = 100 no início)
  - distribuicao_retornos.png   boxplot dos retornos diários por ticker
  - risco_x_retorno.png         volatilidade x retorno acumulado (scatter)
  - correlacao_retornos.png     heatmap de correlação entre os retornos diários

Uso:
    python 02_eda_precos.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")
PLOTS_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Paleta (guia interno de dataviz — 6 séries: slots categóricos 1-6, ordem
# fixa validada para pares adjacentes em gráficos de linha/barra)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

PALETA_TICKERS = {
    "PETR4.SA": "#2a78d6",  # slot 1 azul
    "VALE3.SA": "#eb6834",  # slot 2 laranja
    "ITUB4.SA": "#1baf7a",  # slot 3 aqua
    "WEGE3.SA": "#eda100",  # slot 4 amarelo
    "ABEV3.SA": "#e87ba4",  # slot 5 magenta
    "MGLU3.SA": "#008300",  # slot 6 verde
}
BLUE = "#2a78d6"

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


def main():
    # ==================================================================
    # Carrega dados
    # ==================================================================
    fp = pd.read_csv(os.path.join(OUTPUT_DIR, "fato_preco.csv"), parse_dates=["data"])
    de = pd.read_csv(os.path.join(OUTPUT_DIR, "dim_empresa.csv"))
    fp = fp.merge(de[["ticker", "empresa", "setor"]], on="ticker", how="left")

    tickers = list(PALETA_TICKERS.keys())

    # ==================================================================
    # 1) Checagem de consistência
    # ==================================================================
    print("=== CONSISTÊNCIA ===")
    print(f"Linhas: {len(fp)}  |  Tickers: {fp['ticker'].nunique()}  |  "
          f"Período: {fp['data'].min().date()} a {fp['data'].max().date()}")
    print(f"Nulos por coluna:\n{fp.isna().sum().to_string()}")
    print(f"Duplicatas (data, ticker): {fp.duplicated(subset=['data', 'ticker']).sum()}")
    print("(os nulos em retorno_diario são esperados: 1 por ticker, no 1º dia de cada série)")

    # ==================================================================
    # 2) Evolução de preço normalizado (rebase = 100 no primeiro dia)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(11, 6))
    resumo_retorno_total = {}
    for ticker in tickers:
        serie = fp[fp["ticker"] == ticker].sort_values("data")
        base = serie["fechamento"].iloc[0]
        normalizado = serie["fechamento"] / base * 100
        resumo_retorno_total[ticker] = normalizado.iloc[-1] - 100
        ax.plot(serie["data"], normalizado, color=PALETA_TICKERS[ticker],
                 linewidth=1.8, label=ticker, zorder=3)

    ax.axhline(100, color=INK_MUTED, linewidth=1, linestyle="--", zorder=2)
    ax.set_xlabel("Data", color=INK_SECONDARY)
    ax.set_ylabel("Preço normalizado (base 100 no início do período)", color=INK_SECONDARY)
    ax.set_title("Evolução de preço — carteira B3 (2 anos, rebase = 100)",
                 color=INK_PRIMARY, fontsize=13, loc="left")
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK_MUTED)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.legend(title="Ticker", frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "preco_normalizado.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\n=== RETORNO TOTAL NO PERÍODO (rebase, %) ===")
    ranking_retorno = pd.Series(resumo_retorno_total).sort_values(ascending=False)
    print(ranking_retorno.round(1).to_string())

    # ==================================================================
    # 3) Distribuição dos retornos diários (boxplot por ticker)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    dados_box = [fp.loc[fp["ticker"] == t, "retorno_diario"].dropna() * 100 for t in tickers]
    bp = ax.boxplot(dados_box, tick_labels=tickers, patch_artist=True, widths=0.6,
                     medianprops=dict(color=INK_PRIMARY, linewidth=1.5),
                     flierprops=dict(markersize=3, markeredgecolor=INK_MUTED, alpha=0.5))
    for patch, ticker in zip(bp["boxes"], tickers):
        patch.set_facecolor(PALETA_TICKERS[ticker])
        patch.set_alpha(0.8)
        patch.set_edgecolor(PAGE)

    ax.axhline(0, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_ylabel("Retorno diário (%)", color=INK_SECONDARY)
    ax.set_title("Distribuição dos retornos diários por ticker", color=INK_PRIMARY, fontsize=13, loc="left")
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.tick_params(colors=INK_MUTED)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "distribuicao_retornos.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # 4) Risco x retorno (volatilidade x retorno acumulado)
    # ==================================================================
    stats = fp.groupby("ticker")["retorno_diario"].agg(["std", "mean"]).dropna()
    stats["volatilidade_anualizada_%"] = stats["std"] * np.sqrt(252) * 100
    stats["retorno_total_%"] = pd.Series(resumo_retorno_total)
    stats["setor"] = de.set_index("ticker")["setor"]

    print("\n=== RISCO x RETORNO ===")
    print(stats[["volatilidade_anualizada_%", "retorno_total_%", "setor"]].round(1).to_string())

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for ticker in tickers:
        row = stats.loc[ticker]
        ax.scatter(row["volatilidade_anualizada_%"], row["retorno_total_%"],
                    color=PALETA_TICKERS[ticker], s=140, edgecolor=PAGE, linewidth=1, zorder=3)
        ax.annotate(ticker, (row["volatilidade_anualizada_%"], row["retorno_total_%"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9, color=INK_PRIMARY)

    ax.axhline(0, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_xlabel("Volatilidade anualizada (%)", color=INK_SECONDARY)
    ax.set_ylabel("Retorno total no período (%)", color=INK_SECONDARY)
    ax.set_title("Risco x retorno — carteira B3 (2 anos)", color=INK_PRIMARY, fontsize=13, loc="left")
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK_MUTED)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "risco_x_retorno.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # 5) Correlação entre os retornos diários
    # ==================================================================
    pivot = fp.pivot(index="data", columns="ticker", values="retorno_diario")[tickers]
    corr = pivot.corr()

    diverging_cmap = LinearSegmentedColormap.from_list("diverging", ["#e34948", "#f0efec", BLUE])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.values, cmap=diverging_cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(tickers))); ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha="right", color=INK_SECONDARY, fontsize=9)
    ax.set_yticklabels(tickers, color=INK_SECONDARY, fontsize=9)
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            val = corr.values[i, j]
            cor_texto = "#ffffff" if abs(val) > 0.5 else INK_PRIMARY
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=cor_texto, fontsize=8.5)
    ax.set_title("Correlação entre retornos diários", color=INK_PRIMARY, fontsize=13, loc="left")
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=INK_SECONDARY)
    cbar.outline.set_edgecolor(BASELINE)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "correlacao_retornos.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\n=== PAR MAIS E MENOS CORRELACIONADO ===")
    corr_pares = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    print(f"Mais correlacionado: {corr_pares.idxmax()} = {corr_pares.max():.2f}")
    print(f"Menos correlacionado: {corr_pares.idxmin()} = {corr_pares.min():.2f}")

    print("\nGráficos salvos em:", PLOTS_DIR)
    for nome in ["preco_normalizado.png", "distribuicao_retornos.png", "risco_x_retorno.png", "correlacao_retornos.png"]:
        print(" -", os.path.join(PLOTS_DIR, nome))


if __name__ == "__main__":
    main()
