"""
Stock Intelligence Platform - Gerador do Dashboard HTML
=========================================================
Lê fato_preco.csv, dim_empresa.csv e sharpe_real.csv (e, se existirem,
dim_fundamentos.csv, fato_preco_com_indicadores.csv e os resultados de
ML), calcula os agregados necessários (série normalizada, variação
diária, correlação) e injeta tudo como JSON dentro de
dashboard_template.html, gerando o dashboard.html final — um único
arquivo, sem servidor, abre direto no navegador.

Rode de novo sempre que atualizar os dados (nova rodada da Etapa 1).

Uso:
    python 04_gerar_dashboard.py
"""

import json
from pathlib import Path

import pandas as pd

FATO_PRECO = "output/fato_preco.csv"
DIM_EMPRESA = "output/dim_empresa.csv"
SHARPE = "output/sharpe_real.csv"
FUNDAMENTOS = "output/dim_fundamentos.csv"
INDICADORES = "output/fato_preco_com_indicadores.csv"
SPLIT_UNICO = "output/split_unico_resultados.csv"
WALKFORWARD = "output/walkforward_resultados.csv"
TEMPLATE = "dashboard_template.html"
OUTPUT = "output/dashboard.html"

CAMPOS_INDICADORES = ["mm20", "mm50", "mm200", "rsi14"]

CAMPOS_FUNDAMENTOS = [
    "pl", "pvp", "roe", "roa", "margem_liquida", "margem_operacional",
    "divida_patrimonio", "liquidez_corrente", "dividend_yield", "crescimento_receita",
]


def montar_dados() -> dict:
    fp = pd.read_csv(FATO_PRECO, parse_dates=["data"])
    de = pd.read_csv(DIM_EMPRESA)
    sharpe = pd.read_csv(SHARPE)
    fund = pd.read_csv(FUNDAMENTOS) if Path(FUNDAMENTOS).exists() else None
    ind = pd.read_csv(INDICADORES, parse_dates=["data"]) if Path(INDICADORES).exists() else None

    datas = sorted(fp["data"].unique())
    datas_str = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in datas]

    series = {}
    tickers_info = []

    for ticker, g in fp.groupby("ticker"):
        g = g.set_index("data").reindex(datas)
        close = g["fechamento"].tolist()
        base = close[0]
        norm = [round(c / base * 100, 2) if pd.notna(c) else None for c in close]
        close_clean = [round(c, 2) if pd.notna(c) else None for c in close]
        series[ticker] = {"close": close_clean, "norm100": norm}

        if ind is not None:
            g_ind = ind[ind["ticker"] == ticker].set_index("data").reindex(datas)
            for campo in CAMPOS_INDICADORES:
                series[ticker][campo] = [
                    round(float(v), 2) if pd.notna(v) else None for v in g_ind[campo]
                ]

        row_de = de[de["ticker"] == ticker].iloc[0]
        validos = g.dropna(subset=["fechamento"])
        ultimo, penultimo = validos.iloc[-1], validos.iloc[-2]
        var_dia = (ultimo["fechamento"] / penultimo["fechamento"] - 1) * 100
        row_sharpe = sharpe[sharpe["ticker"] == ticker].iloc[0]

        info = {
            "ticker": ticker.replace(".SA", ""),
            "ticker_full": ticker,
            "empresa": row_de["empresa"],
            "setor": row_de["setor"],
            "market_cap": float(row_de["market_cap"]),
            "preco_atual": round(float(ultimo["fechamento"]), 2),
            "variacao_dia_pct": round(float(var_dia), 2),
            "retorno_anualizado_pct": float(row_sharpe["retorno_anualizado_%"]),
            "vol_anualizada_pct": float(row_sharpe["vol_anualizada_%"]),
            "sharpe_real": float(row_sharpe["sharpe_real"]),
            "excesso_cdi_pct": float(row_sharpe["excesso_sobre_cdi_%"]),
        }

        if fund is not None:
            row_fund = fund[fund["ticker"] == ticker]
            if not row_fund.empty:
                row_fund = row_fund.iloc[0]
                for campo in CAMPOS_FUNDAMENTOS:
                    val = row_fund.get(campo, None)
                    info[campo] = None if pd.isna(val) else round(float(val), 2)

        tickers_info.append(info)

    pivot = fp.pivot(index="data", columns="ticker", values="retorno_diario")
    corr = pivot.corr().round(2)

    ml = None
    if Path(SPLIT_UNICO).exists() and Path(WALKFORWARD).exists():
        split_unico = pd.read_csv(SPLIT_UNICO)
        walkforward = pd.read_csv(WALKFORWARD)
        resumo_wf = walkforward.groupby("modelo")["roc_auc"].agg(["mean", "std"]).round(3).reset_index()

        ml = {
            "split_unico": split_unico.to_dict(orient="records"),
            "walkforward_folds": walkforward[["fold", "periodo_teste", "modelo", "roc_auc"]].to_dict(orient="records"),
            "walkforward_resumo": resumo_wf.to_dict(orient="records"),
        }

    return {
        "meta": {
            "periodo_inicio": datas_str[0],
            "periodo_fim": datas_str[-1],
            "cdi_anual_pct": 13.76,  # ver 03_sharpe_real.py para a fonte
            "n_tickers": len(tickers_info),
            "tem_indicadores": ind is not None,
        },
        "datas": datas_str,
        "series": series,
        "tickers": tickers_info,
        "correlacao": {
            "tickers": [t.replace(".SA", "") for t in corr.columns],
            "matriz": corr.values.tolist(),
        },
        "ml": ml,
    }


def main():
    dados = montar_dados()
    template = Path(TEMPLATE).read_text(encoding="utf-8")
    html_final = template.replace("__DATA_JSON__", json.dumps(dados, ensure_ascii=False))

    Path(OUTPUT).write_text(html_final, encoding="utf-8")
    print(f"Dashboard gerado em {OUTPUT}")
    print(f"{dados['meta']['n_tickers']} tickers | {dados['meta']['periodo_inicio']} a {dados['meta']['periodo_fim']}")


if __name__ == "__main__":
    main()
