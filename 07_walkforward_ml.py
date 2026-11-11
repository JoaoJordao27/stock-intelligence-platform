"""
Stock Intelligence Platform - Machine Learning: Validação Walk-Forward
=========================================================
Repete o experimento de 06_modelo_ml.py com múltiplos splits temporais
em janela expansiva, em vez de um único corte treino/teste. Isso
responde à pergunta que o resultado único de 06 deixava em aberto: o
ROC-AUC fraco foi azar de uma janela de teste específica, ou é
consistente ao longo do tempo?

Janela expansiva: a cada fold, o treino cresce (sempre desde o início
da série) e o teste avança para o próximo bloco de tempo — nunca há
dado futuro vazando para o treino.

Uso:
    python 07_walkforward_ml.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

INPUT_PATH = "output/fato_preco_com_indicadores.csv"
OUTPUT_PATH = "output/walkforward_resultados.csv"

HORIZONTE_DIAS = 20
CDI_ANUAL = 0.1376
CDI_20D = (1 + CDI_ANUAL) ** (HORIZONTE_DIAS / 252) - 1
N_FOLDS = 5

FEATURES = [
    "preco_vs_mm20", "preco_vs_mm50", "preco_vs_mm200",
    "rsi14", "retorno_diario", "volume_z",
]


def construir_features_e_alvo(fp: pd.DataFrame) -> pd.DataFrame:
    partes = []
    for ticker, g in fp.groupby("ticker"):
        g = g.sort_values("data").copy()
        g["preco_vs_mm20"] = g["fechamento"] / g["mm20"] - 1
        g["preco_vs_mm50"] = g["fechamento"] / g["mm50"] - 1
        g["preco_vs_mm200"] = g["fechamento"] / g["mm200"] - 1
        g["volume_z"] = (g["volume"] - g["volume"].rolling(60).mean()) / g["volume"].rolling(60).std()
        g["retorno_futuro_20d"] = g["fechamento"].shift(-HORIZONTE_DIAS) / g["fechamento"] - 1
        g["alvo"] = (g["retorno_futuro_20d"] > CDI_20D).astype(int)
        partes.append(g)
    return pd.concat(partes)


def gerar_folds(datas_unicas: list, n_folds: int):
    """Janela expansiva: divide as datas em (n_folds + 1) blocos.
    O bloco 0 é sempre treino; cada bloco seguinte vira teste de um
    fold, com o treino acumulando tudo que veio antes dele.
    """
    blocos = np.array_split(datas_unicas, n_folds + 1)
    for i in range(1, n_folds + 1):
        treino_datas = np.concatenate(blocos[:i])
        teste_datas = blocos[i]
        yield treino_datas, teste_datas


def treinar_e_avaliar(X_treino, y_treino, X_teste, y_teste):
    scaler = StandardScaler()
    X_treino_esc = scaler.fit_transform(X_treino)
    X_teste_esc = scaler.transform(X_teste)

    modelos = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42),
    }

    linhas = []
    for nome, modelo in modelos.items():
        if nome == "Regressão Logística":
            modelo.fit(X_treino_esc, y_treino)
            proba = modelo.predict_proba(X_teste_esc)[:, 1]
            pred = modelo.predict(X_teste_esc)
        else:
            modelo.fit(X_treino, y_treino)
            proba = modelo.predict_proba(X_teste)[:, 1]
            pred = modelo.predict(X_teste)

        linhas.append({
            "modelo": nome,
            "acuracia": accuracy_score(y_teste, pred),
            "f1": f1_score(y_teste, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_teste, proba) if y_teste.nunique() > 1 else np.nan,
        })
    return linhas


def main():
    fp = pd.read_csv(INPUT_PATH, parse_dates=["data"])
    df = construir_features_e_alvo(fp).dropna(subset=FEATURES + ["alvo"])

    datas_unicas = sorted(df["data"].unique())
    resultados = []

    for fold_i, (treino_datas, teste_datas) in enumerate(gerar_folds(datas_unicas, N_FOLDS), start=1):
        treino = df[df["data"].isin(treino_datas)]
        teste = df[df["data"].isin(teste_datas)]

        if teste["alvo"].nunique() < 2 or len(treino) < 100:
            print(f"Fold {fold_i}: pulado (dados insuficientes ou alvo sem variação)")
            continue

        linhas = treinar_e_avaliar(
            treino[FEATURES], treino["alvo"], teste[FEATURES], teste["alvo"]
        )
        for linha in linhas:
            linha["fold"] = fold_i
            linha["periodo_teste"] = f"{pd.Timestamp(teste_datas.min()).date()} a {pd.Timestamp(teste_datas.max()).date()}"
            linha["n_treino"] = len(treino)
            linha["n_teste"] = len(teste)
        resultados.extend(linhas)

    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv(OUTPUT_PATH, index=False)

    print("--- Resultado por fold ---")
    print(resultados_df[["fold", "periodo_teste", "modelo", "acuracia", "f1", "roc_auc"]]
          .round(3).to_string(index=False))

    print("\n--- Média e desvio-padrão do ROC-AUC por modelo (across folds) ---")
    resumo = resultados_df.groupby("modelo")["roc_auc"].agg(["mean", "std"]).round(3)
    print(resumo.to_string())

    print(f"\nSalvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
