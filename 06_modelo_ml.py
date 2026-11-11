"""
Stock Intelligence Platform - Machine Learning
=========================================================
Alvo: 1 se o retorno dos próximos 20 pregões supera o CDI equivalente
no mesmo período, 0 caso contrário. Classificação binária.

Features: indicadores técnicos já calculados (MM20/50/200, RSI) mais
retorno diário e volume — todos normalizados por ticker para não
misturar escalas de preço diferentes entre ações.

Modelos: Regressão Logística, Random Forest e XGBoost (os três
sugeridos no case), comparados por acurácia, F1 e ROC-AUC.

Split temporal (não aleatório): treina no passado, testa no futuro —
split aleatório aqui seria vazamento de dado, já que o alvo é
construído a partir de preços futuros.

Uso:
    python 06_modelo_ml.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

INPUT_PATH = "output/fato_preco_com_indicadores.csv"
OUTPUT_PATH = "output/previsoes_ml.csv"

HORIZONTE_DIAS = 20
CDI_ANUAL = 0.1376  # mesmo benchmark usado no Sharpe (03_sharpe_real.py)
CDI_20D = (1 + CDI_ANUAL) ** (HORIZONTE_DIAS / 252) - 1

FEATURES = [
    "preco_vs_mm20", "preco_vs_mm50", "preco_vs_mm200",
    "rsi14", "retorno_diario", "volume_z",
]


def construir_features_e_alvo(fp: pd.DataFrame) -> pd.DataFrame:
    """Gera as features relativas (independentes de escala de preço) e
    o alvo binário de retorno futuro vs. CDI, por ticker.
    """
    partes = []
    for ticker, g in fp.groupby("ticker"):
        g = g.sort_values("data").copy()

        g["preco_vs_mm20"] = g["fechamento"] / g["mm20"] - 1
        g["preco_vs_mm50"] = g["fechamento"] / g["mm50"] - 1
        g["preco_vs_mm200"] = g["fechamento"] / g["mm200"] - 1
        g["volume_z"] = (g["volume"] - g["volume"].rolling(60).mean()) / g["volume"].rolling(60).std()

        # Alvo: retorno realizado nos próximos HORIZONTE_DIAS pregões
        g["retorno_futuro_20d"] = g["fechamento"].shift(-HORIZONTE_DIAS) / g["fechamento"] - 1
        g["alvo"] = (g["retorno_futuro_20d"] > CDI_20D).astype(int)

        partes.append(g)

    df = pd.concat(partes)
    return df


def split_temporal(df: pd.DataFrame, frac_treino: float = 0.8):
    """Split cronológico: as datas mais antigas viram treino, as mais
    recentes viram teste. Evita vazamento de informação futura.
    """
    datas_ordenadas = sorted(df["data"].unique())
    corte = datas_ordenadas[int(len(datas_ordenadas) * frac_treino)]

    treino = df[df["data"] < corte]
    teste = df[df["data"] >= corte]
    return treino, teste, corte


def avaliar(nome: str, y_true, y_pred, y_proba) -> dict:
    return {
        "modelo": nome,
        "acuracia": round(accuracy_score(y_true, y_pred), 3),
        "f1": round(f1_score(y_true, y_pred), 3),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 3),
    }


def main():
    fp = pd.read_csv(INPUT_PATH, parse_dates=["data"])
    df = construir_features_e_alvo(fp)

    # remove linhas sem features completas (início da série, por causa
    # das janelas de MM200/volume_z) ou sem alvo (últimos 20 dias, sem
    # retorno futuro conhecido)
    df_modelo = df.dropna(subset=FEATURES + ["alvo"])

    print(f"Linhas utilizáveis: {len(df_modelo)} de {len(df)} totais")
    print(f"CDI equivalente em {HORIZONTE_DIAS} dias: {CDI_20D*100:.2f}%")
    print(f"Balanceamento do alvo: {df_modelo['alvo'].mean()*100:.1f}% de casos positivos\n")

    treino, teste, data_corte = split_temporal(df_modelo)
    print(f"Split temporal em {pd.Timestamp(data_corte).date()} — treino: {len(treino)} linhas, teste: {len(teste)} linhas\n")

    X_treino, y_treino = treino[FEATURES], treino["alvo"]
    X_teste, y_teste = teste[FEATURES], teste["alvo"]

    scaler = StandardScaler()
    X_treino_esc = scaler.fit_transform(X_treino)
    X_teste_esc = scaler.transform(X_teste)

    resultados = []

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_treino_esc, y_treino)
    proba = log_reg.predict_proba(X_teste_esc)[:, 1]
    resultados.append(avaliar("Regressão Logística", y_teste, log_reg.predict(X_teste_esc), proba))

    rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42)
    rf.fit(X_treino, y_treino)
    proba = rf.predict_proba(X_teste)[:, 1]
    resultados.append(avaliar("Random Forest", y_teste, rf.predict(X_teste), proba))

    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42)
    xgb.fit(X_treino, y_treino)
    proba = xgb.predict_proba(X_teste)[:, 1]
    resultados.append(avaliar("XGBoost", y_teste, xgb.predict(X_teste), proba))

    resultados_df = pd.DataFrame(resultados)
    print("--- Comparação de modelos (conjunto de teste) ---")
    print(resultados_df.to_string(index=False))
    resultados_df.to_csv("output/split_unico_resultados.csv", index=False)

    print("\n--- Importância das features (Random Forest) ---")
    importancias = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(importancias.round(3).to_string())

    # salva previsões do melhor modelo por ROC-AUC para inspeção
    melhor = resultados_df.loc[resultados_df["roc_auc"].idxmax(), "modelo"]
    print(f"\nMelhor modelo por ROC-AUC: {melhor}")

    teste_saida = teste[["data", "ticker", "retorno_futuro_20d", "alvo"]].copy()
    teste_saida["proba_xgb"] = xgb.predict_proba(X_teste)[:, 1]
    teste_saida.to_csv(OUTPUT_PATH, index=False)
    print(f"Previsões salvas em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
