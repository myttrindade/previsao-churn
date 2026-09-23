"""Treina e compara modelos de churn, escolhe o ponto de corte pelo resultado financeiro
e salva o modelo, as métricas e os gráficos.

Uso:
    python -m churn.treino
"""

from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import negocio
from churn.dados import ALVO, CATEGORICAS, FEATURES, NOMES, NUMERICAS, RAIZ, carregar, nome_legivel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEMENTE = 42
PASTA_MODELOS = RAIZ / "models"
PASTA_RELATORIOS = RAIZ / "reports"
PASTA_FIGURAS = PASTA_RELATORIOS / "figuras"

AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#898781"


def preprocessador(escalar: bool) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler() if escalar else "passthrough", NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ], verbose_feature_names_out=False)


def candidatos() -> dict[str, Pipeline]:
    return {
        "Regressão logística": Pipeline([
            ("prep", preprocessador(escalar=True)),
            ("modelo", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessador(escalar=False)),
            ("modelo", RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                              class_weight="balanced", random_state=SEMENTE, n_jobs=-1)),
        ]),
        "LightGBM": Pipeline([
            ("prep", preprocessador(escalar=False)),
            ("modelo", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                          min_child_samples=30, subsample=0.8, subsample_freq=1,
                                          colsample_bytree=0.8, reg_lambda=1.0,
                                          random_state=SEMENTE, verbose=-1)),
        ]),
    }


def estilo(ax, titulo: str):
    ax.set_title(titulo, loc="left", fontsize=12, fontweight="bold", color="#0b0b0b")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    for pasta in (PASTA_MODELOS, PASTA_FIGURAS):
        pasta.mkdir(parents=True, exist_ok=True)

    df = carregar()
    X, y = df[FEATURES], df[ALVO]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMENTE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)

    # 1. Comparação por validação cruzada (previsões fora da amostra no treino)
    comparacao, oof = {}, {}
    for nome, pipe in candidatos().items():
        prob = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        oof[nome] = prob
        comparacao[nome] = {"roc_auc": roc_auc_score(y_tr, prob),
                            "pr_auc": average_precision_score(y_tr, prob)}
        print(f"{nome:22s} ROC-AUC {comparacao[nome]['roc_auc']:.3f}  PR-AUC {comparacao[nome]['pr_auc']:.3f}")

    escolhido = max(comparacao, key=lambda n: comparacao[n]["pr_auc"])
    print(f"modelo escolhido: {escolhido}")

    # 2. Ponto de corte que maximiza o resultado da campanha, escolhido só com dados de treino
    cortes, valores = negocio.curva_resultado(y_tr.values, oof[escolhido], X_tr["MonthlyCharges"].values)
    corte = float(cortes[valores.argmax()])

    # 3. Treino final e avaliação no conjunto de teste (nunca visto)
    modelo = candidatos()[escolhido].fit(X_tr, y_tr)
    prob_te = modelo.predict_proba(X_te)[:, 1]
    pred_te = prob_te >= corte
    mens_te = X_te["MonthlyCharges"].values
    tn, fp, fn, tp = confusion_matrix(y_te, pred_te).ravel()

    estrategias = {
        "Não fazer nada": 0.0,
        "Abordar todos os clientes": negocio.resultado_campanha(y_te, np.ones(len(y_te), bool), mens_te),
        "Abordar só quem o modelo indica": negocio.resultado_campanha(y_te, pred_te, mens_te),
    }
    metricas = {
        "modelo": escolhido,
        "comparacao_validacao_cruzada": comparacao,
        "ponto_de_corte": corte,
        "teste": {
            "clientes": int(len(y_te)),
            "taxa_churn": float(y_te.mean()),
            "roc_auc": float(roc_auc_score(y_te, prob_te)),
            "pr_auc": float(average_precision_score(y_te, prob_te)),
            "precisao": float(precision_score(y_te, pred_te)),
            "recall": float(recall_score(y_te, pred_te)),
            "matriz_confusao": {"vp": int(tp), "fp": int(fp), "fn": int(fn), "vn": int(tn)},
            "clientes_abordados": int(pred_te.sum()),
        },
        "premissas_negocio": {
            "meses_gratis_oferta": negocio.MESES_GRATIS,
            "custo_operacional_usd": negocio.CUSTO_OPERACIONAL,
            "taxa_sucesso_oferta": negocio.TAXA_SUCESSO,
            "horizonte_meses": negocio.HORIZONTE_MESES,
        },
        "resultado_campanha_teste_usd": estrategias,
    }
    (PASTA_RELATORIOS / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metricas["teste"], indent=2))
    print(json.dumps(estrategias, ensure_ascii=False, indent=2))

    # 4. Modelo final treinado com todos os dados, para produção
    final = candidatos()[escolhido].fit(X, y)
    joblib.dump({
        "pipeline": final,
        "ponto_de_corte": corte,
        "features": FEATURES,
        "nomes": NOMES,
        "modelo": escolhido,
        "metricas_teste": metricas["teste"],
    }, PASTA_MODELOS / "modelo_churn.joblib")

    # 5. Gráficos
    fig, ax = plt.subplots(figsize=(7, 3.2))
    nomes = list(comparacao)
    xs = np.arange(len(nomes))
    ax.bar(xs - 0.18, [comparacao[n]["roc_auc"] for n in nomes], 0.34, color=AZUL, label="ROC-AUC")
    ax.bar(xs + 0.18, [comparacao[n]["pr_auc"] for n in nomes], 0.34, color=LARANJA, label="PR-AUC")
    for i, n in enumerate(nomes):
        ax.text(i - 0.18, comparacao[n]["roc_auc"] + 0.01, f"{comparacao[n]['roc_auc']:.3f}", ha="center", fontsize=8, color="#52514e")
        ax.text(i + 0.18, comparacao[n]["pr_auc"] + 0.01, f"{comparacao[n]['pr_auc']:.3f}", ha="center", fontsize=8, color="#52514e")
    ax.set_xticks(xs, nomes); ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    estilo(ax, "Comparação de modelos (validação cruzada, 5 dobras)")
    fig.tight_layout(); fig.savefig(PASTA_FIGURAS / "comparacao_modelos.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(cortes, valores / 1000, color=AZUL, linewidth=2)
    ax.axvline(corte, color=CINZA, linestyle="--", linewidth=1)
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.annotate(f"corte ótimo: {corte:.2f}", (corte, valores.max() / 1000 * 0.35), xytext=(8, 0),
                textcoords="offset points", fontsize=9, color="#52514e")
    ax.set_xlabel("Probabilidade mínima para abordar o cliente", fontsize=9, color="#52514e")
    ax.set_ylabel("Resultado (mil US$)", fontsize=9, color="#52514e")
    estilo(ax, "Resultado da campanha por ponto de corte (treino)")
    fig.tight_layout(); fig.savefig(PASTA_FIGURAS / "curva_resultado.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 2.6))
    rotulos = list(estrategias)
    vals = [estrategias[r] / 1000 for r in rotulos]
    cores = [CINZA, LARANJA, AZUL]
    ax.barh(rotulos, vals, color=cores, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + (1 if v >= 0 else -1), i, f"US$ {v:,.1f} mil".replace(",", "X").replace(".", ",").replace("X", "."),
                va="center", ha="left" if v >= 0 else "right", fontsize=9, color="#52514e")
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    ax.invert_yaxis(); ax.grid(False)
    estilo(ax, "Resultado da campanha (clientes de teste)")
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8); ax.grid(axis="y", visible=False)
    ax.set_xlim(min(0, min(vals)) * 1.35 - 2, max(vals) * 1.35)
    fig.tight_layout(); fig.savefig(PASTA_FIGURAS / "resultado_estrategias.png", dpi=160); plt.close(fig)

    salvar_shap(final, X)


def salvar_shap(pipeline: Pipeline, X: pd.DataFrame) -> None:
    import shap

    prep, modelo = pipeline.named_steps["prep"], pipeline.named_steps["modelo"]
    amostra = X.sample(1500, random_state=SEMENTE)
    Xt = pd.DataFrame(prep.transform(amostra), columns=[nome_legivel(c) for c in prep.get_feature_names_out()])
    if isinstance(modelo, LogisticRegression):
        valores = shap.LinearExplainer(modelo, Xt)(Xt)
    else:
        valores = shap.TreeExplainer(modelo)(Xt)
        if valores.values.ndim == 3:
            valores = valores[:, :, 1]
    plt.figure()
    shap.plots.beeswarm(valores, max_display=12, show=False)
    fig = plt.gcf()
    fig.axes[0].set_xlabel("Impacto na previsão (valor SHAP)")
    fig.canvas.draw()
    rotulos = [r.get_text().replace("Sum of", "Soma de").replace("other features", "outras variáveis")
               for r in fig.axes[0].get_yticklabels()]
    fig.axes[0].set_yticks(fig.axes[0].get_yticks(), rotulos)
    barra = fig.axes[-1]
    barra.set_ylabel("Valor da variável")
    barra.set_yticklabels(["Baixo", "Alto"])
    plt.title("O que mais influencia a previsão de churn (SHAP)", loc="left", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(PASTA_FIGURAS / "shap_importancia.png", dpi=160, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()
