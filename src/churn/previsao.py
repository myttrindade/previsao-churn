"""Previsão para um cliente, com a explicação dos fatores que mais pesaram."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churn.dados import CATEGORICAS, NOMES, NUMERICAS, VALORES, limpar

ARQUIVO_MODELO = Path(__file__).resolve().parents[2] / "models" / "modelo_churn.joblib"


class Previsor:
    def __init__(self, caminho: Path = ARQUIVO_MODELO):
        artefato = joblib.load(caminho)
        self.pipeline = artefato["pipeline"]
        self.corte = artefato["ponto_de_corte"]
        self.features = artefato["features"]
        self.nome_modelo = artefato["modelo"]
        self.metricas = artefato["metricas_teste"]
        self.prep = self.pipeline.named_steps["prep"]
        self.modelo = self.pipeline.named_steps["modelo"]
        colunas = self.prep.get_feature_names_out()
        # Para cada coluna transformada, qual variável original ela representa.
        self.origem = np.array([self._variavel_original(c) for c in colunas])

    @staticmethod
    def _variavel_original(coluna: str) -> str:
        if coluna in NUMERICAS:
            return coluna
        return next(v for v in CATEGORICAS if coluna.startswith(v + "_"))

    def _contribuicoes(self, Xt: np.ndarray) -> np.ndarray:
        """Contribuição de cada coluna transformada para o log-odds do churn."""
        if hasattr(self.modelo, "booster_"):  # LightGBM: valores SHAP exatos nativos
            return self.modelo.predict(Xt, pred_contrib=True)[:, :-1]
        return Xt * self.modelo.coef_[0]      # modelos lineares

    def prever(self, cliente: dict) -> dict:
        df = limpar(pd.DataFrame([cliente]))[self.features]
        prob = float(self.pipeline.predict_proba(df)[0, 1])
        Xt = self.prep.transform(df)
        contrib = self._contribuicoes(Xt)[0]

        por_variavel = pd.Series(contrib).groupby(self.origem).sum().sort_values(key=np.abs, ascending=False)
        fatores = []
        for variavel, impacto in por_variavel.head(3).items():
            valor = df.iloc[0][variavel]
            if variavel in NUMERICAS:
                texto_valor = (f"{valor:.0f}" if variavel == "tenure" else
                               "US$ " + f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            else:
                texto_valor = VALORES.get(valor, valor)
            fatores.append({
                "variavel": NOMES[variavel],
                "valor": texto_valor,
                "efeito": "aumenta o risco" if impacto > 0 else "reduz o risco",
                "impacto": round(float(impacto), 3),
            })

        if prob >= max(0.6, self.corte):
            risco = "alto"
        elif prob >= self.corte:
            risco = "médio"
        else:
            risco = "baixo"

        return {
            "probabilidade_churn": round(prob, 4),
            "risco": risco,
            "abordar_cliente": prob >= self.corte,
            "ponto_de_corte": self.corte,
            "principais_fatores": fatores,
        }
