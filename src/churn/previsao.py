"""Previsão de churn para um cliente ou para uma base inteira, com os fatores que mais pesaram."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churn import negocio
from churn.dados import CATEGORICAS, NOMES, NUMERICAS, VALORES, limpar
from churn.entrada import ErroDeEntrada, MapeamentoNecessario, padronizar  # noqa: F401

ARQUIVO_MODELO = Path(__file__).resolve().parents[2] / "models" / "modelo_churn.joblib"
LIMITE_LINHAS = 50_000
IMPACTO_MINIMO = 0.15  # contribuição mínima (log-odds) para um fator ser citado como motivo


def _formata_valor(variavel: str, valor) -> str:
    if variavel == "tenure":
        return f"{valor:.0f}"
    if variavel in NUMERICAS:
        return "US$ " + f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return VALORES.get(valor, valor)


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
        encoder = self.prep.named_transformers_["cat"]
        self.valores_aceitos = {v: set(map(str, cats)) for v, cats in zip(CATEGORICAS, encoder.categories_)}

    @staticmethod
    def _variavel_original(coluna: str) -> str:
        if coluna in NUMERICAS:
            return coluna
        return next(v for v in CATEGORICAS if coluna.startswith(v + "_"))

    def _contribuicoes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Contribuição de cada variável original para o log-odds do churn (uma linha por cliente)."""
        Xt = self.prep.transform(df)
        if hasattr(self.modelo, "booster_"):  # LightGBM: valores SHAP exatos nativos
            contrib = self.modelo.predict(Xt, pred_contrib=True)[:, :-1]
        else:                                  # modelos lineares
            contrib = Xt * self.modelo.coef_[0]
        return pd.DataFrame(contrib).T.groupby(self.origem).sum().T

    def _nivel(self, prob: np.ndarray) -> np.ndarray:
        return np.select([prob >= max(0.6, self.corte), prob >= self.corte], ["alto", "médio"], "baixo")

    # ---------- um cliente ----------
    def prever(self, cliente: dict) -> dict:
        df = limpar(pd.DataFrame([cliente]))[self.features]
        prob = float(self.pipeline.predict_proba(df)[0, 1])
        contrib = self._contribuicoes(df).iloc[0].sort_values(key=np.abs, ascending=False)
        fatores = [{
            "variavel": NOMES[v],
            "valor": _formata_valor(v, df.iloc[0][v]),
            "efeito": "aumenta o risco" if impacto > 0 else "reduz o risco",
            "impacto": round(float(impacto), 3),
        } for v, impacto in contrib.head(3).items()]
        return {
            "probabilidade_churn": round(prob, 4),
            "risco": str(self._nivel(np.array([prob]))[0]),
            "abordar_cliente": prob >= self.corte,
            "ponto_de_corte": self.corte,
            "principais_fatores": fatores,
        }

    # ---------- base inteira ----------
    def analisar_base(self, bruto: pd.DataFrame, mapeamento: dict[str, str] | None = None) -> dict:
        if len(bruto) > LIMITE_LINHAS:
            raise ErroDeEntrada(f"A planilha tem {len(bruto):,} linhas; o limite é {LIMITE_LINHAS:,}.")
        ids, nomes, df = padronizar(bruto, mapeamento)
        X = df[self.features]
        for c in CATEGORICAS:  # garantia extra: só valores que o modelo conhece
            invalidos = set(X[c].astype(str)) - self.valores_aceitos[c]
            if invalidos:
                raise ErroDeEntrada(f"Valores não reconhecidos em {NOMES[c]}: {', '.join(sorted(invalidos))}.")

        prob = self.pipeline.predict_proba(X)[:, 1]
        abordar = prob >= self.corte
        nivel = self._nivel(prob)
        contrib = self._contribuicoes(X)
        mens = X["MonthlyCharges"].to_numpy()
        # Valor esperado da oferta para cada cliente abordado, com as premissas de negócio do projeto.
        ganho = prob * mens * negocio.HORIZONTE_MESES * negocio.TAXA_SUCESSO \
            - (mens * negocio.MESES_GRATIS + negocio.CUSTO_OPERACIONAL)

        clientes = []
        for i in range(len(X)):
            c = contrib.iloc[i]
            alta = c[c > IMPACTO_MINIMO].sort_values(ascending=False).head(2)
            motivos = [f"{NOMES[v]}: {_formata_valor(v, X.iloc[i][v])}" for v in alta.index]
            protege = c.idxmin()
            clientes.append({
                "id": ids.iloc[i],
                "nome": nomes.iloc[i] or None,
                "probabilidade_churn": round(float(prob[i]), 4),
                "risco": str(nivel[i]),
                "abordar_cliente": bool(abordar[i]),
                "mensalidade": round(float(mens[i]), 2),
                "meses_como_cliente": int(X.iloc[i]["tenure"]),
                "contrato": VALORES.get(X.iloc[i]["Contract"], X.iloc[i]["Contract"]),
                "motivos_risco": motivos,
                "fator_protecao": f"{NOMES[protege]}: {_formata_valor(protege, X.iloc[i][protege])}",
                "dados": {k: (v.item() if hasattr(v, "item") else v) for k, v in X.iloc[i].items()},
            })
        clientes.sort(key=lambda c: c["probabilidade_churn"], reverse=True)

        return {
            "resumo": {
                "clientes": int(len(X)),
                "abordar": int(abordar.sum()),
                "risco_alto": int((nivel == "alto").sum()),
                "risco_medio": int((nivel == "médio").sum()),
                "risco_baixo": int((nivel == "baixo").sum()),
                "receita_mensal_abordados": round(float(mens[abordar].sum()), 2),
                "receita_mensal_em_risco": round(float((prob * mens).sum()), 2),
                "ganho_esperado_campanha": round(float(ganho[abordar].sum()), 2),
                "ponto_de_corte": self.corte,
            },
            "clientes": clientes,
        }
