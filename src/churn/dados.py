"""Leitura e limpeza da base Telco Customer Churn (IBM)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO = RAIZ / "data" / "telco_churn.csv"

ALVO = "Churn"
NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERICAS + CATEGORICAS

# Nomes em português para relatórios, API e interface.
NOMES = {
    "tenure": "Meses como cliente",
    "MonthlyCharges": "Mensalidade",
    "TotalCharges": "Total já pago",
    "gender": "Gênero",
    "SeniorCitizen": "Idoso",
    "Partner": "Tem cônjuge",
    "Dependents": "Tem dependentes",
    "PhoneService": "Telefone",
    "MultipleLines": "Múltiplas linhas",
    "InternetService": "Tipo de internet",
    "OnlineSecurity": "Segurança online",
    "OnlineBackup": "Backup online",
    "DeviceProtection": "Proteção de aparelho",
    "TechSupport": "Suporte técnico",
    "StreamingTV": "Streaming de TV",
    "StreamingMovies": "Streaming de filmes",
    "Contract": "Tipo de contrato",
    "PaperlessBilling": "Fatura digital",
    "PaymentMethod": "Forma de pagamento",
}


VALORES = {
    "Yes": "sim", "No": "não", "Male": "masculino", "Female": "feminino",
    "No phone service": "sem telefone", "No internet service": "sem internet",
    "DSL": "DSL", "Fiber optic": "fibra óptica",
    "Month-to-month": "mensal", "One year": "1 ano", "Two year": "2 anos",
    "Electronic check": "boleto eletrônico", "Mailed check": "cheque pelo correio",
    "Bank transfer (automatic)": "débito automático", "Credit card (automatic)": "cartão de crédito",
}


def nome_legivel(coluna: str) -> str:
    """Converte uma coluna do modelo (ex.: 'Contract_Month-to-month') em texto em português."""
    if coluna in NOMES:
        return NOMES[coluna]
    for original in CATEGORICAS:
        if coluna.startswith(original + "_"):
            valor = coluna[len(original) + 1:]
            return f"{NOMES[original]}: {VALORES.get(valor, valor)}"
    return coluna


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges vem como texto e fica em branco para clientes no primeiro mês (tenure = 0).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    if ALVO in df:
        df[ALVO] = (df[ALVO] == "Yes").astype(int)
    return df.drop(columns=["customerID"], errors="ignore")


def carregar() -> pd.DataFrame:
    return limpar(pd.read_csv(ARQUIVO))
