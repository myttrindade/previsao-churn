from fastapi.testclient import TestClient

from api.main import app

cliente = TestClient(app)

RISCO = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5, "TotalCharges": 280.0,
}
FIEL = {**RISCO, "tenure": 60, "Contract": "Two year", "PaymentMethod": "Credit card (automatic)",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "TechSupport": "Yes",
        "MonthlyCharges": 65.2, "TotalCharges": 3900.0, "Partner": "Yes", "Dependents": "Yes"}


def test_saude():
    r = cliente.get("/saude")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_cliente_de_risco_deve_ser_abordado():
    r = cliente.post("/prever", json=RISCO).json()
    assert r["abordar_cliente"] is True
    assert r["risco"] in {"médio", "alto"}
    assert len(r["principais_fatores"]) == 3


def test_cliente_fiel_tem_risco_menor():
    risco = cliente.post("/prever", json=RISCO).json()["probabilidade_churn"]
    fiel = cliente.post("/prever", json=FIEL).json()
    assert fiel["probabilidade_churn"] < risco
    assert fiel["abordar_cliente"] is False


def test_valor_invalido_e_rejeitado():
    r = cliente.post("/prever", json={**RISCO, "Contract": "Three year"})
    assert r.status_code == 422


def test_pagina_inicial():
    r = cliente.get("/")
    assert r.status_code == 200
    assert "Previsão de Churn" in r.text
