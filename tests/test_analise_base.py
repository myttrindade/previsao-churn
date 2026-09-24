import io

import pandas as pd
from fastapi.testclient import TestClient

from api.main import BASE_EXEMPLO, app

cliente = TestClient(app)


def _envia(df: pd.DataFrame, sep: str = ","):
    arquivo = io.BytesIO(df.to_csv(index=False, sep=sep).encode("utf-8"))
    return cliente.post("/analisar-base", files={"arquivo": ("clientes.csv", arquivo, "text/csv")})


def test_base_exemplo_vem_ordenada_por_risco():
    r = cliente.get("/analisar-base/exemplo").json()
    probs = [c["probabilidade_churn"] for c in r["clientes"]]
    assert r["resumo"]["clientes"] == 500
    assert probs == sorted(probs, reverse=True)
    assert r["resumo"]["abordar"] == sum(c["abordar_cliente"] for c in r["clientes"])


def test_upload_mantem_o_id_do_cliente():
    df = pd.read_csv(BASE_EXEMPLO).head(20)
    r = _envia(df)
    assert r.status_code == 200
    assert {c["id"] for c in r.json()["clientes"]} == set(df["customerID"])


def test_aceita_ponto_e_virgula_e_idoso_como_texto():
    df = pd.read_csv(BASE_EXEMPLO).head(10)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    r = _envia(df, sep=";")
    assert r.status_code == 200
    assert r.json()["resumo"]["clientes"] == 10


def test_coluna_ausente_gera_mensagem_clara():
    df = pd.read_csv(BASE_EXEMPLO).head(5).drop(columns=["Contract"])
    r = _envia(df)
    assert r.status_code == 422
    assert "Contract" in r.json()["detail"]


def test_valor_desconhecido_gera_mensagem_clara():
    df = pd.read_csv(BASE_EXEMPLO).head(5)
    df.loc[0, "Contract"] = "Three year"
    r = _envia(df)
    assert r.status_code == 422
    assert "Three year" in r.json()["detail"]


def test_modelo_de_planilha_tem_as_colunas_obrigatorias():
    r = cliente.get("/modelo-planilha.csv")
    colunas = r.text.splitlines()[0].split(",")
    assert r.status_code == 200
    assert {"customerID", "Contract", "tenure", "MonthlyCharges"} <= set(colunas)
