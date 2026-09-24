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


def test_coluna_ausente_pede_correspondencia():
    df = pd.read_csv(BASE_EXEMPLO).head(5).drop(columns=["Contract"])
    r = _envia(df)
    assert r.status_code == 422
    detalhe = r.json()["detail"]
    assert detalhe["tipo"] == "mapeamento"
    assert [f["campo"] for f in detalhe["faltando"]] == ["Contract"]


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


def _excel_pt() -> bytes:
    return cliente.get("/base-exemplo.xlsx").content


def test_excel_em_portugues_e_reconhecido_automaticamente():
    r = cliente.post("/analisar-base", files={"arquivo": ("clientes.xlsx", io.BytesIO(_excel_pt()),
                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert r.status_code == 200
    corpo = r.json()
    exemplo = cliente.get("/analisar-base/exemplo").json()
    assert corpo["resumo"] == exemplo["resumo"]  # mesmo resultado da base original em inglês
    assert corpo["clientes"][0]["id"].startswith("CLI-")


def test_coluna_com_outro_nome_usa_a_correspondencia_informada():
    df = pd.read_excel(io.BytesIO(_excel_pt())).head(30).rename(columns={"Tipo de contrato": "Modalidade"})
    arquivo = io.BytesIO(df.to_csv(index=False, sep=";").encode("utf-8"))
    r1 = cliente.post("/analisar-base", files={"arquivo": ("c.csv", arquivo, "text/csv")})
    assert r1.status_code == 422
    assert r1.json()["detail"]["faltando"][0]["campo"] == "Contract"
    assert "Modalidade" in r1.json()["detail"]["colunas_arquivo"]

    arquivo.seek(0)
    r2 = cliente.post("/analisar-base", files={"arquivo": ("c.csv", arquivo, "text/csv")},
                      data={"mapeamento": '{"Contract": "Modalidade"}'})
    assert r2.status_code == 200
    assert r2.json()["resumo"]["clientes"] == 30


def test_numeros_no_formato_brasileiro_e_total_opcional():
    df = pd.read_excel(io.BytesIO(_excel_pt())).head(10).drop(columns=["Total já pago"])
    df["Mensalidade"] = df["Mensalidade"].map(lambda v: f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    r = _envia(df, sep=";")
    assert r.status_code == 200
    assert r.json()["resumo"]["clientes"] == 10


def test_valor_desconhecido_em_portugues_gera_mensagem_clara():
    df = pd.read_excel(io.BytesIO(_excel_pt())).head(5)
    df.loc[0, "Tipo de contrato"] = "trimestral"
    r = _envia(df)
    assert r.status_code == 422
    assert "trimestral" in r.json()["detail"]


def test_cada_cliente_traz_os_dados_para_simulacao():
    c = cliente.get("/analisar-base/exemplo").json()["clientes"][0]
    simulado = cliente.post("/prever", json={**c["dados"], "SeniorCitizen": 1 if c["dados"]["SeniorCitizen"] == "Yes" else 0})
    assert simulado.status_code == 200
    assert abs(simulado.json()["probabilidade_churn"] - c["probabilidade_churn"]) < 1e-3


def test_nome_do_cliente_e_reconhecido_em_portugues_e_ingles():
    exemplo = cliente.get("/analisar-base/exemplo").json()["clientes"]
    assert all(c["nome"] for c in exemplo)
    df = pd.read_excel(io.BytesIO(_excel_pt())).head(5).rename(columns={"Nome do cliente": "Razão social"})
    r = _envia(df).json()["clientes"]
    assert {c["nome"] for c in r} == set(df["Razão social"])
