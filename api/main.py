"""API de previsão de churn.

Uso local:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churn.dados import NOMES, VALORES  # noqa: E402
from churn.entrada import ErroDeEntrada, MapeamentoNecessario, ler_arquivo  # noqa: E402
from churn.previsao import Previsor  # noqa: E402

SimNao = Literal["Yes", "No"]
SemInternet = Literal["Yes", "No", "No internet service"]


class Cliente(BaseModel):
    gender: Literal["Male", "Female"] = Field(description="Gênero")
    SeniorCitizen: Literal[0, 1] = Field(description="1 se o cliente tem 65 anos ou mais")
    Partner: SimNao = Field(description="Tem cônjuge")
    Dependents: SimNao = Field(description="Tem dependentes")
    tenure: int = Field(ge=0, le=120, description="Meses como cliente")
    PhoneService: SimNao
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: SemInternet
    OnlineBackup: SemInternet
    DeviceProtection: SemInternet
    TechSupport: SemInternet
    StreamingTV: SemInternet
    StreamingMovies: SemInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: SimNao
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(gt=0, le=500, description="Mensalidade (US$)")
    TotalCharges: float = Field(ge=0, description="Total já pago (US$)")

    model_config = {"json_schema_extra": {"examples": [{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5, "TotalCharges": 280.0,
    }]}}


class Fator(BaseModel):
    variavel: str
    valor: str
    efeito: str
    impacto: float


class Resposta(BaseModel):
    probabilidade_churn: float
    risco: Literal["baixo", "médio", "alto"]
    abordar_cliente: bool
    ponto_de_corte: float
    principais_fatores: list[Fator]


previsor = Previsor()
app = FastAPI(
    title="API de Previsão de Churn",
    description="Estima a chance de um cliente cancelar o serviço e explica os fatores que mais pesaram.",
    version="1.0.0",
)
PAGINA = Path(__file__).resolve().parent / "static" / "index.html"
BASE_EXEMPLO = Path(__file__).resolve().parent / "exemplos" / "base_exemplo_clientes.csv"
TAMANHO_MAXIMO = 10 * 1024 * 1024  # 10 MB


@app.get("/", include_in_schema=False)
def pagina():
    return FileResponse(PAGINA)


@app.get("/saude")
def saude():
    return {"status": "ok", "modelo": previsor.nome_modelo, "ponto_de_corte": previsor.corte,
            "metricas_teste": previsor.metricas}


@app.post("/prever", response_model=Resposta)
def prever(cliente: Cliente):
    return previsor.prever(cliente.model_dump())


class ClienteAnalisado(BaseModel):
    id: str
    probabilidade_churn: float
    risco: Literal["baixo", "médio", "alto"]
    abordar_cliente: bool
    mensalidade: float
    meses_como_cliente: int
    contrato: str
    motivos_risco: list[str]
    fator_protecao: str
    dados: dict[str, str | int | float]


class Resumo(BaseModel):
    clientes: int
    abordar: int
    risco_alto: int
    risco_medio: int
    risco_baixo: int
    receita_mensal_abordados: float
    receita_mensal_em_risco: float
    ganho_esperado_campanha: float
    ponto_de_corte: float


class AnaliseBase(BaseModel):
    resumo: Resumo
    clientes: list[ClienteAnalisado]


def _analisar(df: pd.DataFrame, mapeamento: dict[str, str] | None = None) -> dict:
    try:
        return previsor.analisar_base(df, mapeamento)
    except MapeamentoNecessario as pendente:
        raise HTTPException(status_code=422, detail=pendente.detalhe())
    except ErroDeEntrada as erro:
        raise HTTPException(status_code=422, detail=str(erro))


@app.post("/analisar-base", response_model=AnaliseBase)
async def analisar_base(
    arquivo: UploadFile = File(description="Planilha Excel (.xlsx) ou CSV com um cliente por linha"),
    mapeamento: str | None = Form(None, description='JSON opcional {"campo do modelo": "coluna da planilha"}'),
):
    """Analisa uma base inteira: devolve os clientes ordenados do maior para o menor risco.

    Reconhece colunas e valores em português ou inglês. Se alguma coluna obrigatória não for
    reconhecida, responde 422 com `tipo: mapeamento` e a lista de campos para o usuário indicar.
    """
    conteudo = await arquivo.read()
    if len(conteudo) > TAMANHO_MAXIMO:
        raise HTTPException(status_code=413, detail="Arquivo maior que 10 MB.")
    try:
        mapa = json.loads(mapeamento) if mapeamento else None
        df = ler_arquivo(conteudo, arquivo.filename)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Mapeamento de colunas inválido.")
    except ErroDeEntrada as erro:
        raise HTTPException(status_code=422, detail=str(erro))
    return _analisar(df, mapa)


@app.get("/analisar-base/exemplo", response_model=AnaliseBase)
def analisar_base_exemplo():
    """Analisa a base de exemplo com 500 clientes fictícios."""
    return _analisar(pd.read_csv(BASE_EXEMPLO))


def _em_portugues(df: pd.DataFrame) -> pd.DataFrame:
    """Converte a base do formato original para colunas e valores em português."""
    pt = df.copy()
    pt["SeniorCitizen"] = pt["SeniorCitizen"].map({0: "não", 1: "sim"})
    for c in pt.columns:
        if pt[c].dtype == object and c != "customerID":
            pt[c] = pt[c].map(lambda v: VALORES.get(v, v))
    return pt.rename(columns={"customerID": "ID do cliente", **NOMES})


def _excel(df: pd.DataFrame, nome: str) -> Response:
    saida = io.BytesIO()
    with pd.ExcelWriter(saida, engine="openpyxl") as escritor:
        df.to_excel(escritor, index=False, sheet_name="clientes")
        aba = escritor.sheets["clientes"]
        for coluna in aba.columns:
            aba.column_dimensions[coluna[0].column_letter].width = max(14, len(str(coluna[0].value)) + 4)
    return Response(saida.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f"attachment; filename={nome}"})


@app.get("/base-exemplo.csv", include_in_schema=False)
def baixar_base_exemplo():
    return FileResponse(BASE_EXEMPLO, media_type="text/csv", filename="base_exemplo_clientes.csv")


@app.get("/base-exemplo.xlsx", include_in_schema=False)
def baixar_base_exemplo_excel():
    return _excel(_em_portugues(pd.read_csv(BASE_EXEMPLO)), "base_exemplo_clientes.xlsx")


@app.get("/modelo-planilha.xlsx", include_in_schema=False)
def baixar_modelo_planilha_excel():
    return _excel(_em_portugues(pd.read_csv(BASE_EXEMPLO, nrows=3)), "modelo_planilha_clientes.xlsx")


@app.get("/modelo-planilha.csv", include_in_schema=False)
def baixar_modelo_planilha():
    modelo = pd.read_csv(BASE_EXEMPLO, nrows=3)
    return Response(modelo.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=modelo_planilha_clientes.csv"})
