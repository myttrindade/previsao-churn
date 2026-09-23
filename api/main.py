"""API de previsão de churn.

Uso local:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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
