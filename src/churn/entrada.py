"""Leitura flexível de planilhas de clientes.

Aceita CSV (vírgula ou ponto e vírgula) e Excel, reconhece nomes de colunas e
valores em português ou inglês e converte tudo para o formato usado no treino.
Quando uma coluna obrigatória não é reconhecida, devolve a lista para o usuário
indicar qual coluna da planilha corresponde a ela.
"""

from __future__ import annotations

import io
import re
import unicodedata

import pandas as pd

from churn.dados import CATEGORICAS, NOMES

OBRIGATORIAS = [
    "tenure", "MonthlyCharges", "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
OPCIONAIS = ["TotalCharges"]  # se faltar, é estimado como meses × mensalidade
ID = "__id__"


def normaliza(texto) -> str:
    """Minúsculas, sem acentos e sem pontuação: 'Tipo de Contrato' -> 'tipodecontrato'."""
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", texto.lower())


# Nomes de coluna aceitos para cada variável (além do próprio nome original).
SINONIMOS = {
    ID: ["customerid", "id", "iddocliente", "idcliente", "codigo", "codigodocliente", "codigocliente", "cliente",
         "clienteid", "codcliente", "matricula", "cpfcnpj", "cnpj", "cpf", "conta"],
    "tenure": ["mesescomocliente", "meses", "tempodecasa", "tempodecasameses", "tempocliente", "antiguidade", "mesesdecontrato"],
    "MonthlyCharges": ["mensalidade", "valormensal", "valormensalidade", "cobrancamensal", "fatura", "valorfatura", "mensalidadeus"],
    "TotalCharges": ["totaljapago", "totalpago", "valortotal", "totalcobrado", "totalfaturado"],
    "gender": ["genero", "sexo"],
    "SeniorCitizen": ["idoso", "65anosoumais", "maiorde65", "terceiraidade"],
    "Partner": ["temconjuge", "conjuge", "casado", "parceiro"],
    "Dependents": ["temdependentes", "dependentes", "filhos"],
    "PhoneService": ["telefone", "servicodetelefone", "temtelefone", "telefonefixo"],
    "MultipleLines": ["multiplaslinhas", "variaslinhas", "linhasmultiplas"],
    "InternetService": ["tipodeinternet", "internet", "servicodeinternet", "tipointernet"],
    "OnlineSecurity": ["segurancaonline", "seguranca"],
    "OnlineBackup": ["backuponline", "backup"],
    "DeviceProtection": ["protecaodeaparelho", "protecaodedispositivo", "protecaoaparelho", "seguroaparelho"],
    "TechSupport": ["suportetecnico", "suporte"],
    "StreamingTV": ["streamingdetv", "streamingtv", "tv"],
    "StreamingMovies": ["streamingdefilmes", "streamingfilmes", "filmes"],
    "Contract": ["tipodecontrato", "contrato", "plano", "tipodeplano", "fidelidade"],
    "PaperlessBilling": ["faturadigital", "faturaonline", "contadigital", "faturaeletronica"],
    "PaymentMethod": ["formadepagamento", "pagamento", "metododepagamento", "meiodepagamento"],
}

SIM = {"yes", "sim", "s", "y", "true", "verdadeiro", "1", "x"}
NAO = {"no", "nao", "n", "false", "falso", "0"}
SEM_INTERNET = {"nointernetservice", "seminternet", "naoteminternet", "semservicodeinternet"}
SEM_TELEFONE = {"nophoneservice", "semtelefone", "naotemtelefone", "semservicodetelefone"}

VALORES = {
    "gender": {"male": "Male", "masculino": "Male", "m": "Male", "homem": "Male",
               "female": "Female", "feminino": "Female", "f": "Female", "mulher": "Female"},
    "InternetService": {"dsl": "DSL", "adsl": "DSL", "fiberoptic": "Fiber optic", "fibraoptica": "Fiber optic",
                        "fibraotica": "Fiber optic", "fibra": "Fiber optic", "no": "No", "nao": "No",
                        **{v: "No" for v in SEM_INTERNET}},
    "Contract": {"monthtomonth": "Month-to-month", "mensal": "Month-to-month", "mesames": "Month-to-month",
                 "semfidelidade": "Month-to-month", "oneyear": "One year", "1ano": "One year", "umano": "One year",
                 "anual": "One year", "12meses": "One year", "twoyear": "Two year", "2anos": "Two year",
                 "doisanos": "Two year", "bienal": "Two year", "24meses": "Two year"},
    "PaymentMethod": {"electroniccheck": "Electronic check", "boleto": "Electronic check",
                      "boletoeletronico": "Electronic check", "pix": "Electronic check",
                      "mailedcheck": "Mailed check", "cheque": "Mailed check", "chequepelocorreio": "Mailed check",
                      "banktransferautomatic": "Bank transfer (automatic)", "debitoautomatico": "Bank transfer (automatic)",
                      "transferenciabancaria": "Bank transfer (automatic)", "debito": "Bank transfer (automatic)",
                      "creditcardautomatic": "Credit card (automatic)", "cartaodecredito": "Credit card (automatic)",
                      "cartao": "Credit card (automatic)", "creditcard": "Credit card (automatic)"},
}
SERVICOS_INTERNET = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


class ErroDeEntrada(ValueError):
    """Planilha fora do formato esperado, com mensagem para o usuário."""


class MapeamentoNecessario(ErroDeEntrada):
    """Faltam colunas obrigatórias que o usuário precisa indicar."""

    def __init__(self, faltando: list[str], colunas: list[str], reconhecidas: dict[str, str]):
        super().__init__("Algumas colunas não foram reconhecidas.")
        self.faltando, self.colunas, self.reconhecidas = faltando, colunas, reconhecidas

    def detalhe(self) -> dict:
        return {
            "tipo": "mapeamento",
            "mensagem": "Não reconheci algumas colunas. Indique qual coluna da sua planilha corresponde a cada informação.",
            "faltando": [{"campo": c, "nome": NOMES[c]} for c in self.faltando],
            "colunas_arquivo": self.colunas,
            "reconhecidas": {NOMES.get(k, "ID do cliente"): v for k, v in self.reconhecidas.items()},
        }


def ler_arquivo(conteudo: bytes, nome: str) -> pd.DataFrame:
    nome = (nome or "").lower()
    try:
        if nome.endswith((".xlsx", ".xlsm", ".xls")) or conteudo[:2] == b"PK":
            return pd.read_excel(io.BytesIO(conteudo), dtype=object)
        try:
            texto = conteudo.decode("utf-8-sig")
        except UnicodeDecodeError:
            texto = conteudo.decode("latin-1")
        return pd.read_csv(io.StringIO(texto), sep=None, engine="python", dtype=str)
    except Exception:
        raise ErroDeEntrada("Não foi possível ler o arquivo. Envie uma planilha Excel (.xlsx) ou CSV.")


def reconhecer_colunas(colunas: list[str], mapeamento: dict[str, str] | None = None) -> dict[str, str]:
    """Devolve {variável do modelo: coluna da planilha}."""
    por_nome = {normaliza(c): c for c in colunas}
    achadas = {}
    for campo in [ID] + OBRIGATORIAS + OPCIONAIS:
        for candidato in [campo] + SINONIMOS.get(campo, []):
            if normaliza(candidato) in por_nome:
                achadas[campo] = por_nome[normaliza(candidato)]
                break
    for campo, coluna in (mapeamento or {}).items():
        if coluna in colunas:
            achadas[campo] = coluna
    return achadas


def _numero(serie: pd.Series) -> pd.Series:
    """Converte '1.234,56', 'R$ 95,50', 'US$ 95.50' ou 95.5 em número."""
    def conv(v):
        if pd.isna(v) or isinstance(v, (int, float)):
            return v
        t = re.sub(r"[^0-9,.\-]", "", str(v))
        if "," in t and "." in t:
            t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
        elif "," in t:
            t = t.replace(",", ".")
        return t or None
    return pd.to_numeric(serie.map(conv), errors="coerce")


def _categoria(campo: str, valor) -> str | None:
    if pd.isna(valor) or str(valor).strip() == "":
        return None
    if isinstance(valor, (int, float)) and valor in (0, 1):  # Excel costuma trazer 0/1 como número
        valor = "Yes" if valor == 1 else "No"
    n = normaliza(valor)
    if campo in VALORES:
        return VALORES[campo].get(n)
    if campo in SERVICOS_INTERNET and n in SEM_INTERNET:
        return "No internet service"
    if campo == "MultipleLines" and n in SEM_TELEFONE:
        return "No phone service"
    if n in SIM:
        return "Yes"
    if n in NAO:
        return "No"
    return None


def padronizar(bruto: pd.DataFrame, mapeamento: dict[str, str] | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Converte a planilha do usuário no formato do modelo. Devolve (ids, dados)."""
    bruto = bruto.dropna(how="all")
    bruto.columns = [str(c).strip() for c in bruto.columns]
    if bruto.empty:
        raise ErroDeEntrada("A planilha está vazia.")
    achadas = reconhecer_colunas(list(bruto.columns), mapeamento)
    faltando = [c for c in OBRIGATORIAS if c not in achadas]
    if faltando:
        raise MapeamentoNecessario(faltando, list(bruto.columns), achadas)

    bruto = bruto.reset_index(drop=True)
    ids = (bruto[achadas[ID]].astype(str).str.strip() if ID in achadas
           else pd.Series([f"linha {i + 2}" for i in range(len(bruto))]))
    df = pd.DataFrame(index=bruto.index)
    for campo in ("tenure", "MonthlyCharges"):
        df[campo] = _numero(bruto[achadas[campo]])
        if df[campo].isna().any():
            linhas = (df.index[df[campo].isna()] + 2)[:5].tolist()
            raise ErroDeEntrada(f"Valores vazios ou não numéricos em '{achadas[campo]}' ({NOMES[campo]}), linhas {linhas}.")
    df["TotalCharges"] = (_numero(bruto[achadas["TotalCharges"]]) if "TotalCharges" in achadas
                          else df["tenure"] * df["MonthlyCharges"])
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])

    for campo in CATEGORICAS:
        original = bruto[achadas[campo]]
        convertido = original.map(lambda v, c=campo: _categoria(c, v))
        if campo == "SeniorCitizen":
            convertido = convertido.map({"Yes": "Yes", "No": "No"})
        if convertido.isna().any():
            ruins = sorted({str(v) for v in original[convertido.isna()]})[:5]
            raise ErroDeEntrada(f"Não reconheci os valores {', '.join(repr(r) for r in ruins)} na coluna "
                                f"'{achadas[campo]}' ({NOMES[campo]}).")
        df[campo] = convertido
    return ids, df
