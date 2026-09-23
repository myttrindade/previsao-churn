"""Premissas de negócio usadas para transformar probabilidade de churn em decisão.

Cenário: cada cliente abordado recebe uma oferta de retenção de 1 mês grátis.
A oferta custa a mensalidade de todos os abordados (inclusive de quem não ia cancelar)
e só gera receita quando convence alguém que realmente ia cancelar.
O ponto de corte do modelo é escolhido para maximizar esse resultado.
"""

from __future__ import annotations

import numpy as np

MESES_GRATIS = 1          # desconto oferecido, em meses de mensalidade
CUSTO_OPERACIONAL = 5.0   # US$ por cliente abordado (ligação, e-mail, atendimento)
TAXA_SUCESSO = 0.30       # fração dos clientes que iam cancelar e ficam após a oferta
HORIZONTE_MESES = 12      # receita preservada por cliente retido (meses de mensalidade)


def resultado_campanha(y_real, contatar, mensalidade) -> float:
    """Receita preservada menos o custo das ofertas e dos contatos (US$)."""
    y_real, contatar, mensalidade = np.asarray(y_real), np.asarray(contatar, bool), np.asarray(mensalidade)
    receita = (contatar & (y_real == 1)) * mensalidade * HORIZONTE_MESES * TAXA_SUCESSO
    custo = contatar * (mensalidade * MESES_GRATIS + CUSTO_OPERACIONAL)
    return float(receita.sum() - custo.sum())


def curva_resultado(y_real, prob, mensalidade, cortes=None):
    cortes = np.round(np.arange(0.05, 0.951, 0.01), 2) if cortes is None else cortes
    valores = [resultado_campanha(y_real, prob >= c, mensalidade) for c in cortes]
    return np.asarray(cortes), np.asarray(valores)
