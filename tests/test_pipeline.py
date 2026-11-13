"""Testes da geracao fundamentada (requisitos 3 e 4 do PDF).

Mesma filosofia do test_retriever.py: mocka LLM e indice pra testar so a logica de
orquestracao (validacao de citacao, threshold de recusa) - rapido e sem custo de API.
"""

from unittest.mock import Mock

from langchain_core.documents import Document

import pipeline
from pipeline import Evidencia, RespostaRAG, responder, validar_citacoes


def _doc(review_id: str, nota: int = 5, data: str = "2018-01-01") -> Document:
    return Document(
        page_content=f"review {review_id}",
        metadata={"review_id": review_id, "review_score": nota, "review_creation_date": data},
    )


def _resposta(evidencias: list[Evidencia]) -> RespostaRAG:
    return RespostaRAG(evidencia_suficiente=True, resposta="resposta qualquer", evidencias=evidencias)


def test_validar_citacoes_aceita_citacao_que_esta_no_contexto_recuperado():
    docs_e_scores = [(_doc("r1"), 0.9)]
    resposta = _resposta([Evidencia(review_id="r1", nota=5, data="2018-01-01", trecho="texto")])

    resultado = validar_citacoes(resposta, docs_e_scores)

    assert resultado is resposta


def test_validar_citacoes_rejeita_citacao_fora_do_contexto_recuperado():
    docs_e_scores = [(_doc("r1"), 0.9)]
    resposta = _resposta([Evidencia(review_id="r99", nota=5, data="2018-01-01", trecho="texto")])

    try:
        validar_citacoes(resposta, docs_e_scores)
        assert False, "deveria ter levantado ValueError"
    except ValueError as erro:
        assert "r99" in str(erro)


def test_responder_nao_chama_llm_de_geracao_quando_score_abaixo_do_threshold(monkeypatch):
    docs_com_score_baixo = [(_doc("r1"), 0.50)]
    monkeypatch.setattr(pipeline, "recuperar", Mock(return_value=docs_com_score_baixo))
    llm_geracao = Mock()

    resultado = responder("pergunta fora de escopo", indice=Mock(), llm_retrieval=Mock(), llm_geracao=llm_geracao)

    llm_geracao.with_structured_output.assert_not_called()
    assert resultado.evidencia_suficiente is False
    assert resultado.motivo_recusa == "fora_de_escopo"
    assert resultado.evidencias == []


def test_responder_chama_llm_de_geracao_quando_score_acima_do_threshold(monkeypatch):
    doc = _doc("r1")
    docs_com_score_alto = [(doc, 0.80)]
    monkeypatch.setattr(pipeline, "recuperar", Mock(return_value=docs_com_score_alto))

    resposta_do_llm = _resposta([Evidencia(review_id="r1", nota=5, data="2018-01-01", trecho="review r1")])
    llm_geracao = Mock()
    llm_geracao.with_structured_output.return_value.invoke.return_value = resposta_do_llm

    resultado = responder("pergunta respondivel", indice=Mock(), llm_retrieval=Mock(), llm_geracao=llm_geracao)

    llm_geracao.with_structured_output.assert_called_once_with(RespostaRAG)
    assert resultado is resposta_do_llm


def test_responder_propaga_erro_quando_llm_cita_review_id_inexistente(monkeypatch):
    docs_com_score_alto = [(_doc("r1"), 0.80)]
    monkeypatch.setattr(pipeline, "recuperar", Mock(return_value=docs_com_score_alto))

    resposta_com_citacao_invalida = _resposta(
        [Evidencia(review_id="r_inventado", nota=5, data="2018-01-01", trecho="texto")]
    )
    llm_geracao = Mock()
    llm_geracao.with_structured_output.return_value.invoke.return_value = resposta_com_citacao_invalida

    try:
        responder("pergunta", indice=Mock(), llm_retrieval=Mock(), llm_geracao=llm_geracao)
        assert False, "deveria ter levantado ValueError"
    except ValueError:
        pass
