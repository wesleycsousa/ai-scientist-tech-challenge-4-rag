"""Testes do retrieval multi-query (requisito 2 do PDF: transformacao da
pergunta como etapa separada da recuperacao em si).

Os testes unitarios mockam o LLM e o indice FAISS pra testar so a logica de
orquestracao (uniao sem duplicar, numero de buscas feitas, corte no top-K) -
rapidos e sem custo de API. O teste marcado como integration roda contra o
indice FAISS e o Gemini de verdade; fica de fora da rodada padrao
(`pytest -m "not integration"`) porque e mais lento e consome cota da API.
"""

from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from retriever import _unir_sem_duplicar, gerar_variacoes_pergunta, recuperar


def _doc(review_id: str) -> Document:
    return Document(page_content=f"review {review_id}", metadata={"review_id": review_id})


def test_unir_sem_duplicar_mantem_maior_score_entre_buscas():
    doc = _doc("r1")
    resultado = _unir_sem_duplicar([
        [(doc, 0.70)],
        [(doc, 0.85)],
    ])
    assert resultado == [(doc, 0.85)]


def test_unir_sem_duplicar_preserva_ids_sem_sobreposicao():
    doc_a, doc_b = _doc("r1"), _doc("r2")
    resultado = _unir_sem_duplicar([[(doc_a, 0.60)], [(doc_b, 0.80)]])
    assert resultado == [(doc_b, 0.80), (doc_a, 0.60)]


def test_gerar_variacoes_pergunta_faz_parsing_linha_a_linha():
    llm_falso = Mock()
    llm_falso.invoke.return_value = Mock(text="Primeira reformulacao\nSegunda reformulacao\n")

    variacoes = gerar_variacoes_pergunta("pergunta original", llm_falso, n=2)

    assert variacoes == ["Primeira reformulacao", "Segunda reformulacao"]


def test_recuperar_sem_multi_query_faz_so_uma_busca_e_nao_chama_llm():
    indice_falso = Mock()
    indice_falso.similarity_search_with_relevance_scores.return_value = [(_doc("r1"), 0.9)]
    llm_falso = Mock()

    resultado = recuperar("pergunta", indice_falso, llm_falso, k=5, usar_multi_query=False)

    assert indice_falso.similarity_search_with_relevance_scores.call_count == 1
    llm_falso.invoke.assert_not_called()
    assert len(resultado) == 1


def test_recuperar_com_multi_query_busca_original_mais_variacoes():
    indice_falso = Mock()
    indice_falso.similarity_search_with_relevance_scores.return_value = [(_doc("r1"), 0.9)]
    llm_falso = Mock()
    llm_falso.invoke.return_value = Mock(text="variacao 1\nvariacao 2\n")

    recuperar("pergunta", indice_falso, llm_falso, k=5, usar_multi_query=True)

    # pergunta original + 2 variacoes geradas pelo LLM = 3 buscas no indice
    assert indice_falso.similarity_search_with_relevance_scores.call_count == 3


def test_recuperar_corta_no_top_k_final():
    docs_e_scores = [(_doc(f"r{i}"), 1.0 - i * 0.01) for i in range(10)]
    indice_falso = Mock()
    indice_falso.similarity_search_with_relevance_scores.return_value = docs_e_scores
    llm_falso = Mock()

    resultado = recuperar("pergunta", indice_falso, llm_falso, k=3, usar_multi_query=False)

    assert len(resultado) == 3


@pytest.mark.integration
def test_recuperar_contra_indice_e_llm_reais():
    """Smoke test contra o indice FAISS e o Gemini de verdade, sem mock.

    Fica marcado como integration e fora da rodada padrao porque consome
    cota da API e depende do indice ja estar montado em disco.
    """
    from dotenv import load_dotenv
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    from retriever import criar_llm

    load_dotenv()
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    indice = FAISS.load_local("data/processed/faiss_index", embeddings, allow_dangerous_deserialization=True)

    resultado = recuperar(
        "Quais os principais problemas relatados com a entrega dos produtos?",
        indice,
        criar_llm(),
        k=5,
    )

    assert len(resultado) == 5
    assert all(score > 0.5 for _, score in resultado)
