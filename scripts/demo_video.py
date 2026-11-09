"""Demonstracao narrada do pipeline, pro video de apresentacao (requisito do desafio).

Mesmo espirito do executar_e_mostrar() da Aula04 (LangChain): chama as mesmas pecas que
responder() ja usa, na mesma ordem, so que imprimindo cada etapa entre as chamadas - a
logica de decisao fica intacta em pipeline.py/retriever.py, esse arquivo so narra o que
ja acontece la dentro. gerar_variacoes_pergunta() roda aqui e de novo dentro de recuperar()
(1 chamada de API redundante por caso) - aceitavel num script ocasional de demonstracao,
onde clareza didatica vale mais que economizar uma chamada.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).parent.parent
sys.path.insert(0, str(RAIZ / "src" / "generation"))
sys.path.insert(0, str(RAIZ / "src" / "retrieval"))
sys.path.insert(0, str(RAIZ / "src" / "ingestion"))

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from build_index import CAMINHO_FAISS_INDEX, MODELO_EMBEDDING
from pipeline import THRESHOLD_FORA_ESCOPO, RespostaRAG, criar_llm_geracao, gerar_resposta, validar_citacoes
from retriever import criar_llm, gerar_variacoes_pergunta, recuperar


def carregar_indice() -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDING)
    return FAISS.load_local(str(RAIZ / CAMINHO_FAISS_INDEX), embeddings, allow_dangerous_deserialization=True)


def imprimir_resposta(resposta: RespostaRAG) -> None:
    print(f"\n{resposta.resposta}\n")
    if resposta.evidencias:
        print("Evidências usadas:")
        for evidencia in resposta.evidencias:
            print(f"  - [{evidencia.review_id}] nota {evidencia.nota} ({evidencia.data}): {evidencia.trecho}")
    elif resposta.motivo_recusa:
        print(f"(recusa - motivo: {resposta.motivo_recusa})")


def demonstrar(
    titulo: str,
    pergunta: str,
    indice: FAISS,
    llm_retrieval: ChatGoogleGenerativeAI,
    llm_geracao: ChatGoogleGenerativeAI,
) -> None:
    print(f"\n{'=' * 60}\n{titulo}\n{'=' * 60}")
    print(f"Pergunta: {pergunta}")

    variacoes = gerar_variacoes_pergunta(pergunta, llm_retrieval)
    print("\n[transformação da pergunta] variações geradas pelo multi-query:")
    for variacao in variacoes:
        print(f"  - {variacao}")

    documentos_e_scores = recuperar(pergunta, indice, llm_retrieval)
    print("\n[recuperação] documentos encontrados:")
    if not documentos_e_scores:
        print("  (nenhum documento retornado)")
    for documento, score in documentos_e_scores:
        metadata = documento.metadata
        print(f"  - [{metadata['review_id'][:8]}] score {score:.3f} nota {metadata['review_score']}: {documento.page_content}")

    melhor_score = max((score for _, score in documentos_e_scores), default=0.0)
    print(f"\n[threshold] melhor score = {melhor_score:.3f} | corte (THRESHOLD_FORA_ESCOPO) = {THRESHOLD_FORA_ESCOPO}")

    if melhor_score < THRESHOLD_FORA_ESCOPO:
        print("  -> abaixo do corte: recusa direto, sem gastar uma chamada do LLM de geração")
        resposta = RespostaRAG(
            evidencia_suficiente=False,
            motivo_recusa="fora_de_escopo",
            resposta="Essa pergunta não parece ter relação com as avaliações de clientes da base.",
            evidencias=[],
        )
    else:
        print("  -> acima do corte: segue pro LLM de geração")
        resposta = gerar_resposta(pergunta, documentos_e_scores, llm_geracao)
        print("\n[geração] schema estruturado recebido do LLM:")
        print(f"  evidencia_suficiente = {resposta.evidencia_suficiente}")
        print(f"  motivo_recusa        = {resposta.motivo_recusa}")
        print(f"  resposta             = {resposta.resposta}")
        print(f"  evidencias citadas   = {len(resposta.evidencias)}")

        resposta = validar_citacoes(resposta, documentos_e_scores)
        print("\n[validação] todas as citações confirmadas contra o contexto recuperado - OK")

    print("\n[resposta final ao usuário]")
    imprimir_resposta(resposta)


if __name__ == "__main__":
    load_dotenv(RAIZ / ".env")
    indice = carregar_indice()
    llm_retrieval = criar_llm()
    llm_geracao = criar_llm_geracao()

    demonstrar(
        "CASO 1 — pergunta respondível pela base",
        "O que os clientes elogiam sobre o prazo de entrega?",
        indice,
        llm_retrieval,
        llm_geracao,
    )
    demonstrar(
        "CASO 2 — fora de escopo (recusa decidida pelo threshold, sem chamar o LLM de geração)",
        "Qual é a capital da França?",
        indice,
        llm_retrieval,
        llm_geracao,
    )
    demonstrar(
        "CASO 3 — evidência insuficiente (recusa decidida pelo LLM via schema)",
        "Qual o tempo medio de entrega relatado pelos clientes?",
        indice,
        llm_retrieval,
        llm_geracao,
    )
    demonstrar(
        "CASO 4 — sem evidência relevante (recusa decidida pelo LLM, motivo diferente do caso 3)",
        "Qual a nota media dos produtos de informatica?",
        indice,
        llm_retrieval,
        llm_geracao,
    )
