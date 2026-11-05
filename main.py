"""Ponto de entrada: pergunta -> recuperacao -> geracao, de ponta a ponta.

Carrega o indice FAISS ja construido (ver src/ingestion/build_index.py) e abre um loop
simples pedindo a pergunta, mostrando a resposta fundamentada e as evidencias usadas.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).parent
sys.path.insert(0, str(RAIZ / "src" / "generation"))
sys.path.insert(0, str(RAIZ / "src" / "retrieval"))
sys.path.insert(0, str(RAIZ / "src" / "ingestion"))

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from build_index import CAMINHO_FAISS_INDEX, MODELO_EMBEDDING
from pipeline import RespostaRAG, criar_llm_geracao, responder
from retriever import criar_llm


def carregar_indice() -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDING)
    return FAISS.load_local(CAMINHO_FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)


def imprimir_resposta(resposta: RespostaRAG) -> None:
    print(f"\n{resposta.resposta}\n")
    if resposta.evidencias:
        print("Evidências usadas:")
        for evidencia in resposta.evidencias:
            print(f"  - [{evidencia.review_id}] nota {evidencia.nota} ({evidencia.data}): {evidencia.trecho}")
    elif resposta.motivo_recusa:
        print(f"(recusa - motivo: {resposta.motivo_recusa})")
    print()


def main() -> None:
    load_dotenv()
    indice = carregar_indice()
    llm_retrieval = criar_llm()
    llm_geracao = criar_llm_geracao()

    print("Pergunte sobre as avaliações de clientes da Olist (Ctrl+C pra sair)\n")
    while True:
        pergunta = input("Pergunta: ").strip()
        if not pergunta:
            continue

        try:
            resposta = responder(pergunta, indice, llm_retrieval, llm_geracao)
        except ValueError:
            print("\nNão consegui confirmar a evidência desta resposta - tente reformular a pergunta.\n")
            continue

        imprimir_resposta(resposta)


if __name__ == "__main__":
    main()
