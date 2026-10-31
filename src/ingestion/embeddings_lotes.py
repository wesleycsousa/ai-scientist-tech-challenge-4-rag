"""Geracao dos embeddings em lotes espacados no tempo, com checkpoint em disco.

Isolado do resto da ingestao de proposito: isso NAO e como seria feito numa
API sem limite de tier gratuito. Lá cabe uma chamada so, com o batch inteiro
de uma vez (tipo `embeddings.embed_documents(textos)` direto, como o
`FAISS.from_documents()` original fazia). Essa complicacao de lotes, pausa,
retry em 429 e checkpoint em disco existe so por causa das cotas do tier
gratuito do Gemini (100 chamadas/minuto, 1000 chamadas/dia) - cada chamada
carrega no maximo 100 textos, e se o processo cair no meio (ex: bateu a cota
do dia), sem checkpoint todo o progresso daquela execucao se perde.
"""

import pickle
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError

TAMANHO_LOTE = 100
PAUSA_ENTRE_LOTES_SEGUNDOS = 1.0
PAUSA_RETRY_429_SEGUNDOS = 60
LOTES_POR_CHECKPOINT = 5
CAMINHO_CHECKPOINT = Path("data/processed/embeddings_checkpoint.pkl")


def _dividir_em_lotes(documentos: list[Document]) -> list[list[Document]]:
    return [documentos[i : i + TAMANHO_LOTE] for i in range(0, len(documentos), TAMANHO_LOTE)]


def _carregar_checkpoint() -> tuple[list[tuple[str, list[float]]], list[dict], int]:
    """Retoma de onde parou, se existir checkpoint de uma execucao anterior."""
    if not CAMINHO_CHECKPOINT.exists():
        return [], [], 0
    with open(CAMINHO_CHECKPOINT, "rb") as arquivo:
        estado = pickle.load(arquivo)
    print(
        f"checkpoint encontrado: retomando do lote {estado['proximo_lote']} "
        f"({len(estado['pares'])} documentos ja embedados)"
    )
    return estado["pares"], estado["metadados"], estado["proximo_lote"]


def _salvar_checkpoint(pares: list, metadados: list, proximo_lote: int) -> None:
    CAMINHO_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with open(CAMINHO_CHECKPOINT, "wb") as arquivo:
        pickle.dump({"pares": pares, "metadados": metadados, "proximo_lote": proximo_lote}, arquivo)


def gerar_embeddings_em_lotes(
    documentos: list[Document], embeddings: GoogleGenerativeAIEmbeddings
) -> tuple[list[tuple[str, list[float]]], list[dict]]:
    """Embedda os documentos em lotes, com pausa, retry em 429 e checkpoint em disco.

    Retoma de um checkpoint anterior se ele existir (ver _carregar_checkpoint).
    Ao terminar com sucesso, apaga o checkpoint - o indice FAISS final e que
    passa a ser o artefato valido, nao faz sentido manter os dois.
    """
    lotes = _dividir_em_lotes(documentos)
    pares_texto_vetor, metadados, inicio = _carregar_checkpoint()

    for i in range(inicio, len(lotes)):
        lote = lotes[i]
        textos_lote = [doc.page_content for doc in lote]

        try:
            vetores_lote = embeddings.embed_documents(textos_lote)
        except GoogleGenerativeAIError as erro:
            if getattr(erro.__cause__, "code", None) != 429:
                raise
            print(f"lote {i + 1}/{len(lotes)}: 429, esperando {PAUSA_RETRY_429_SEGUNDOS}s e tentando de novo...")
            time.sleep(PAUSA_RETRY_429_SEGUNDOS)
            vetores_lote = embeddings.embed_documents(textos_lote)

        pares_texto_vetor.extend(zip(textos_lote, vetores_lote))
        metadados.extend(doc.metadata for doc in lote)

        if (i + 1) % LOTES_POR_CHECKPOINT == 0 or i + 1 == len(lotes):
            _salvar_checkpoint(pares_texto_vetor, metadados, i + 1)
            print(f"lote {i + 1}/{len(lotes)} embedado e salvo em checkpoint ({len(pares_texto_vetor)} documentos)")

        time.sleep(PAUSA_ENTRE_LOTES_SEGUNDOS)

    CAMINHO_CHECKPOINT.unlink(missing_ok=True)
    return pares_texto_vetor, metadados
