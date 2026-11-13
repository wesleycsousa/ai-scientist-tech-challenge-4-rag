"""Geracao dos embeddings em lotes espacados no tempo, com checkpoint em disco.

Isolado do resto da ingestao de proposito: isso NAO e como seria feito numa
API sem limite de uso. La cabe uma chamada so, com o batch inteiro de uma vez
(tipo `embeddings.embed_documents(textos)` direto, como o
`FAISS.from_documents()` original fazia). Essa complicacao de lotes, pausa,
retry em 429 e checkpoint em disco existe por causa dos limites da API do
Gemini no tier pago (Tier 1, confirmado com o suporte do Google):

- 100 textos por chamada e o maximo absoluto aceito pelo endpoint
  (`batchEmbedContents` rejeita qualquer coisa acima disso).
- 3.000 RPM (requisicoes por minuto) e 1.000.000 TPM (tokens por minuto) -
  e cada TEXTO dentro do lote conta como 1 unidade de RPM, nao a chamada
  HTTP como um todo. Agrupar 100 textos numa chamada economiza latencia de
  rede, mas nao "engana" a cota: 1 chamada com 100 textos ainda consome 100
  unidades de RPM.
- Sem limite diario de requisicoes no Tier 1 (esse teto so existe no tier
  gratuito - foi o que travou a execucao antes do upgrade).

O checkpoint alimenta o indice FAISS incrementalmente (lote a lote), em vez
de acumular todos os vetores numa lista Python e so montar o indice no
final. Essa segunda abordagem foi a original e quebrou na pratica: o
OOM killer do Linux matou o processo em ~76% da base (31 mil de 40 mil
reviews), porque manter ~31 mil vetores como objetos Python (lista de
tuplas de float) consome bem mais RAM do que o array compacto que o
proprio FAISS usa internamente pra guardar a mesma informacao. Alimentando
o indice lote a lote, cada vetor vira parte do array do FAISS e pode sair
da memoria Python assim que o lote e processado - o uso de RAM para de
crescer com o tamanho total da base.
"""

import pickle
import random
import re
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError

TAMANHO_LOTE = 100
# ritmo visado: ~25 lotes/min = ~2.500 textos/min, uns 80% dos 3.000 RPM do
# Tier 1 - a margem de seguranca que o proprio suporte do Google recomendou
PAUSA_ENTRE_LOTES_SEGUNDOS = 2.4
MAX_TENTATIVAS_429 = 5
PAUSA_BASE_RETRY_429_SEGUNDOS = 2.0
LOTES_POR_CHECKPOINT = 5
CAMINHO_CHECKPOINT_INDICE = Path("data/processed/indice_checkpoint")
CAMINHO_CHECKPOINT_LOTE = Path("data/processed/indice_checkpoint_lote.txt")


def _dividir_em_lotes(documentos: list[Document]) -> list[list[Document]]:
    return [documentos[i : i + TAMANHO_LOTE] for i in range(0, len(documentos), TAMANHO_LOTE)]


def _carregar_checkpoint(embeddings: GoogleGenerativeAIEmbeddings) -> tuple[FAISS | None, int]:
    """Retoma de onde parou, se existir checkpoint (indice parcial) de uma execucao anterior."""
    if not CAMINHO_CHECKPOINT_LOTE.exists():
        return None, 0
    proximo_lote = int(CAMINHO_CHECKPOINT_LOTE.read_text().strip())
    indice = FAISS.load_local(
        str(CAMINHO_CHECKPOINT_INDICE), embeddings, allow_dangerous_deserialization=True
    )
    print(
        f"checkpoint encontrado: retomando do lote {proximo_lote} "
        f"({indice.index.ntotal} documentos ja indexados)"
    )
    return indice, proximo_lote


def _salvar_checkpoint(indice: FAISS, proximo_lote: int) -> None:
    CAMINHO_CHECKPOINT_INDICE.parent.mkdir(parents=True, exist_ok=True)
    indice.save_local(str(CAMINHO_CHECKPOINT_INDICE))
    CAMINHO_CHECKPOINT_LOTE.write_text(str(proximo_lote))


def _limpar_checkpoint() -> None:
    CAMINHO_CHECKPOINT_LOTE.unlink(missing_ok=True)
    if CAMINHO_CHECKPOINT_INDICE.exists():
        for arquivo in CAMINHO_CHECKPOINT_INDICE.glob("*"):
            arquivo.unlink()
        CAMINHO_CHECKPOINT_INDICE.rmdir()


def _extrair_retry_delay_segundos(erro: Exception) -> float | None:
    """Le o retryDelay que a propria API devolve no corpo do erro 429, quando vem."""
    correspondencia = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)s", str(erro))
    return float(correspondencia.group(1)) if correspondencia else None


def _embed_com_retry(
    embeddings: GoogleGenerativeAIEmbeddings, textos: list[str], indice_lote: int, total_lotes: int
) -> list[list[float]]:
    """Tenta embedar um lote; em 429, espera com backoff exponencial (+jitter) e tenta de novo.

    Prioriza o retryDelay que a API sugere no proprio erro; se nao vier, usa
    backoff exponencial (2s, 4s, 8s, 16s, 32s) como o Google recomendou.
    """
    tentativa = 0
    while True:
        try:
            return embeddings.embed_documents(textos)
        except GoogleGenerativeAIError as erro:
            tentativa += 1
            if getattr(erro.__cause__, "code", None) != 429 or tentativa >= MAX_TENTATIVAS_429:
                raise
            espera = _extrair_retry_delay_segundos(erro) or PAUSA_BASE_RETRY_429_SEGUNDOS * (2 ** (tentativa - 1))
            espera += random.uniform(0, 1)  # jitter - evita varios lotes reagindo no mesmo instante
            print(
                f"lote {indice_lote + 1}/{total_lotes}: 429 (tentativa {tentativa}/{MAX_TENTATIVAS_429}), "
                f"esperando {espera:.1f}s..."
            )
            time.sleep(espera)


def gerar_indice_em_lotes(documentos: list[Document], embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    """Embedda os documentos em lotes e alimenta o indice FAISS incrementalmente.

    Retoma de um checkpoint anterior se existir (ver _carregar_checkpoint).
    Cada lote entra no indice assim que e embedado (`add_embeddings`, ou
    `from_embeddings` pro primeiro lote) - os vetores daquele lote nao ficam
    guardados numa lista Python depois disso, só dentro do indice.
    """
    lotes = _dividir_em_lotes(documentos)
    indice, inicio = _carregar_checkpoint(embeddings)

    for i in range(inicio, len(lotes)):
        lote = lotes[i]
        textos_lote = [doc.page_content for doc in lote]

        vetores_lote = _embed_com_retry(embeddings, textos_lote, i, len(lotes))
        pares_lote = list(zip(textos_lote, vetores_lote))
        metadados_lote = [doc.metadata for doc in lote]

        if indice is None:
            indice = FAISS.from_embeddings(
                pares_lote, embeddings, metadatas=metadados_lote, distance_strategy=DistanceStrategy.COSINE
            )
        else:
            indice.add_embeddings(pares_lote, metadatas=metadados_lote)

        if (i + 1) % LOTES_POR_CHECKPOINT == 0 or i + 1 == len(lotes):
            _salvar_checkpoint(indice, i + 1)
            print(f"lote {i + 1}/{len(lotes)} indexado e salvo em checkpoint ({indice.index.ntotal} documentos)")

        time.sleep(PAUSA_ENTRE_LOTES_SEGUNDOS)

    _limpar_checkpoint()
    return indice
