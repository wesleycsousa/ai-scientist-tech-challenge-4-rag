"""Recuperacao semantica das avaliacoes mais relevantes para uma pergunta.

Fluxo (requisito 2 do PDF: "transformacao da pergunta" e "recuperacao
semantica" sao etapas distintas, nao uma coisa so): o LLM gera variacoes da
pergunta original, cada variacao (+ a pergunta original) busca no FAISS
separadamente, os resultados sao unidos deduplicando por review_id
(mantendo o maior score quando a mesma review aparece em mais de uma
busca), e so entao corta no top-K final. Mesmo padrao do material do curso
(LangChain Aula03/video02).

MMR (diversidade na selecao final, pra nao trazer varias reviews quase
identicas) e etapa separada, ainda nao implementada aqui - ver cronograma
de quarta-feira.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

MODELO_CHAT = "gemini-3.8-flash"
NUM_VARIACOES = 2
K_FINAL = 5

PROMPT_VARIACOES = """Reformule a pergunta abaixo de {n} jeitos diferentes, \
mudando vocabulario e angulo mas mantendo exatamente o mesmo significado. \
Responda so com as {n} reformulacoes, uma por linha, sem numeracao nem \
texto extra.

Pergunta original: {pergunta}"""


def criar_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=MODELO_CHAT)


def gerar_variacoes_pergunta(pergunta: str, llm: ChatGoogleGenerativeAI, n: int = NUM_VARIACOES) -> list[str]:
    """Pede ao LLM pra reformular a pergunta de n jeitos diferentes (transformacao da pergunta)."""
    resposta = llm.invoke(PROMPT_VARIACOES.format(n=n, pergunta=pergunta))
    linhas = [linha.strip() for linha in resposta.text.strip().split("\n") if linha.strip()]
    return linhas[:n]


def _buscar_uma_pergunta(indice: FAISS, pergunta: str, k: int) -> list[tuple[Document, float]]:
    return indice.similarity_search_with_relevance_scores(pergunta, k=k)


def _unir_sem_duplicar(resultados_por_pergunta: list[list[tuple[Document, float]]]) -> list[tuple[Document, float]]:
    """Junta os resultados de varias buscas, mantendo so a maior score quando o mesmo review_id repete."""
    melhor_por_id: dict[str, tuple[Document, float]] = {}
    for resultados in resultados_por_pergunta:
        for documento, score in resultados:
            review_id = documento.metadata["review_id"]
            if review_id not in melhor_por_id or score > melhor_por_id[review_id][1]:
                melhor_por_id[review_id] = (documento, score)
    return sorted(melhor_por_id.values(), key=lambda par: par[1], reverse=True)


def recuperar(
    pergunta: str,
    indice: FAISS,
    llm: ChatGoogleGenerativeAI,
    k: int = K_FINAL,
    usar_multi_query: bool = True,
) -> list[tuple[Document, float]]:
    """Recupera as k avaliacoes mais relevantes pra uma pergunta.

    Com usar_multi_query=True (padrao): busca a pergunta original + variacoes
    geradas pelo LLM, une sem duplicar por review_id, corta no top-k final.
    Com usar_multi_query=False: so a busca direta da pergunta original - existe
    pra comparar os dois modos nas perguntas de teste (cronograma de terca).
    """
    perguntas = [pergunta]
    if usar_multi_query:
        perguntas += gerar_variacoes_pergunta(pergunta, llm)

    resultados_por_pergunta = [_buscar_uma_pergunta(indice, p, k) for p in perguntas]
    unidos = _unir_sem_duplicar(resultados_por_pergunta)
    return unidos[:k]
