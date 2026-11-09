"""Roda data/perguntas_teste.json pelo pipeline completo e monta a tabela de avaliacao.

Reaproveitavel: diferente do script de calibracao do threshold (descartavel, rodado uma vez),
esse aqui roda de novo a cada ajuste de prompt/top-K feito nesta etapa. A coluna veredito fica
sempre vazia - exige leitura humana da resposta, nao tem como automatizar sem outro LLM
julgando (complexidade desproporcional pro escopo do desafio).

Saida em docs/, nao em _apoio/: essa tabela e evidencia de avaliacao do sistema (requisitos
3 e 4 do PDF), entao faz parte da entrega, diferente do material de apoio pessoal. Rodar de
novo sobrescreve o arquivo e limpa a coluna veredito - preencher de novo a mao depois.
"""

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).parent.parent
sys.path.insert(0, str(RAIZ / "src" / "generation"))
sys.path.insert(0, str(RAIZ / "src" / "retrieval"))
sys.path.insert(0, str(RAIZ / "src" / "ingestion"))

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from build_index import CAMINHO_FAISS_INDEX, MODELO_EMBEDDING
from pipeline import criar_llm_geracao, responder
from retriever import criar_llm

CAMINHO_PERGUNTAS = RAIZ / "data" / "perguntas_teste.json"
CAMINHO_SAIDA = RAIZ / "docs" / "avaliacao_perguntas_teste.md"


def carregar_indice() -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDING)
    return FAISS.load_local(str(RAIZ / CAMINHO_FAISS_INDEX), embeddings, allow_dangerous_deserialization=True)


def avaliar_pergunta(item: dict, indice: FAISS, llm_retrieval, llm_geracao) -> dict:
    pergunta = item["pergunta"]
    try:
        resposta = responder(pergunta, indice, llm_retrieval, llm_geracao)
    except ValueError:
        return {
            **item,
            "evidencias_resumo": "-",
            "resposta": "[ERRO: citacao invalida fora do contexto]",
            "motivo_recusa": "-",
        }

    return {
        **item,
        "evidencias_resumo": (
            ", ".join(f"{e.review_id[:8]} (nota {e.nota})" for e in resposta.evidencias)
            if resposta.evidencias
            else "-"
        ),
        "resposta": resposta.resposta.replace("\n", " "),
        "motivo_recusa": resposta.motivo_recusa or "-",
    }


def montar_tabela_markdown(linhas: list[dict]) -> str:
    cabecalho = "| id | categoria | pergunta | evidencias | resposta | motivo_recusa | veredito |\n"
    cabecalho += "|---|---|---|---|---|---|---|\n"
    corpo = ""
    for linha in linhas:
        corpo += (
            f"| {linha['id']} | {linha['categoria']} | {linha['pergunta']} | "
            f"{linha['evidencias_resumo']} | {linha['resposta']} | {linha['motivo_recusa']} | |\n"
        )
    return cabecalho + corpo


def main() -> None:
    load_dotenv(RAIZ / ".env")
    indice = carregar_indice()
    llm_retrieval = criar_llm()
    llm_geracao = criar_llm_geracao()

    perguntas = json.loads(CAMINHO_PERGUNTAS.read_text())

    linhas = []
    for item in perguntas:
        print(f"avaliando {item['id']}: {item['pergunta'][:60]}...")
        linhas.append(avaliar_pergunta(item, indice, llm_retrieval, llm_geracao))

    tabela = montar_tabela_markdown(linhas)
    CAMINHO_SAIDA.write_text(tabela)
    print(f"\ntabela salva em {CAMINHO_SAIDA}")


if __name__ == "__main__":
    main()
