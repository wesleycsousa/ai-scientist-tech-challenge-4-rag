"""Construcao da base de conhecimento: selecao, limpeza e indexacao das avaliacoes."""

import re
import unicodedata

import pandas as pd

CAMINHO_REVIEWS_RAW = "data/raw/olist_order_reviews_dataset.csv"
CAMINHO_REVIEWS_LIMPO = "data/processed/reviews_limpo.parquet"

COLUNAS_USADAS = [
    "review_id",
    "order_id",
    "review_score",
    "review_comment_message",
    "review_creation_date",
]
# descartei review_comment_title: so 11,7% das linhas tem (confirmado na EDA),
# o PDF do desafio tambem nao cita nenhum campo especifico pra usar

# contracoes informais comuns em review de PT-BR, mesmo mapa da aula de PLN
CONTRACOES = {
    "vc": "você", "tb": "também", "tbm": "também",
    "pq": "por que", "obg": "obrigado", "blz": "beleza",
    "td": "tudo", "msm": "mesmo", "vlw": "valeu",
    "n": "não", "naum": "não", "cmprar": "comprar",
    "mt": "muito", "pra": "para",
}


def limpar_normalizar(texto: str, remove_acentos: bool = False) -> str:
    """Normaliza o texto de uma review antes do embedding (receita da aula de PLN)."""
    texto = texto.lower()  # "Bom" e "bom" tem que virar o mesmo vetor
    texto = re.sub(r"<[^>]+>", " ", texto)  # html perdido no meio do texto
    texto = re.sub(r"http\S+|www\.\S+", " ", texto)  # link nao ajuda em nada semanticamente
    texto = re.sub(r"\S+@\S+", " ", texto)  # e-mail, mesma ideia
    palavras = [CONTRACOES.get(p, p) for p in texto.split()]  # "vc"->"você" etc
    texto = " ".join(palavras)
    # fica so letra (com acento), espaco e pontuacao basica - tira emoji, aspas, digito
    texto = re.sub(r"[^a-záàâãäéèêëíìîïóòôõöúùûüç\s.,!?]", " ", texto)
    if remove_acentos:
        nfkd = unicodedata.normalize("NFKD", texto)
        texto = "".join(c for c in nfkd if not unicodedata.combining(c))
    texto = re.sub(r"\s+", " ", texto).strip()  # espaco duplo/quebra de linha solta
    return texto


def carregar_e_limpar_reviews(caminho_csv: str = CAMINHO_REVIEWS_RAW) -> pd.DataFrame:
    """Carrega as reviews da Olist e devolve so o que entra na base de conhecimento."""
    df = pd.read_csv(caminho_csv)
    df = df[COLUNAS_USADAS]

    df = df.dropna(subset=["review_comment_message"])  # sem texto nao ha o que indexar

    # mesma review aparece com 2 order_id quando o pedido foi dividido entre
    # vendedores (visto na EDA: 814 casos) -> sem isso o texto entraria 2x no indice
    df = df.drop_duplicates(subset=["review_id"], keep="first")

    df["review_comment_message"] = df["review_comment_message"].apply(limpar_normalizar)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    reviews = carregar_e_limpar_reviews()
    print(f"reviews limpas: {len(reviews)} de 99224 originais")
    reviews.to_parquet(CAMINHO_REVIEWS_LIMPO, index=False)
    print(f"salvo em {CAMINHO_REVIEWS_LIMPO}")

# TODO: organizacao dos documentos (chunking)
# TODO: geracao dos embeddings (Gemini) e indexacao (FAISS)
