# Voice of Customer Intelligence com RAG

> Tech Challenge - Fase 4 - POSTECH IA para Devs / Data Science
> Status: em desenvolvimento

## Descricao do projeto

(a preencher conforme o pipeline avanca)

Sistema de exploracao inteligente das avaliacoes de clientes da base
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data),
usando Retrieval-Augmented Generation (RAG) para responder perguntas em
linguagem natural com respostas fundamentadas nas avaliacoes reais.

## Arquitetura

```
Pergunta do usuario
    -> Processamento da pergunta
    -> Representacao vetorial / Embeddings
    -> Recuperacao das avaliacoes mais relevantes
    -> Construcao do contexto
    -> Geracao da resposta (LLM)
    -> Resposta fundamentada nas evidencias recuperadas
```

(detalhar decisoes de arquitetura aqui conforme implementado)

## Tecnologias e bibliotecas

- Python
- LangChain (orquestracao do pipeline RAG)
- Google Gemini (embeddings e geracao via API)
- FAISS (indice vetorial local)
- Pandas (preparacao dos dados)

## Preparacao da base

(a preencher: selecao dos dados, limpeza e normalizacao do texto, tratamento
de valores ausentes, estrategia de chunking/organizacao dos documentos,
criacao dos embeddings, armazenamento e indexacao)

## Estrategia do pipeline RAG

(a preencher: transformacao da pergunta, recuperacao semantica, selecao dos
conteudos mais relevantes, construcao do contexto enviado ao modelo, geracao
da resposta, e como o sistema reconhece perguntas sem evidencia suficiente)

## Decisoes tecnicas

(justificativas das principais escolhas, a preencher ao longo do projeto)

## Como rodar

### 1. Pre-requisitos

- Python 3.11+
- Conta no [Google AI Studio](https://aistudio.google.com/) para gerar uma chave de API Gemini (camada gratuita disponivel)

### 2. Instalacao

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuracao

```bash
cp .env.example .env
# preencher GOOGLE_API_KEY no .env
```

### 4. Dados

Baixar o dataset do Kaggle ([Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data))
e colocar os CSVs em `data/raw/`.

### 5. Execucao

(a preencher conforme os scripts forem implementados)

## Exemplos de perguntas e respostas

(a preencher com exemplos reais, incluindo as evidencias recuperadas)

## Reproduzindo do zero

(a preencher: passo a passo completo, do ambiente limpo ate a primeira resposta gerada)
