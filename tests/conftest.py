"""Deixa os modulos de src/ importaveis nos testes, do mesmo jeito que os
scripts da ingestao e do retrieval ja se importam entre si (import direto
pelo nome do arquivo, sem pacote)."""

import sys
from pathlib import Path

RAIZ = Path(__file__).parent.parent
sys.path.insert(0, str(RAIZ / "src" / "retrieval"))
sys.path.insert(0, str(RAIZ / "src" / "ingestion"))
