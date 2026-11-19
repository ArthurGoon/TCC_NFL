#!/usr/bin/env python3
"""Contraste do Aalen preditivo: diagnostico de posto e especificacao curta.

Mostra por que o Aalen nao estima com o pool grande e que ele volta a estimar
com a especificacao curta da inferencia (fora do 'mesmo X').

Executa as fontes abaixo via ``runpy``, no mesmo padrao dos demais passos
deste diretorio. As fontes ficam em ``documentos/estudo_*/``; o manifesto lista os
resultados que elas gravam.
"""
from __future__ import annotations

import os
import runpy
import shutil
from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "run_all.py").exists() and (candidate / "DataBase").exists():
            return candidate
    raise RuntimeError("Project root not found")


ROOT = find_project_root()
os.chdir(ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

SOURCES = [
    "documentos/estudo_aalen_preditivo/diag_aalen_fold.py",
    "documentos/estudo_aalen_selecao/run_contraste.py",
]

# O contraste grava em `estudo_aalen_selecao`; as mesmas tabelas e a figura tambem
# ficam em `estudo_aalen_preditivo`, junto do diagnostico de posto.
COPIAS = [
    ("figuras/cindex_contraste.pdf", "figuras/cindex_contraste.pdf"),
    ("figuras/cindex_contraste.png", "figuras/cindex_contraste.png"),
    ("resultados/metricas_walkforward_aalen_spec_curta.csv", "resultados/metricas_walkforward_aalen_spec_curta.csv"),
    ("resultados/posto_por_fold.csv", "resultados/posto_por_fold.csv"),
    ("resultados/resumo_oficial_2018_2021.csv", "resultados/resumo_oficial_2018_2021.csv"),
]


def copiar_para_o_estudo_preditivo() -> None:
    origem = ROOT / "documentos" / "estudo_aalen_selecao"
    destino = ROOT / "documentos" / "estudo_aalen_preditivo"
    for de, para in COPIAS:
        fonte = origem / de
        alvo = destino / para
        if not fonte.exists():
            raise FileNotFoundError(f"Artefato de origem ausente para a copia: {de}")
        alvo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fonte, alvo)
        print(f"  copiado: {alvo.relative_to(ROOT)}", flush=True)


def main() -> None:
    for relative in SOURCES:
        script = ROOT / relative
        if not script.exists():
            raise FileNotFoundError(f"Fonte ausente: {relative}")
        print(f"--- {relative} ---", flush=True)
        runpy.run_path(str(script), run_name="__main__")
    copiar_para_o_estudo_preditivo()


if __name__ == "__main__":
    main()
