#!/usr/bin/env python3
"""Busca aninhada de hiperparametros (Tabela 5.8), fold a fold. PESADO.

Grade larga: Cox 7 pontos, GBSA 121, RSF 226, em QB, RB e WR, para as temporadas
2018 a 2024. A escolha acontece so com o treino de cada fold (ultimos 2 anos desse
treino como validacao interna). Nao e pulado por `--fast`: o pacote oficial depende
da tabela que ele grava.

Executa as fontes abaixo via ``runpy``, no mesmo padrao dos demais passos
deste diretorio. As fontes ficam em ``documentos/estudo_*/``; o manifesto lista os
resultados que elas gravam.
"""
from __future__ import annotations

import os
import runpy
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
    "documentos/estudo_tune_preditivo/run_tune.py",
]


def main() -> None:
    for relative in SOURCES:
        script = ROOT / relative
        if not script.exists():
            raise FileNotFoundError(f"Fonte ausente: {relative}")
        print(f"--- {relative} ---", flush=True)
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
