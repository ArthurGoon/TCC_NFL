#!/usr/bin/env python3
"""Sensibilidade de fronteira da grade de hiperparametros (Tabela do tune).

Le os hiperparametros escolhidos pelo `tune_preditivo` e, para cada aresta da
grade tocada em algum fold, mede o C interno (calibracao) e o C de teste um
passo alem da fronteira, com os demais eixos fixados nos valores escolhidos.
Nao reabre o congelamento de lambda: e sondagem condicional, nao re-tunagem.

Executa a fonte abaixo via ``runpy``, no mesmo padrao dos demais passos deste
diretorio.
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
    "documentos/estudo_fronteira_tune/run_fronteira.py",
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
