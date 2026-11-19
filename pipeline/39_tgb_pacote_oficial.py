#!/usr/bin/env python3
"""Pacote preditivo OFICIAL do Cap. 5 (walk-forward 2018-2021, landmark, 2024).

E a fonte dos numeros oficiais da escrita: usa, em cada fold, o HP escolhido pela
busca aninhada (`pipeline/42`, tabela `hps_por_fold.csv`) e grava em
`documentos/estudo_modelos_finais`. Sensibilidade com valores padrao:
`run_oficial.py --hp default` (grava em `resultados/sensibilidade_hp_default/`).

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
    "documentos/estudo_modelos_finais/run_oficial.py",
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
