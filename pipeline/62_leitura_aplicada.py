#!/usr/bin/env python3
"""Leitura aplicada dos resultados preditivos (Secao de apoio). Passo 62.

Executa ``documentos/estudo_leitura_aplicada/run_leitura_aplicada.py`` via ``runpy``, no
mesmo padrao de ``43_leitura_preditiva.py``. Nao reajusta modelos: le artefatos existentes
e grava casos, tipologia do erro, concordancia entre modelos, inferencia x importancia e
DP das covariaveis em ``documentos/estudo_leitura_aplicada/resultados/``.
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
    "documentos/estudo_leitura_aplicada/run_leitura_aplicada.py",
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
