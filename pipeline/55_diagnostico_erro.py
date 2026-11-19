#!/usr/bin/env python3
"""Mapa onde/quando/como do erro preditivo e calibracao por quintil.

Le o out-of-fold do pacote oficial e a matriz de sobrevivencia; nao ajusta
modelo algum. Confere tambem a coincidencia de superestimacao grave entre
familias nos cortes de 2 e 3 anos (a do corte 2 replica `auditoria.json`).

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
    "documentos/estudo_diagnostico_erro/run_diagnostico.py",
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
