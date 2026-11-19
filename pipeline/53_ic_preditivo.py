#!/usr/bin/env python3
"""Intervalos de confianca e comparacao pareada das metricas preditivas.

Bootstrap por jogador sobre o out-of-fold do pacote oficial: IC 95% para C, IBS
e AUC media, diferenca de C pareada e sensibilidade ao conjunto de folds. Le a
matriz de sobrevivencia que `tgb_pacote_oficial` grava e exige que a replica
reproduza `metricas_walkforward_folds.csv` antes de reamostrar.

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
    "documentos/estudo_ic_preditivo/run_ic.py",
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
