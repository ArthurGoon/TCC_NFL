#!/usr/bin/env python3
"""Gera a Figura 1.1 (campo e posicoes da NFL) em `Escrita Final/figuras/`.

A figura e desenhada por `plot_campo_posicoes_nfl`, de
`Predição/shared/thesis_figure_style.py`.

Nao altera estilo, tamanho nem aparencia: apenas chama a funcao ja calibrada e
aprovada, gravando onde o LaTeX le.
"""
from __future__ import annotations

import os
import sys
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
sys.path.insert(0, str(ROOT / "Predição" / "shared"))

from thesis_figure_style import plot_campo_posicoes_nfl  # noqa: E402

DEST = ROOT / "Escrita Final" / "figuras" / "campo_posicoes_nfl.png"


def main() -> int:
    print("Figura 1.1: campo e posicoes da NFL")
    saida = plot_campo_posicoes_nfl(DEST)
    pdf = Path(saida).with_suffix(".pdf")
    print(f"  gravado: {pdf.relative_to(ROOT)}")
    print(f"  gravado: {Path(saida).relative_to(ROOT)}")
    if not pdf.exists():
        raise RuntimeError(f"PDF nao foi gerado: {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
