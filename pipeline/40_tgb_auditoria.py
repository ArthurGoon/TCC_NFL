#!/usr/bin/env python3
"""Auditoria do pacote oficial: protocolo, top-k, erros OOF e face validity 2024.

Nao reajusta modelos: le os CSV do passo anterior e o painel cru. E a fonte
das taxas de acerto em horizonte, do vies e dos casos citados no Cap. 5.

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
    "documentos/estudo_modelos_finais/auditar_oficial.py",
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
