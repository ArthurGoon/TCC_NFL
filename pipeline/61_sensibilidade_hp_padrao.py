#!/usr/bin/env python3
"""Sensibilidade do pacote oficial com os hiperparametros PADRAO (scikit-survival e lifelines).

Mesmo walk-forward, landmark e lista 2024 do passo `tgb_pacote_oficial`, trocando a
busca aninhada pelos valores padrao (`tgb_protocol.DEFAULT_*`). Serve para comparar
o protocolo principal com o que um analista obteria sem tunar. Grava em
`documentos/estudo_modelos_finais/resultados/sensibilidade_hp_default/`, sem tocar nos
artefatos do pacote oficial.

Executa a fonte abaixo via ``runpy`` com ``--hp default``, no mesmo padrao dos demais
passos deste diretorio.
"""
from __future__ import annotations

import os
import runpy
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

SOURCE = "documentos/estudo_modelos_finais/run_oficial.py"
ARGS = ["--hp", "default"]


def main() -> None:
    script = ROOT / SOURCE
    if not script.exists():
        raise FileNotFoundError(f"Fonte ausente: {SOURCE}")
    print(f"--- {SOURCE} {' '.join(ARGS)} ---", flush=True)
    argv_anterior = sys.argv
    sys.argv = [str(script), *ARGS]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = argv_anterior


if __name__ == "__main__":
    main()
