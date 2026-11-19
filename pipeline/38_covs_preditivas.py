#!/usr/bin/env python3
"""Gera o X preditivo CONGELADO: `Predicao/shared/covs_preditivas_{POS}.json`.

O ramo preditivo usa o mesmo painel posicional do ramo inferencial:

  1. base: `Predição/shared/df_cox_{POS}_features.pkl` (passo `add_features`);
  2. excluir identificadores e colunas de tempo, o prefixo `delta_`, as variacoes
     recentes e a rodada do draft, que nao fazem parte do painel;
  3. filtro de colinearidade `|r| > 0.85` CONGELADO nas temporadas 2012-2017, que
     nao avalia os indicadores de dado faltante, como no ramo inferencial.

O congelamento existe para que a escolha do X preditivo nao veja as temporadas
usadas como teste no walk-forward. E o mesmo X para KM, Cox, GBSA e RSF, de
modo que o C-index compare estimadores e nao listas de covariaveis.
"""
from __future__ import annotations

import json
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
sys.path.insert(0, str(ROOT / "Inferência" / "shared"))

import pandas as pd  # noqa: E402

from survival_utils import filter_collinear_covariates  # noqa: E402

POSITIONS = ("QB", "RB", "WR")
PRED_SHARED = ROOT / "Predição" / "shared"

# Identificadores e colunas de tempo: nao sao covariaveis.
EXCLUDE_COLS = {
    "player_id",
    "player_name",
    "season",
    "start",
    "stop",
    "event",
    "entry",
}
FREEZE_SEASONS = (2012, 2017)
COLLINEAR_THRESHOLD = 0.85
# Fora do painel: metadado zerado, variacoes recentes (testadas no passo 68) e rodada do draft,
# que repete a posicao no draft.
ALWAYS_EXCLUDE = {
    "contract_year_flag",
    "depth_worsened_lag",
    "offense_pct_declined_lag",
    "games_missed_increased_lag",
    "efficiency_declined_lag",
    "seasons_since_gap",
    "rookie_flag",
    "draft_round",
}


def candidate_covariates(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in EXCLUDE_COLS
        and c not in ALWAYS_EXCLUDE
        and not c.startswith("delta_")
        and not c.endswith("_missing")
    ]


def indicadores_faltantes(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_missing")]


def main() -> int:
    print("X preditivo congelado (mesmo conjunto para KM, Cox, GBSA e RSF)")
    print(f"Congelamento: {FREEZE_SEASONS[0]}-{FREEZE_SEASONS[1]} | |r| > {COLLINEAR_THRESHOLD}")
    for pos in POSITIONS:
        panel_path = PRED_SHARED / f"df_cox_{pos}_features.pkl"
        if not panel_path.exists():
            raise FileNotFoundError(f"Painel ausente: {panel_path}. Rode o passo add_features.")
        df = pd.read_pickle(panel_path)
        candidatas = candidate_covariates(df)
        freeze = df[df["season"].between(*FREEZE_SEASONS)]
        selecionadas = filter_collinear_covariates(
            freeze, candidatas, threshold=COLLINEAR_THRESHOLD, verbose=False
        )
        selecionadas = list(selecionadas) + indicadores_faltantes(df)
        payload = {
            "posicao": pos,
            "n": len(selecionadas),
            "freeze_train_seasons": f"{FREEZE_SEASONS[0]}-{FREEZE_SEASONS[1]}",
            "collinear_threshold": COLLINEAR_THRESHOLD,
            "covariates": list(selecionadas),
        }
        out = PRED_SHARED / f"covs_preditivas_{pos}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"  {pos}: {len(candidatas) + len(indicadores_faltantes(df))} candidatas -> {len(selecionadas)} covariaveis "
            f"-> {out.relative_to(ROOT)}"
        )
    print("\nX preditivo congelado concluido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
