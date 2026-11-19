#!/usr/bin/env python3
"""Quanto o desfecho futuro no treino infla o C do walk-forward.

O protocolo oficial rotula cada linha de treino so com o que se sabia na origem do
fold (`tgb_protocol.rotulo_em_tempo_real`). Este estudo refaz o walk-forward com o
rotulo calculado com a carreira inteira ate 2024, e compara:

- C medio nos folds de referencia, nos dois protocolos;
- C separado entre jogadores de teste que ja tinham linha no treino e jogadores novos.

Se esse rotulo vazasse o desfecho, o ganho apareceria so nos jogadores ja vistos
(o modelo "lembra" o desfecho deles). Nada oficial e sobrescrito.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
from sksurv.metrics import concordance_index_censored

ROOT = Path(__file__).resolve().parents[2]
RES = Path(__file__).resolve().parent / "resultados"
RES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "Predição" / "shared"))

spec = importlib.util.spec_from_file_location("ofi_vaz", ROOT / "documentos" / "estudo_modelos_finais" / "run_oficial.py")
ofi = importlib.util.module_from_spec(spec)
sys.modules["ofi_vaz"] = ofi
spec.loader.exec_module(ofi)
ofi.RES = Path(tempfile.mkdtemp())
split_correto = ofi.walkforward_split


def split_com_futuro(panel, season):
    """Rotulo com futuro: o treino carrega o tempo restante calculado com a carreira inteira."""
    tr = panel[panel["snapshot_season"] < season].copy()
    te = panel[panel["snapshot_season"] == season].copy()
    return tr, te


def main() -> None:
    medias, grupos = [], []
    for nome, split in (("correto", split_correto), ("com_futuro", split_com_futuro)):
        ofi.walkforward_split = split
        folds, preds = ofi.walkforward()
        f = folds[(folds.protocolo == "walkforward") & (folds.oficial == 1) & (folds.modelo != "km")]
        for (pos, mod), g in f.groupby(["posicao", "modelo"]):
            medias.append({"rotulo": nome, "posicao": pos, "modelo": mod, "c_index": float(g.c_index.mean())})
        preds = preds[preds.modelo != "km"]
        for pos in ("QB", "RB", "WR"):
            panel, _ = ofi.load_panel(pos)
            for (mod, S), g in preds[preds.posicao == pos].groupby(["modelo", "snapshot_season"]):
                vistos = set(panel.loc[panel.snapshot_season < S, "player_id"])
                for grupo, sub in (("ja_no_treino", g[g.player_id.isin(vistos)]), ("novo", g[~g.player_id.isin(vistos)])):
                    if sub.residual_event.sum() < 3 or len(sub) < 10:
                        continue
                    c = concordance_index_censored(sub.residual_event.astype(bool).to_numpy(),
                                                   sub.residual_duration_model.to_numpy(float), sub.risk.to_numpy(float))[0]
                    grupos.append({"rotulo": nome, "posicao": pos, "modelo": mod, "fold": int(S), "grupo": grupo,
                                   "n": int(len(sub)), "c_index": float(c)})
    ofi.walkforward_split = split_correto
    m = pd.DataFrame(medias)
    m.to_csv(RES / "c_por_protocolo.csv", index=False)
    g = pd.DataFrame(grupos)
    g.to_csv(RES / "c_por_grupo_folds.csv", index=False)
    resumo_g = g.groupby(["posicao", "modelo", "grupo", "rotulo"])["c_index"].mean().unstack(["grupo", "rotulo"])
    resumo_g.columns = [f"{a}__{b}" for a, b in resumo_g.columns]
    resumo_g.reset_index().to_csv(RES / "c_por_grupo.csv", index=False)
    piv = m.pivot_table(index=["posicao", "modelo"], columns="rotulo", values="c_index")
    resumo = {
        "c_correto": {f"{p}-{mo}": round(v, 4) for (p, mo), v in piv["correto"].items()},
        "c_com_futuro": {f"{p}-{mo}": round(v, 4) for (p, mo), v in piv["com_futuro"].items()},
        "faixa_correto": [round(float(piv["correto"].min()), 4), round(float(piv["correto"].max()), 4)],
        "faixa_com_futuro": [round(float(piv["com_futuro"].min()), 4), round(float(piv["com_futuro"].max()), 4)],
        "frac_teste_ja_no_treino": round(float(g[g.grupo == "ja_no_treino"].n.sum() / g.n.sum()), 4),
    }
    (RES / "resumo_vazamento.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(piv.round(4).to_string())
    print(resumo_g.round(3).to_string())
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
