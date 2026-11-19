#!/usr/bin/env python3
"""Diagnostico computacional Aalen com covariaveis selecionadas no painel completo."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "Predição" / "shared"))
sys.path.insert(0, str(ROOT / "Inferência" / "shared"))

from tgb_protocol import (  # noqa: E402
    PRED_SHARED,
    evaluate_fold,
    load_panel,
    mask_test_support,
    preprocess,
    walkforward_split,
    y_surv,
    evento_em_tempo_real_cp,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "resultados"
FIG = HERE / "figuras"
TMP = RES / "aalen_tmp"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

def _find_rscript() -> str:
    """Rscript do PATH, com fallback para instalacoes padrao.

    Antes havia um caminho absoluto fixo do Windows nesta linha, o que tornava o
    passo irreprodutivel em qualquer outra maquina e violava a auditoria.
    """
    found = shutil.which("Rscript")
    if found:
        return found
    for base in (Path.home() / "AppData" / "Local" / "Programs" / "R", Path("/usr/lib/R")):
        if not base.exists():
            continue
        matches = sorted(base.glob("R-*/bin/Rscript*"), reverse=True)
        matches += sorted(base.glob("R-*/bin/x64/Rscript*"), reverse=True)
        if matches:
            return str(matches[0])
    raise FileNotFoundError("Rscript nao encontrado no PATH; instale R 4.x.")


RSCRIPT = Path(_find_rscript())
R_FIT = HERE / "run_aalen_fit.R"
INFER_AALEN = ROOT / "Inferência" / "Modelagem de Aalen" / "resultados"
FORBIDDEN = {"years_exp", "first_observed_season"}
DIAGNOSTIC_SEASONS = (2018, 2019, 2020, 2021)
POSITIONS = ("QB", "RB", "WR")
DIAGNOSTIC_FLAGS = {
    "protocolo_poshoc": True,
    "avaliacao_prospectiva": False,
    "comparavel_com_ranking_principal": False,
    "backtest_honesto": False,
    "periodo_selecao_covariaveis": "2012-2024",
    "periodo_folds_diagnosticos": "2018-2021",
}


def load_inferential_list(pos: str) -> list[str]:
    raw = json.loads((INFER_AALEN / f"aalen_final_covs_{pos}.json").read_text(encoding="utf-8"))
    return [c for c in raw if c not in FORBIDDEN]


def filter_diagnostic_covariates(cols: list[str]) -> tuple[list[str], list[str]]:
    """Filtra a lista inferencial para o diagnostico computacional Aalen.

    As dummies `_missing` repetem o padrao de historico nao observado. No fold
    QB 2018 elas ficam colineares entre si depois da padronizacao e derrubam o
    posto de X(t). Sua remocao nao transforma esta analise pos-selecao em
    avaliacao prospectiva ou comparacao preditiva.
    """
    dropped = [c for c in cols if c.endswith("_missing")]
    keep = [c for c in cols if not c.endswith("_missing")]
    return keep, dropped


def design_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    events = df.loc[df["event"] == 1, "stop"].to_numpy(dtype=float)
    rows = []
    for t in np.unique(events):
        mask = (df["start"] < t) & (df["stop"] >= t)
        block = df.loc[mask, cols].to_numpy(dtype=float)
        if block.size == 0:
            continue
        X = np.hstack([np.ones((block.shape[0], 1)), block])
        xtx = X.T @ X
        rank = int(np.linalg.matrix_rank(xtx, tol=1e-8))
        rows.append(
            {
                "t": float(t),
                "Y": int(X.shape[0]),
                "p_plus_1": int(X.shape[1]),
                "rank": rank,
                "full_rank": int(rank == X.shape[1]),
                "deficit": int(X.shape[1] - rank),
            }
        )
    return pd.DataFrame(rows)


def run_aalen(train_cp: pd.DataFrame, test_rows: pd.DataFrame, covs: list[str], tag: str):
    train_csv = TMP / f"train_{tag}.csv"
    test_csv = TMP / f"test_{tag}.csv"
    pred_csv = TMP / f"pred_{tag}.csv"
    export_cols = ["player_id", "start", "stop", "event"] + covs
    train_cp[export_cols].to_csv(train_csv, index=False)
    test_export = test_rows.copy()
    if "stop" not in test_export.columns:
        test_export["stop"] = test_export.get("snapshot_stop", test_export["start"])
    if "event" not in test_export.columns:
        test_export["event"] = 0
    test_export[export_cols].to_csv(test_csv, index=False)
    proc = subprocess.run(
        [str(RSCRIPT), str(R_FIT), str(train_csv), str(test_csv), ",".join(covs), str(pred_csv)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not pred_csv.exists():
        err = (proc.stderr or proc.stdout or "")[-800:]
        return None, err, {}
    pred = pd.read_csv(pred_csv)
    summary_path = pred_csv.with_name(pred_csv.stem + "_summary.json")
    info = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return pred, None, info


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.45, color="0.65")


def previsao_por_experiencia() -> pd.DataFrame:
    """Resumo auxiliar pos-selecao por faixa de experiencia nos folds diagnosticos.

    Mostra onde o Aalen deixa de prever risco: nas faixas de carreira longa a matriz de
    covariaveis perde posto e os incrementos de B(t) ficam nulos ou negativos, de modo que
    o risco condicional truncado em zero da S = 1 (nenhuma saida prevista).
    """
    linhas = []
    for f in sorted(TMP.glob("pred_*_20*.csv")):
        if f.name.endswith("_summary.json"):
            continue
        _, pos, season = f.stem.split("_")
        d = pd.read_csv(f)
        faixa = pd.cut(d["start_test"], [-1, 3, 6, 9, 60], labels=["0-3", "4-6", "7-9", "10+"])
        for nome, g in d.groupby(faixa, observed=True):
            linhas.append({
                "posicao": pos, "fold": int(season), "faixa_anos": str(nome), "n": int(len(g)),
                "s1_medio": float(g["S_cond_h1"].mean()),
                "frac_sem_risco": float((g["S_cond_h1"] >= 1 - 1e-12).mean()),
                **DIAGNOSTIC_FLAGS,
            })
    out = pd.DataFrame(linhas)
    out.to_csv(RES / "previsao_por_experiencia.csv", index=False)
    return out


def main() -> None:
    specs = {}
    fold_rows = []
    rank_rows = []
    notes = []

    for pos in POSITIONS:
        infer_list = load_inferential_list(pos)
        present, dropped_missing = filter_diagnostic_covariates(infer_list)
        panel, pool = load_panel(pos)
        df_raw = pd.read_pickle(PRED_SHARED / f"df_cox_{pos}_features.pkl")
        present = [c for c in present if c in df_raw.columns]
        specs[pos] = {
            "fonte": f"aalen_final_covs_{pos}.json",
            "regra": (
                "Lista inferencial selecionada no painel completo 2012-2024 "
                "(screening univariado p<0.05, teto eventos/10, years_exp proibido), "
                "sem dummies _missing. Uso pos-selecao, apenas diagnostico computacional."
            ),
            "covariaveis": present,
            "removidas_missing": dropped_missing,
            "n": len(present),
            "p_plus_1": len(present) + 1,
            **DIAGNOSTIC_FLAGS,
        }
        (RES / f"spec_aalen_preditivo_{pos}.json").write_text(
            json.dumps(specs[pos], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"=== {pos} spec n={len(present)} ===", flush=True)

        for season in DIAGNOSTIC_SEASONS:
            tr_raw, te_raw = walkforward_split(panel, season)
            if len(tr_raw) < 50 or len(te_raw) < 10 or int(te_raw["residual_event"].sum()) < 1:
                notes.append(f"{pos} {season}: fold residual insuficiente")
                continue
            tr, te, _use, _, _ = preprocess(tr_raw, te_raw, pool)
            te = mask_test_support(tr, te)
            if len(te) < 5 or int(te["residual_event"].sum()) < 1:
                notes.append(f"{pos} {season}: teste sem suporte")
                continue

            train_cp = evento_em_tempo_real_cp(df_raw, season)
            test_cp = df_raw[df_raw["season"] == season].copy()
            test_cp = test_cp[test_cp["player_id"].isin(te["player_id"])].drop_duplicates("player_id")
            tr_cp, te_cp, use_a, _, _ = preprocess(train_cp, test_cp, present)

            stats = design_stats(tr_cp, use_a)
            if len(stats):
                stats = stats.assign(posicao=pos, teste_em_temporada=int(season))
                rank_rows.append(stats.assign(**DIAGNOSTIC_FLAGS))
                frac = float(stats["full_rank"].mean())
            else:
                frac = np.nan

            pred, err, info = run_aalen(tr_cp, te_cp, use_a, f"{pos}_{season}")
            if pred is None:
                notes.append(f"{pos} {season}: {err}")
                print(f"  FAIL {pos} {season}: {err}", flush=True)
                continue

            te = te.merge(pred, on="player_id", how="left")
            s_cols = [c for c in te.columns if str(c).startswith("S_cond_h")]
            surv = te[s_cols].to_numpy(dtype=float)
            times = np.array([float(str(c).replace("S_cond_h", "")) for c in s_cols], dtype=float)
            degenerado = bool(np.nanmax(np.abs(surv - 1.0)) < 1e-12)
            risk = np.zeros(len(te)) if degenerado else (1.0 - np.nanmean(np.clip(surv, 0, 1), axis=1))
            mets = evaluate_fold(
                y_surv(tr, "residual_duration_model", "residual_event"),
                y_surv(te, "residual_duration_model", "residual_event"),
                risk,
                surv,
                times,
                compute_uno=False,
            )
            if degenerado:
                mets["c_index"] = np.nan
            mets.update(
                {
                    "posicao": pos,
                    "modelo": "aalen_spec_curta",
                    "teste_em_temporada": int(season),
                    "n_train_cp": int(len(tr_cp)),
                    "n_test_valid": int(len(te)),
                    "n_covs": int(len(use_a)),
                    "frac_posto_completo": frac,
                    "max_abs_B": info.get("max_abs_B"),
                    "n_train_events": info.get("n_train_events"),
                    "degenerado": int(degenerado),
                    **DIAGNOSTIC_FLAGS,
                }
            )
            fold_rows.append(mets)
            cval = mets.get("c_index", np.nan)
            ctxt = "NA" if not np.isfinite(cval) else f"{cval:.3f}"
            print(
                f"  {pos} {season}: C={ctxt} IBS={mets.get('ibs')} "
                f"max|B|={info.get('max_abs_B')} posto={frac:.2f} deg={degenerado}",
                flush=True,
            )

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(RES / "metricas_walkforward_aalen_spec_curta.csv", index=False)
    if rank_rows:
        pd.concat(rank_rows, ignore_index=True).to_csv(RES / "posto_por_fold.csv", index=False)

    summary_rows = []
    for pos in POSITIONS:
        sub = folds[folds["posicao"] == pos] if len(folds) else folds
        diagnostic = sub[sub["teste_em_temporada"].isin(DIAGNOSTIC_SEASONS)] if len(sub) else sub
        summary_rows.append(
            {
                "posicao": pos,
                "n_folds": int(len(diagnostic)),
                "n_covs": specs[pos]["n"],
                "c_index": float(diagnostic["c_index"].mean()) if len(diagnostic) and diagnostic["c_index"].notna().any() else np.nan,
                "c_index_sd": float(diagnostic["c_index"].std()) if len(diagnostic) and diagnostic["c_index"].notna().sum() > 1 else np.nan,
                "ibs": float(diagnostic["ibs"].mean()) if len(diagnostic) and diagnostic["ibs"].notna().any() else np.nan,
                "frac_posto_completo": float(diagnostic["frac_posto_completo"].mean()) if len(diagnostic) else np.nan,
                "max_abs_B_min": float(diagnostic["max_abs_B"].min()) if len(diagnostic) and diagnostic["max_abs_B"].notna().any() else np.nan,
                "n_degenerado": int(diagnostic["degenerado"].sum()) if len(diagnostic) and "degenerado" in diagnostic else 0,
                **DIAGNOSTIC_FLAGS,
            }
        )
    summary = pd.DataFrame(summary_rows)
    # Resumo do contraste; nao substitui os resultados oficiais.
    summary.to_csv(RES / "resumo_oficial_2018_2021.csv", index=False)
    previsao_por_experiencia()

    payload = {
        "regra": specs,
        "diagnostico_posselecionado_2018_2021": summary.to_dict(orient="records"),
        "notas": notes,
        **DIAGNOSTIC_FLAGS,
        "leitura": (
            "Diagnostico computacional pos-selecao: a lista inferencial foi escolhida "
            "com dados de 2012-2024, incluindo os folds 2018-2021. C e IBS nao sao "
            "avaliacao prospectiva, comparacao com o ranking principal ou backtest honesto. "
            "O ranking principal tgb_oficial, com X congelado em 2012-2017, nao e afetado."
        ),
    }
    (RES / "resumo.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42})
    pos = list(POSITIONS)
    x = np.arange(len(pos))
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    style_ax(ax)
    aalen_c = [float(summary.loc[summary["posicao"] == p, "c_index"].iloc[0]) if len(summary) else np.nan for p in pos]
    valid = np.isfinite(aalen_c)
    ax.bar(x[valid], np.asarray(aalen_c)[valid], 0.48, color="#287c8e", edgecolor="#174955", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pos)
    ax.set_ylabel("Harrell $C$")
    ax.set_ylim(0, 1)
    ax.set_title("Diagnóstico pós-seleção · não prospectivo", fontsize=10, pad=10)
    ax.axhline(0.50, color="0.35", linewidth=0.7, linestyle=":", label="Referência nula (C = 0,50)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "cindex_contraste.pdf", bbox_inches="tight")
    fig.savefig(FIG / "cindex_contraste.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(payload["diagnostico_posselecionado_2018_2021"], indent=2))
    print("NOTAS:", notes)
    shutil.rmtree(TMP, ignore_errors=True)  # arquivos de troca com o R; nao sao artefatos


if __name__ == "__main__":
    main()
