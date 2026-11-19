#!/usr/bin/env python3
"""Pacote preditivo com hiperparametros padrao: mesmo X, HPs da biblioteca, walk-forward + landmark van Houwelingen."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "Predição" / "shared"))
sys.path.insert(0, str(ROOT / "Inferência" / "shared"))

from tgb_protocol import (  # noqa: E402
    HORIZONS,
    LANDMARK_WS,
    LANDMARKS,
    OFFICIAL_SEASONS,
    PRED_SHARED,
    TEST_SEASONS,
    aalen_covs,
    candidate_times,
    common_training_horizon,
    cox_penalizer,
    evaluate_fold,
    km_survival_matrix,
    landmark_split,
    load_covs,
    load_panel,
    mask_test_support,
    media_restrita_sksurv,
    preprocess,
    walkforward_split,
    y_surv,
    evento_em_tempo_real_cp,
)

PRED = ROOT / "Predição"
AALEN_R = PRED / "aalen" / "shared" / "run_aalen_prediction.R"
PACK_DIR = PRED / "pacote_default"
RESULT_DIR = PACK_DIR / "resultados"
FIG_DIR = PACK_DIR / "figuras"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

POSITIONS = ["QB", "RB", "WR"]
TMAX_EXPECT = 12.0
NOTES = []


def log(msg: str) -> None:
    print(msg, flush=True)


def append_note(msg: str) -> None:
    NOTES.append(msg)
    log("NOTA: " + msg)


def surv_from_model_fns(model, X, times: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return np.empty((len(X), 0))
    fns = model.predict_survival_function(X, return_array=False)
    return np.vstack([np.asarray(fn(times), dtype=float) for fn in fns])


def fit_cox(train, covs, pos):
    """Cox preditivo com o penalizer PADRAO (sensibilidade de HP).

    O valor vem de `tgb_protocol.cox_penalizer`, que devolve o padrao do lifelines (0)
    nas tres posicoes (com fallback 0.1 apenas se nao convergir). O protocolo principal
    usa o penalizer escolhido pela busca aninhada em cada fold (run_oficial.py).
    """
    fit_cols = ["residual_duration_model", "residual_event"] + covs
    congelado = cox_penalizer(pos)
    tentativas = [(congelado, "padrao")]
    if congelado == 0.0:
        tentativas.append((0.1, "stability0.1"))
    last_err = None
    for pen, tag in tentativas:
        try:
            cph = CoxPHFitter(penalizer=pen)
            cph.fit(
                train[fit_cols],
                duration_col="residual_duration_model",
                event_col="residual_event",
                show_progress=False,
            )
            if tag != "padrao":
                append_note(f"Cox {pos}: penalizer={pen} ({tag}) — o padrao nao convergiu.")
            return cph, pen
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Cox nao convergiu ({pos}): {last_err}")


def run_aalen(train_cp: pd.DataFrame, test_rows: pd.DataFrame, covs: list[str], tag: str):
    work = RESULT_DIR / "aalen_tmp"
    work.mkdir(parents=True, exist_ok=True)
    train_csv = work / f"train_{tag}.csv"
    test_csv = work / f"test_{tag}.csv"
    pred_csv = work / f"pred_{tag}.csv"
    model_rds = work / f"model_{tag}.rds"
    export_cols = ["player_id", "start", "stop", "event"] + covs
    for col in covs:
        if col not in train_cp.columns or col not in test_rows.columns:
            return None, "cov_missing"
    train_cp[export_cols].to_csv(train_csv, index=False)
    test_export = test_rows.copy()
    if "stop" not in test_export.columns:
        test_export["stop"] = test_export.get("snapshot_stop", test_export["start"])
    if "event" not in test_export.columns:
        test_export["event"] = 0
    test_export[export_cols].to_csv(test_csv, index=False)
    cmd = [
        "Rscript",
        str(AALEN_R),
        str(train_csv),
        str(test_csv),
        ",".join(covs),
        str(pred_csv),
        str(model_rds),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if proc.returncode != 0 or not pred_csv.exists():
        err = (proc.stderr or proc.stdout or "")[-500:]
        return None, err
    pred = pd.read_csv(pred_csv)
    lambda_cols = [c for c in pred.columns if str(c).startswith("Lambda_cond_h")]
    pct_trunc = np.nan
    if lambda_cols:
        arr = pred[lambda_cols].to_numpy(dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if len(arr):
            pct_trunc = float(100.0 * np.mean(arr < 0))
    return pred, pct_trunc


def metrics_from_sksurv(train, test, time_col, event_col, risk, surv, times, uno=True):
    ytr = y_surv(train, time_col, event_col)
    yte = y_surv(test, time_col, event_col)
    return evaluate_fold(ytr, yte, risk, surv, times, compute_uno=uno)


def expected_from_sf(model, X, tmax: float = TMAX_EXPECT, exige_dominio: bool = True):
    fns = model.predict_survival_function(X, return_array=False)
    return media_restrita_sksurv(fns, tmax, exige_dominio=exige_dominio)


def walkforward_python(pos: str, modelo: str):
    panel, covs = load_panel(pos)
    df_raw = pd.read_pickle(PRED_SHARED / f"df_cox_{pos}_features.pkl")
    rows = []
    for season in TEST_SEASONS:
        tr_raw, te_raw = walkforward_split(panel, season)
        if len(tr_raw) < 50 or len(te_raw) < 10 or int(te_raw["residual_event"].sum()) < 1:
            continue
        tr, te, use, _, _ = preprocess(tr_raw, te_raw, covs)
        te = mask_test_support(tr, te)
        if len(te) < 5 or int(te["residual_event"].sum()) < 1:
            continue
        times = candidate_times(tr["residual_duration_model"].to_numpy(), te["residual_duration_model"].to_numpy())
        Xtr, Xte = tr[use].to_numpy(dtype=float), te[use].to_numpy(dtype=float)
        ytr = Surv.from_arrays(tr["residual_event"].astype(bool), tr["residual_duration_model"].astype(float))
        note_pen = np.nan
        pct_trunc = np.nan
        uno = True
        try:
            if modelo == "cox":
                cph, pen = fit_cox(tr, use, pos)
                note_pen = pen
                risk = cph.predict_partial_hazard(te[use]).to_numpy().reshape(-1)
                surv = (
                    cph.predict_survival_function(te[use], times=times).T.to_numpy()
                    if len(times)
                    else np.empty((len(te), 0))
                )
            elif modelo == "gbsa":
                model = GradientBoostingSurvivalAnalysis(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    subsample=1.0,
                    random_state=42,
                )
                model.fit(Xtr, ytr)
                risk = model.predict(Xte)
                surv = surv_from_model_fns(model, Xte, times)
            elif modelo == "rsf":
                model = RandomSurvivalForest(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=6,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=1,
                )
                model.fit(Xtr, ytr)
                risk = model.predict(Xte)
                surv = surv_from_model_fns(model, Xte, times)
            elif modelo == "km":
                surv, risk = km_survival_matrix(
                    tr, len(te), times, "residual_duration_model", "residual_event"
                )
                uno = False
            elif modelo == "aalen":
                train_cp = evento_em_tempo_real_cp(df_raw, season)
                test_cp = df_raw[df_raw["season"] == season].copy()
                test_cp = test_cp[test_cp["player_id"].isin(te["player_id"])].drop_duplicates("player_id")
                a_covs = aalen_covs(covs)
                tr_cp, te_cp, use_a, _, _ = preprocess(train_cp, test_cp, a_covs)
                pred, info = run_aalen(tr_cp, te_cp, use_a, f"{pos}_{season}")
                if pred is None:
                    append_note(f"Aalen WF {pos} {season} falhou: {info}")
                    continue
                pct_trunc = info if isinstance(info, float) else np.nan
                te = te.merge(pred, on="player_id", how="left")
                s_cols = [c for c in te.columns if str(c).startswith("S_cond_h")]
                if not s_cols:
                    append_note(f"Aalen WF {pos} {season}: sem colunas S_cond")
                    continue
                surv = te[s_cols].to_numpy(dtype=float)
                times = np.array([float(str(c).replace("S_cond_h", "")) for c in s_cols], dtype=float)
                if np.nanmax(np.abs(surv - 1.0)) < 1e-12:
                    append_note(f"Aalen WF {pos} {season}: S_cond≡1 (B(t) degenerado). C não entra no ranking.")
                    risk = np.zeros(len(te))
                    uno = False
                    mets = metrics_from_sksurv(
                        tr, te, "residual_duration_model", "residual_event", risk, surv, times, uno=uno
                    )
                    mets["c_index"] = np.nan
                    mets["mean_time_auc"] = np.nan
                    mets.update(
                        {
                            "posicao": pos,
                            "modelo": modelo,
                            "protocolo": "walkforward",
                            "teste_em_temporada": int(season),
                            "n_train": int(len(tr_raw)),
                            "n_test": int(len(te_raw)),
                            "n_test_valid": int(len(te)),
                            "n_covs": int(len(use_a)),
                            "cox_penalizer": note_pen,
                            "pct_trunc": pct_trunc,
                            "aalen_degenerado": 1,
                        }
                    )
                    rows.append(mets)
                    log(f"  {modelo} {pos} {season}: DEGENERADO S≡1 n={len(te)}")
                    continue
                risk = 1.0 - np.nanmean(np.clip(surv, 0, 1), axis=1)
                uno = False
            else:
                raise ValueError(modelo)
        except Exception:
            append_note(f"{modelo} WF {pos} {season} erro:\n{traceback.format_exc()[-400:]}")
            continue
        mets = metrics_from_sksurv(
            tr, te, "residual_duration_model", "residual_event", risk, surv, times, uno=uno
        )
        mets.update(
            {
                "posicao": pos,
                "modelo": modelo,
                "protocolo": "walkforward",
                "teste_em_temporada": int(season),
                "n_train": int(len(tr_raw)),
                "n_test": int(len(te_raw)),
                "n_test_valid": int(len(te)),
                "n_covs": int(len(use)),
                "cox_penalizer": note_pen,
                "pct_trunc": pct_trunc,
            }
        )
        rows.append(mets)
        cval = mets.get("c_index", np.nan)
        ctxt = "nan" if not np.isfinite(cval) else f"{cval:.3f}"
        log(f"  {modelo} {pos} {season}: C={ctxt} n={mets['n_test_valid']}")
    return rows


def landmark_python(pos: str, modelo: str):
    panel, covs = load_panel(pos)
    df_raw = pd.read_pickle(PRED_SHARED / f"df_cox_{pos}_features.pkl")
    rows = []
    for t_l in LANDMARKS:
        for w in LANDMARK_WS:
            tr_raw, te_raw = landmark_split(panel, t_l, float(w))
            if len(tr_raw) < 50 or len(te_raw) < 10 or int(te_raw["lm_event"].sum()) < 1:
                continue
            # preprocess usa residual cols só para impute; outcome landmark é lm_*
            tr, te, use, _, _ = preprocess(tr_raw, te_raw, covs)
            # NÃO filtrar lm_time < w: isso apaga os sobreviventes do horizonte e zera o C.
            if len(te) < 5 or int(te["lm_event"].sum()) < 1:
                continue
            times = candidate_times(tr["lm_time"].to_numpy(), te["lm_time"].to_numpy(), horizons=np.array([float(w)]))
            # também avalia 1..w se w>1
            extra = np.array([h for h in HORIZONS if h <= w], dtype=float)
            times = candidate_times(tr["lm_time"].to_numpy(), te["lm_time"].to_numpy(), horizons=extra)
            Xtr, Xte = tr[use].to_numpy(dtype=float), te[use].to_numpy(dtype=float)
            ytr = Surv.from_arrays(tr["lm_event"].astype(bool), tr["lm_time"].astype(float))
            uno = True
            pct_trunc = np.nan
            try:
                if modelo == "cox":
                    fit_cols = ["lm_time", "lm_event"] + use
                    cph = None
                    for pen in (0.0, 0.1):
                        try:
                            cph = CoxPHFitter(penalizer=pen)
                            cph.fit(tr[fit_cols], duration_col="lm_time", event_col="lm_event", show_progress=False)
                            break
                        except Exception:
                            cph = None
                    if cph is None:
                        continue
                    risk = cph.predict_partial_hazard(te[use]).to_numpy().reshape(-1)
                    surv = (
                        cph.predict_survival_function(te[use], times=times).T.to_numpy()
                        if len(times)
                        else np.empty((len(te), 0))
                    )
                elif modelo == "gbsa":
                    model = GradientBoostingSurvivalAnalysis(
                        n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=42
                    )
                    model.fit(Xtr, ytr)
                    risk = model.predict(Xte)
                    surv = surv_from_model_fns(model, Xte, times)
                elif modelo == "rsf":
                    if pos == "QB":
                        continue
                    model = RandomSurvivalForest(
                        n_estimators=100,
                        max_depth=None,
                        min_samples_split=6,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=1,
                    )
                    model.fit(Xtr, ytr)
                    risk = model.predict(Xte)
                    surv = surv_from_model_fns(model, Xte, times)
                elif modelo == "km":
                    surv, risk = km_survival_matrix(tr, len(te), times, "lm_time", "lm_event")
                    uno = False
                elif modelo == "aalen":
                    train_cp = evento_em_tempo_real_cp(df_raw, t_l)
                    test_cp = df_raw[df_raw["season"] == t_l].copy()
                    test_cp = test_cp[test_cp["player_id"].isin(te["player_id"])].drop_duplicates("player_id")
                    tr_cp, te_cp, use_a, _, _ = preprocess(train_cp, test_cp, aalen_covs(covs))
                    pred, info = run_aalen(tr_cp, te_cp, use_a, f"lm_{pos}_{t_l}_w{w}")
                    if pred is None:
                        append_note(f"Aalen LM {pos} {t_l} w={w} falhou: {info}")
                        continue
                    pct_trunc = info if isinstance(info, float) else np.nan
                    merged = te.merge(pred, on="player_id", how="left")
                    col = f"S_cond_h{int(w)}"
                    if col not in merged.columns:
                        append_note(f"Aalen LM {pos} {t_l} w={w}: sem {col}")
                        continue
                    s_w = merged[col].to_numpy(dtype=float)
                    surv = s_w.reshape(-1, 1)
                    times = np.array([float(w)])
                    risk = 1.0 - np.clip(s_w, 0, 1)
                    uno = False
                    te = merged
                else:
                    raise ValueError(modelo)
            except Exception:
                append_note(f"{modelo} LM {pos} tL={t_l} w={w} erro:\n{traceback.format_exc()[-400:]}")
                continue
            mets = metrics_from_sksurv(tr, te, "lm_time", "lm_event", risk, surv, times, uno=uno)
            mets.update(
                {
                    "posicao": pos,
                    "modelo": modelo,
                    "protocolo": "landmark",
                    "landmark": int(t_l),
                    "horizon_w": int(w),
                    "n_train": int(len(tr_raw)),
                    "n_test": int(len(te_raw)),
                    "n_test_valid": int(len(te)),
                    "n_covs": int(len(use)),
                    "pct_trunc": pct_trunc,
                }
            )
            rows.append(mets)
            cval = mets.get("c_index", np.nan)
            ctxt = "nan" if not np.isfinite(cval) else f"{cval:.3f}"
            log(f"  {modelo} LM {pos} tL={t_l} w={w}: C={ctxt}")
    return rows


def predict_2024_gbsa():
    out_rows = []
    prepared = {}
    train_followup = {}
    for pos in POSITIONS:
        panel, covs = load_panel(pos)
        # Rótulos do treino usam a origem 2024; retratos de 2024 ficam reservados ao teste.
        tr_raw, te_raw = walkforward_split(panel, 2024)
        prepared[pos] = (covs, tr_raw, te_raw)
        train_followup[pos] = tr_raw["residual_duration_model"]
    tau_effective, max_train_by_position = common_training_horizon(train_followup, TMAX_EXPECT)
    for pos in POSITIONS:
        covs, tr_raw, te_raw = prepared[pos]
        tr, te, use, _, _ = preprocess(tr_raw, te_raw, covs)
        ytr = Surv.from_arrays(tr["residual_event"].astype(bool), tr["residual_duration_model"].astype(float))
        model = GradientBoostingSurvivalAnalysis(
            n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=42
        )
        model.fit(tr[use].to_numpy(dtype=float), ytr)
        expected, median = expected_from_sf(
            model, te[use].to_numpy(dtype=float), tmax=tau_effective, exige_dominio=True
        )
        pred = te[["player_id", "player_name", "snapshot_season"]].copy()
        pred["posicao"] = pos
        pred["expected_years"] = expected
        pred["median_years"] = median
        pred["tau_requested"] = TMAX_EXPECT
        pred["tau_effective"] = tau_effective
        pred["tau_efetivo"] = tau_effective
        pred["maxT_train"] = max_train_by_position[pos]
        pred = pred.sort_values("expected_years", ascending=False).reset_index(drop=True)
        pred["rank"] = np.arange(1, len(pred) + 1)
        pred.to_csv(RESULT_DIR / f"predicao_gbsa_2024_{pos}.csv", index=False)
        out_rows.append(pred)
        log(f"  GBSA 2024 {pos}: {len(pred)} ativos")
    return pd.concat(out_rows, ignore_index=True)


def summarize(df: pd.DataFrame, by=("modelo", "posicao")) -> pd.DataFrame:
    num = [c for c in df.columns if c not in {"modelo", "posicao", "protocolo"} and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(list(by), as_index=False)[num].agg(["mean", "std", "count"])
    g.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in g.columns]
    return g.reset_index() if "index" in g.columns else g


def barplot(df, y, fname, title):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df.empty or y not in df.columns:
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="posicao", y=y, hue="modelo")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=140)
    plt.close()


def mirror_comparacao(wf: pd.DataFrame) -> None:
    """Grava o recorte oficial (2018-2021) e o landmark por w do pacote default."""

    # Recorte oficial do ranking (2018-2021), derivado das metricas de fold.
    if not wf.empty and "teste_em_temporada" in wf.columns:
        sub = wf[wf["teste_em_temporada"].isin(OFFICIAL_SEASONS)]
        if not sub.empty:
            metricas = [
                c
                for c in ["c_index", "mean_time_auc", "ibs", "brier_1a", "brier_2a", "brier_3a", "pct_trunc"]
                if c in sub.columns
            ]
            comp = sub.groupby(["modelo", "posicao"], as_index=False)[metricas].mean()
            sd = (
                sub.groupby(["modelo", "posicao"], as_index=False)[["c_index"]]
                .std()
                .rename(columns={"c_index": "c_index_sd"})
            )
            n = (
                sub.groupby(["modelo", "posicao"], as_index=False)[["c_index"]]
                .count()
                .rename(columns={"c_index": "n_folds_c"})
            )
            comp = comp.merge(sd, on=["modelo", "posicao"], how="left").merge(
                n, on=["modelo", "posicao"], how="left"
            )
            if "uno_c_index" not in comp.columns:
                comp["uno_c_index"] = np.nan
            ordem = [
                "modelo",
                "posicao",
                "c_index",
                "uno_c_index",
                "mean_time_auc",
                "ibs",
                "brier_1a",
                "brier_2a",
                "brier_3a",
                "pct_trunc",
                "c_index_sd",
                "n_folds_c",
            ]
            comp = comp[[c for c in ordem if c in comp.columns]]
            comp = comp.sort_values(["modelo", "posicao"])
            # Recorte oficial do ranking (2018-2021) do pacote default.
            comp.to_csv(RESULT_DIR / "comparacao_walkforward_2018_2021.csv", index=False)

    # Landmark por w: mesma tabela de comparacao_landmarking.
    src = RESULT_DIR / "comparacao_landmarking.csv"
    if src.exists():
        (RESULT_DIR / "comparacao_landmarking_por_w.csv").write_bytes(src.read_bytes())



def main():
    all_wf, all_lm = [], []
    # Aalen fora da comparacao preditiva (decisao do autor); o estudo de diagnostico de
    # posto (pipeline/45) continua. As funcoes de Aalen acima nao sao chamadas aqui.
    models = ["km", "cox", "gbsa", "rsf"]
    for modelo in models:
        log(f"\n===== WALK-FORWARD {modelo} =====")
        for pos in POSITIONS:
            all_wf.extend(walkforward_python(pos, modelo))
        log(f"\n===== LANDMARK {modelo} =====")
        for pos in POSITIONS:
            all_lm.extend(landmark_python(pos, modelo))

    wf = pd.DataFrame(all_wf)
    lm = pd.DataFrame(all_lm)
    wf.to_csv(RESULT_DIR / "metricas_walkforward_folds.csv", index=False)
    lm.to_csv(RESULT_DIR / "metricas_landmark_folds.csv", index=False)

    if not wf.empty:
        mean_cols = [c for c in ["c_index", "uno_c_index", "mean_time_auc", "ibs", "brier_1a", "brier_2a", "brier_3a", "pct_trunc"] if c in wf.columns]
        comp = wf.groupby(["modelo", "posicao"], as_index=False)[mean_cols].mean()
        std = wf.groupby(["modelo", "posicao"], as_index=False)[["c_index"]].std().rename(columns={"c_index": "c_index_sd"})
        comp = comp.merge(std, on=["modelo", "posicao"], how="left")
        comp.to_csv(RESULT_DIR / "comparacao_completa.csv", index=False)
        barplot(comp, "c_index", "comparacao_cindex.png", "C-index walk-forward (HPs default, mesmo X)")
        if "mean_time_auc" in comp.columns:
            barplot(comp, "mean_time_auc", "comparacao_auc.png", "AUC walk-forward (HPs default, mesmo X)")
        if "ibs" in comp.columns:
            barplot(comp, "ibs", "comparacao_ibs.png", "IBS walk-forward (HPs default, mesmo X)")

    if not lm.empty:
        mean_cols = [c for c in ["c_index", "uno_c_index", "mean_time_auc", "ibs"] if c in lm.columns]
        comp_lm = lm.groupby(["modelo", "posicao"], as_index=False)[mean_cols].mean()
        comp_lm.to_csv(RESULT_DIR / "comparacao_landmarking.csv", index=False)

    log("\n===== GBSA 2024 =====")
    predict_2024_gbsa()

    (RESULT_DIR / "notas_execucao.json").write_text(
        json.dumps({"notas": NOTES, "hps": {
            "gbsa": "n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=42",
            "rsf": "n_estimators=100, max_depth=None, min_samples_split=6, min_samples_leaf=3, random_state=42",
            "cox": (
                "penalizer padrao (tgb_protocol.cox_penalizer, sensibilidade): "
                f"QB={cox_penalizer('QB')}, RB={cox_penalizer('RB')}, WR={cox_penalizer('WR')}"
            ),
            "km": "KaplanMeier sem covariáveis",
        }}, indent=2, ensure_ascii=False)
        + "\n"
    )
    mirror_comparacao(wf)
    log("\nPacote default concluído em " + str(PACK_DIR))


if __name__ == "__main__":
    main()
