#!/usr/bin/env python3
"""Furos do storytelling: C com/sem years_exp, folga do campeão, X dos nomes."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.model_selection import GroupKFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "Predição" / "shared"))
sys.path.insert(0, str(ROOT / "Inferência" / "shared"))

from tgb_protocol import (  # noqa: E402
    candidate_times,
    carrega_hps_aninhados,
    hp_aninhado,
    rsf_permitido,
    evaluate_fold,
    load_covs,
    load_panel,
    mask_test_support,
    preprocess,
    walkforward_split,
    y_surv,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "resultados"
FIG = HERE / "figuras"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

OFFICIAL = (2018, 2019, 2020, 2021)
POSITIONS = ("QB", "RB", "WR")
SEED = 42
# T1: rede elástica no MESMO X congelado do pacote oficial (covs_preditivas_{POS}.json),
# nos mesmos folds 2018-2021. O Cox oficial NÃO é reajustado aqui: os números vêm do
# resumo do pacote oficial. O refiltro de colinearidade no painel inteiro, que gerou os
# O ElasticNet é ajustado aqui dentro (CoxnetSurvivalAnalysis).
OFICIAL_RESUMO = (
    ROOT / "documentos" / "estudo_modelos_finais" / "resultados" / "resumo_walkforward_2018_2021.csv"
)
EN_L1_RATIO = 0.1
EN_ALPHAS = 10.0 ** np.linspace(-3.0, 0.5, 15)
# Meio ponto de C (0,005): abaixo disso dois modelos não são distinguíveis em quatro
# folds, e o C é citado na escrita com três decimais.
EN_LIMIAR_VEREDITO = 0.005
# HPs do protocolo principal: busca aninhada por fold (hps_por_fold.csv), mesmo fold de cada ajuste.
HPS = carrega_hps_aninhados()
NAMES = {
    "QB": ["Josh Allen", "Sam Darnold", "C.J. Stroud", "Patrick Mahomes", "Trey Lance", "Lamar Jackson"],
    "RB": ["Christian McCaffrey", "Derrick Henry", "Jahmyr Gibbs", "Ezekiel Elliott", "Bijan Robinson"],
    "WR": ["Justin Jefferson", "Ja'Marr Chase", "Zay Flowers", "Adam Thielen"],
}
FACE_COLS = {
    "QB": ["years_exp", "passer_rating", "team_pass_attempts_share", "depth_team", "offense_pct"],
    "RB": ["years_exp", "games_missed", "offense_pct", "team_rushing_yards_share", "draft_ovr"],
    "WR": ["years_exp", "games_missed", "offense_pct", "team_targets_share", "draft_ovr"],
}


def spec_covs(covs: list[str], spec: str) -> list[str]:
    if spec == "so_years_exp":
        return [c for c in covs if c == "years_exp"]
    if spec == "sem_years_exp":
        return [c for c in covs if c != "years_exp"]
    if spec == "pool":
        return list(covs)
    raise ValueError(spec)


def fit_risk(modelo: str, pos: str, origem: int, tr: pd.DataFrame, te: pd.DataFrame, use: list[str], times: np.ndarray):
    if modelo == "km":
        kmf = KaplanMeierFitter()
        kmf.fit(tr["residual_duration_model"], event_observed=tr["residual_event"])
        risk = np.zeros(len(te))
        if len(times) == 0:
            surv = np.empty((len(te), 0))
        else:
            s = np.array([float(kmf.predict(t)) for t in times])
            surv = np.tile(s, (len(te), 1))
        return risk, surv
    ytr = Surv.from_arrays(tr["residual_event"].astype(bool), tr["residual_duration_model"].astype(float))
    Xtr = tr[use].to_numpy(dtype=float)
    Xte = te[use].to_numpy(dtype=float)
    hp = hp_aninhado(HPS, pos, modelo, origem) if modelo != "km" else {}
    if modelo == "cox":
        # Mesmo contrato do pacote oficial: recuo para 0.1 so quando o lambda pedido e zero.
        pen = float(hp["penalizer"])
        ultimo = None
        for p in ((pen, 0.1) if pen == 0.0 else (pen,)):
            try:
                cph = CoxPHFitter(penalizer=float(p))
                cph.fit(
                    tr[["residual_duration_model", "residual_event"] + use],
                    duration_col="residual_duration_model",
                    event_col="residual_event",
                    show_progress=False,
                )
                ultimo = None
                break
            except Exception as exc:  # noqa: BLE001
                ultimo = exc
        if ultimo is not None:
            raise RuntimeError(ultimo)
        risk = cph.predict_partial_hazard(te[use]).to_numpy().reshape(-1)
        surv = (
            cph.predict_survival_function(te[use], times=times).T.to_numpy()
            if len(times)
            else np.empty((len(te), 0))
        )
        return risk, surv
    if modelo == "gbsa":
        model = GradientBoostingSurvivalAnalysis(random_state=SEED, **hp)
        model.fit(Xtr, ytr)
        fns = model.predict_survival_function(Xte, return_array=False) if len(times) else []
        surv = (
            np.vstack([np.asarray(fn(times), dtype=float) for fn in fns])
            if len(times)
            else np.empty((len(te), 0))
        )
        return model.predict(Xte), surv
    if modelo == "rsf":
        model = RandomSurvivalForest(random_state=SEED, n_jobs=1, **hp)
        model.fit(Xtr, ytr)
        fns = model.predict_survival_function(Xte, return_array=False) if len(times) else []
        surv = (
            np.vstack([np.asarray(fn(times), dtype=float) for fn in fns])
            if len(times)
            else np.empty((len(te), 0))
        )
        return model.predict(Xte), surv
    raise ValueError(modelo)


def score(tr, te, risk, surv, times):
    mets = evaluate_fold(
        y_surv(tr, "residual_duration_model", "residual_event"),
        y_surv(te, "residual_duration_model", "residual_event"),
        risk,
        surv,
        times,
        compute_uno=False,
    )
    cval = mets.get("c_index", np.nan)
    ibs = mets.get("ibs", np.nan)
    if not np.isfinite(cval):
        try:
            yte = y_surv(te, "residual_duration_model", "residual_event")
            cval = float(concordance_index_censored(yte["event"], yte["time"], risk)[0])
        except Exception:
            cval = np.nan
    return float(cval) if np.isfinite(cval) else np.nan, float(ibs) if np.isfinite(ibs) else np.nan


def fit_elasticnet(tr: pd.DataFrame, te: pd.DataFrame, use: list[str], times: np.ndarray):
    """Rede elástica: alfa por CV interna no treino."""
    ytr = Surv.from_arrays(tr["residual_event"].astype(bool), tr["residual_duration_model"].astype(float))
    Xtr = tr[use].to_numpy(dtype=float)
    Xte = te[use].to_numpy(dtype=float)
    n_splits = min(3, max(2, len(tr) // 30))
    # Dobras por jogador: o mesmo atleta nao pode estar no ajuste e na validacao do alfa.
    kf = GroupKFold(n_splits=n_splits)
    alpha_scores = np.zeros(len(EN_ALPHAS))
    for train_idx, val_idx in kf.split(Xtr, groups=tr["player_id"].to_numpy()):
        for j, alpha in enumerate(EN_ALPHAS):
            m = CoxnetSurvivalAnalysis(l1_ratio=EN_L1_RATIO, alphas=[float(alpha)], max_iter=10000)
            m.fit(Xtr[train_idx], ytr[train_idx])
            risk_cv = m.predict(Xtr[val_idx])
            alpha_scores[j] += concordance_index_censored(
                ytr[val_idx]["event"], ytr[val_idx]["time"], risk_cv
            )[0]
    alpha_scores /= n_splits
    best_alpha = float(EN_ALPHAS[int(np.argmax(alpha_scores))])
    model = CoxnetSurvivalAnalysis(
        l1_ratio=EN_L1_RATIO, alphas=[best_alpha], max_iter=10000, fit_baseline_model=True
    )
    model.fit(Xtr, ytr)
    risk = model.predict(Xte)
    if len(times):
        fns = model.predict_survival_function(Xte, return_array=False)
        surv = np.vstack([np.asarray(fn(times), dtype=float) for fn in fns])
    else:
        surv = np.empty((len(te), 0))
    return risk, surv, best_alpha


def elasticnet_folds() -> pd.DataFrame:
    """Rede elástica no X congelado, folds oficiais 2018-2021."""
    rows = []
    for pos in POSITIONS:
        panel, covs = load_panel(pos)
        print(f"=== EN {pos} ===", flush=True)
        for season in OFFICIAL:
            tr_raw, te_raw = walkforward_split(panel, season)
            if len(tr_raw) < 50 or len(te_raw) < 10:
                continue
            tr, te, use, _, _ = preprocess(tr_raw, te_raw, covs)
            te = mask_test_support(tr, te)
            if len(te) < 5 or int(te["residual_event"].sum()) < 1:
                continue
            times = candidate_times(
                tr["residual_duration_model"].to_numpy(), te["residual_duration_model"].to_numpy()
            )
            try:
                risk, surv, alpha = fit_elasticnet(tr, te, use, times)
                cval, ibs = score(tr, te, risk, surv, times)
            except Exception as exc:
                print(f"  FAIL elasticnet {season}: {exc}", flush=True)
                continue
            rows.append(
                {
                    "posicao": pos,
                    "modelo": "elasticnet",
                    "teste_em_temporada": season,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "n_covs": int(len(use)),
                    "alpha": alpha,
                    "c_index": cval,
                    "ibs": ibs,
                }
            )
            print(f"  elasticnet {season}: C={cval:.3f} alpha={alpha:.4f}", flush=True)
    return pd.DataFrame(rows)


def veredito_elasticnet(delta: float) -> str:
    """Veredito curto a partir do delta de C médio contra o Cox oficial."""
    if not np.isfinite(delta):
        return "indeterminado"
    if delta > EN_LIMIAR_VEREDITO:
        return "melhora"
    if delta < -EN_LIMIAR_VEREDITO:
        return "nao_melhora"
    return "empata"


def elasticnet_vs_cox() -> pd.DataFrame:
    """Rede elástica (X congelado) contra o Cox oficial, sem reajustar o Cox."""
    folds = elasticnet_folds()
    resumo = pd.read_csv(OFICIAL_RESUMO)
    cox = resumo[resumo["modelo"] == "cox"].set_index("posicao")
    rows = []
    for pos in POSITIONS:
        if pos not in cox.index:
            print(f"  {pos}: Cox oficial ausente em {OFICIAL_RESUMO.name}", flush=True)
            continue
        c_cox = float(cox.loc[pos, "c_index"])
        sub = folds[folds["posicao"] == pos] if not folds.empty else folds
        c_en = float(sub["c_index"].mean()) if len(sub) else np.nan
        delta = c_en - c_cox if np.isfinite(c_en) else np.nan
        rows.append(
            {
                "posicao": pos,
                "modelo": "elasticnet",
                "x_fonte": f"Predição/shared/covs_preditivas_{pos}.json",
                # o pré-processamento por fold pode descartar covariável degenerada
                "n_covs": int(sub["n_covs"].max()) if len(sub) else np.nan,
                "n_folds": int(len(sub)),
                "c_index_medio": c_en,
                "c_index_sd": float(sub["c_index"].std()) if len(sub) > 1 else np.nan,
                "ibs_medio": float(sub["ibs"].mean()) if len(sub) and sub["ibs"].notna().any() else np.nan,
                "delta_vs_cox_oficial": delta,
                "veredito": veredito_elasticnet(delta),
            }
        )
        rows.append(
            {
                "posicao": pos,
                "modelo": "cox_oficial",
                "x_fonte": f"Predição/shared/covs_preditivas_{pos}.json",
                "n_covs": int(len(load_covs(pos))),
                "n_folds": int(cox.loc[pos, "n_folds"]),
                "c_index_medio": c_cox,
                "c_index_sd": float(cox.loc[pos, "c_index_sd"]),
                "ibs_medio": float(cox.loc[pos, "ibs"]),
                "delta_vs_cox_oficial": 0.0,
                "veredito": veredito_elasticnet(0.0),
            }
        )
    out = pd.DataFrame(
        rows,
        columns=[
            "posicao",
            "modelo",
            "x_fonte",
            "n_covs",
            "n_folds",
            "c_index_medio",
            "c_index_sd",
            "ibs_medio",
            "delta_vs_cox_oficial",
            "veredito",
        ],
    )
    out.to_csv(RES / "elasticnet_vs_cox.csv", index=False)
    return out, folds


def run_ablation():
    rows = []
    for pos in POSITIONS:
        panel, covs = load_panel(pos)
        print(f"=== {pos} ===", flush=True)
        for season in OFFICIAL:
            tr_raw, te_raw = walkforward_split(panel, season)
            if len(tr_raw) < 50 or len(te_raw) < 10:
                continue
            for spec in ("so_years_exp", "sem_years_exp", "pool"):
                use0 = spec_covs(covs, spec)
                if not use0:
                    continue
                tr, te, use, _, _ = preprocess(tr_raw, te_raw, use0)
                te = mask_test_support(tr, te)
                if len(te) < 5 or int(te["residual_event"].sum()) < 1:
                    continue
                times = candidate_times(
                    tr["residual_duration_model"].to_numpy(), te["residual_duration_model"].to_numpy()
                )
                modelos = ["km", "cox", "gbsa"] + (["rsf"] if rsf_permitido(pos) else [])
                for modelo in modelos:
                    if modelo == "km" and spec != "pool":
                        continue
                    try:
                        risk, surv = fit_risk(modelo, pos, season, tr, te, use if modelo != "km" else [], times)
                        cval, ibs = score(tr, te, risk, surv, times)
                    except Exception as exc:
                        print(f"  FAIL {modelo} {spec} {season}: {exc}", flush=True)
                        continue
                    rows.append(
                        {
                            "posicao": pos,
                            "modelo": modelo,
                            "spec": spec,
                            "teste_em_temporada": season,
                            "n_covs": 0 if modelo == "km" else int(len(use)),
                            "c_index": cval,
                            "ibs": ibs,
                        }
                    )
                    print(f"  {modelo} {spec} {season}: C={cval:.3f}", flush=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(RES / "ablacao_folds.csv", index=False)
    summ = (
        folds.groupby(["posicao", "modelo", "spec"], as_index=False)
        .agg(n_folds=("c_index", "count"), c_index=("c_index", "mean"), c_sd=("c_index", "std"), ibs=("ibs", "mean"))
        .sort_values(["posicao", "modelo", "spec"])
    )
    summ.to_csv(RES / "ablacao_resumo.csv", index=False)
    return folds, summ


def champion_gap(folds: pd.DataFrame):
    pool = folds[(folds.spec == "pool") & (folds.modelo != "km")].copy()
    rows = []
    for pos in POSITIONS:
        wide = pool[pool.posicao == pos].pivot(index="teste_em_temporada", columns="modelo", values="c_index")
        if "gbsa" in wide.columns and "cox" in wide.columns:
            d = wide["gbsa"] - wide["cox"]
            rows.append(
                {
                    "posicao": pos,
                    "par": "gbsa-cox",
                    "delta_medio": float(d.mean()),
                    "delta_sd": float(d.std()),
                    "se": float(d.std() / np.sqrt(len(d))),
                    "folds_gbsa_ganha": int((d > 0).sum()),
                    "n_folds": int(len(d)),
                }
            )
        if "gbsa" in wide.columns and "rsf" in wide.columns:
            d = wide["gbsa"] - wide["rsf"]
            rows.append(
                {
                    "posicao": pos,
                    "par": "gbsa-rsf",
                    "delta_medio": float(d.mean()),
                    "delta_sd": float(d.std()),
                    "se": float(d.std() / np.sqrt(len(d))),
                    "folds_gbsa_ganha": int((d > 0).sum()),
                    "n_folds": int(len(d)),
                }
            )
        if "rsf" in wide.columns and "cox" in wide.columns:
            d = wide["rsf"] - wide["cox"]
            rows.append(
                {
                    "posicao": pos,
                    "par": "rsf-cox",
                    "delta_medio": float(d.mean()),
                    "delta_sd": float(d.std()),
                    "se": float(d.std() / np.sqrt(len(d))),
                    "folds_gbsa_ganha": int((d > 0).sum()),
                    "n_folds": int(len(d)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RES / "folga_campeao.csv", index=False)
    return out


def face_table():
    rows = []
    for pos in POSITIONS:
        panel, covs = load_panel(pos)
        te = panel[panel["snapshot_season"] == 2024].copy()
        cols = [c for c in FACE_COLS[pos] if c in te.columns]
        for nome in NAMES[pos]:
            hit = te[te["player_name"] == nome]
            if hit.empty:
                continue
            row = {"posicao": pos, "player_name": nome}
            for c in cols:
                row[c] = float(pd.to_numeric(hit.iloc[0][c], errors="coerce"))
            if "residual_duration_model" in hit.columns:
                row["residual_duration"] = float(hit.iloc[0]["residual_duration_model"])
            rows.append(row)
        # position medians among 2024 actives
        med = {"posicao": pos, "player_name": "(mediana 2024)"}
        for c in cols:
            med[c] = float(pd.to_numeric(te[c], errors="coerce").median())
        rows.append(med)
    face = pd.DataFrame(rows)
    face.to_csv(RES / "face_validity_2024.csv", index=False)
    return face


def make_fig(summ: pd.DataFrame):
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.6), sharey=True)
    order = [("so_years_exp", "só exp."), ("sem_years_exp", "sem exp."), ("pool", "pool")]
    for ax, pos in zip(axes, POSITIONS):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.45, color="0.65")
        mods = ["cox", "gbsa"] + (["rsf"] if rsf_permitido(pos) else [])
        x = np.arange(len(order))
        width = 0.22
        for i, m in enumerate(mods):
            vals = []
            for spec, _ in order:
                hit = summ[(summ.posicao == pos) & (summ.modelo == m) & (summ.spec == spec)]
                vals.append(float(hit["c_index"].iloc[0]) if len(hit) else np.nan)
            ax.bar(
                x + (i - (len(mods) - 1) / 2) * width,
                vals,
                width=width * 0.95,
                color={0: "0.75", 1: "0.40", 2: "0.15"}[i],
                edgecolor="black",
                linewidth=0.45,
                label=m.upper(),
            )
        km = summ[(summ.posicao == pos) & (summ.modelo == "km")]
        if len(km):
            ax.axhline(float(km["c_index"].iloc[0]), color="0.4", lw=0.6, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in order])
        ax.set_title(pos)
        ax.set_ylim(0.48, 0.90)
    axes[0].set_ylabel("Harrell $C$")
    axes[1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(FIG / "c_com_sem_exp.pdf", bbox_inches="tight")
    fig.savefig(FIG / "c_com_sem_exp.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    folds, summ = run_ablation()
    gap = champion_gap(folds)
    en, en_folds = elasticnet_vs_cox()
    face = face_table()
    make_fig(summ)
    payload = {
        "ablacao": summ.to_dict(orient="records"),
        "folga": gap.to_dict(orient="records"),
        "elasticnet_vs_cox": en.to_dict(orient="records"),
        # detalhe por fold da rede elástica no X congelado (o CSV guarda só as médias)
        "elasticnet_folds": en_folds.to_dict(orient="records") if not en_folds.empty else [],
        "face": face.to_dict(orient="records"),
    }
    (RES / "resumo.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(summ.to_string(index=False))
    print(gap.to_string(index=False))
    print(en.to_string(index=False))
    print(face.to_string(index=False))


if __name__ == "__main__":
    main()
