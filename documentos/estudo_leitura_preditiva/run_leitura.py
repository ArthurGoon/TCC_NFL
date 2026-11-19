#!/usr/bin/env python3
"""Importância comparável (permutação do C), listas 2024 e retrospecto WF/landmark."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import spearmanr
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "Predição" / "shared"))
sys.path.insert(0, str(ROOT / "Inferência" / "shared"))

from tgb_protocol import (  # noqa: E402
    common_training_horizon,
    media_restrita_cox,
    media_restrita_sksurv,
    carrega_hps_aninhados,
    hp_aninhado,
    rsf_permitido,
    landmark_split,
    load_panel,
    mask_test_support,
    preprocess,
    walkforward_split,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "resultados"
FIG = HERE / "figuras"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

OFFICIAL = (2018, 2019, 2020, 2021)
POSITIONS = ("QB", "RB", "WR")
SEED = 42
N_REPEATS = 5

# HPs do protocolo principal: busca aninhada por fold (hps_por_fold.csv). Sem valores padrao fixos.
HPS = carrega_hps_aninhados()

LABELS = {
    "years_exp": "anos de experiência",
    "passer_rating": "passer rating",
    "team_pass_attempts_share": "share de tentativas",
    "depth_team": "profundidade no elenco",
    "offense_pct": "snaps ofensivos",
    "games_missed": "jogos perdidos",
    "team_rushing_yards_share": "share de jardas terrestres",
    "team_targets_share": "share de alvos",
    "draft_ovr": "posição no draft",
    "draft_round": "rodada do draft",
    "weight": "peso",
    "height": "altura",
    "rookie_flag": "flag de calouro",
    "undrafted_flag": "não draftado",
    "pre_window_veteran": "veterano pré-janela",
    "unobs_career_years": "anos sem estatística antes da 1ª linha",
}


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.45, color="0.65")


def expected_from_fns(fns, tmax=12.0, n_grid=49, exige_dominio=True):
    return media_restrita_sksurv(fns, tmax, exige_dominio=exige_dominio)


def tau_efetivo_fns(fns, tmax=12.0):
    supports = [float(fn.x[-1]) for fn in fns]
    return min(float(tmax), min(supports)) if supports else float(tmax)


TMAX_EXPECT = 12.0


def expected_cox(cph, X, tmax=TMAX_EXPECT, n_grid=49):
    return media_restrita_cox(cph, X, tmax)


def fit_model(modelo, tr, use, pos, origem):
    """Ajusta `modelo` com o HP da busca aninhada do fold de origem `origem`.

    Mesmo contrato do pacote oficial (run_oficial.fit_cox/fit_predict): o Cox recua para
    lambda 0.1 so quando o lambda pedido e zero e nao converge; GBSA e RSF usam os
    hiperparametros escolhidos naquele fold. Para a lista 2024 a origem e 2024; para o
    retrospecto do landmark de 2023, a origem e o fold walk-forward de 2023 (LANDMARK_HP_ORIGEM).
    """
    ytr = Surv.from_arrays(tr["residual_event"].astype(bool), tr["residual_duration_model"].astype(float))
    Xtr = tr[use].to_numpy(dtype=float)
    hp = hp_aninhado(HPS, pos, modelo, origem)
    if modelo == "cox":
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
                return cph
            except Exception as exc:  # noqa: BLE001
                ultimo = exc
        raise RuntimeError(ultimo)
    if modelo == "gbsa":
        model = GradientBoostingSurvivalAnalysis(random_state=SEED, **hp)
        model.fit(Xtr, ytr)
        return model
    if modelo == "rsf":
        model = RandomSurvivalForest(random_state=SEED, n_jobs=1, **hp)
        model.fit(Xtr, ytr)
        return model
    raise ValueError(modelo)


def risk_of(modelo, model, frame, use):
    if modelo == "cox":
        return model.predict_partial_hazard(frame[use]).to_numpy().reshape(-1)
    return model.predict(frame[use].to_numpy(dtype=float))


def perm_c_drop(modelo, model, te, use, n_repeats=N_REPEATS, seed=SEED):
    y_event = te["residual_event"].astype(bool).to_numpy()
    y_time = te["residual_duration_model"].astype(float).to_numpy()
    base_risk = risk_of(modelo, model, te, use)
    base = float(concordance_index_censored(y_event, y_time, base_risk)[0])
    rng = np.random.default_rng(seed)
    X = te[use].to_numpy(dtype=float)
    rows = []
    for j, col in enumerate(use):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            te_p = te.copy()
            te_p[col] = Xp[:, j]
            rp = risk_of(modelo, model, te_p, use)
            cp = float(concordance_index_censored(y_event, y_time, rp)[0])
            drops.append(base - cp)
        rows.append({"feature": col, "perm_drop": float(np.mean(drops)), "perm_drop_sd": float(np.std(drops))})
    return base, pd.DataFrame(rows)


def native_importance(modelo, model, use):
    if modelo == "cox":
        vals = model.params_.reindex(use).abs().to_numpy(dtype=float)
        return dict(zip(use, vals))
    if modelo == "gbsa":
        vals = np.asarray(model.feature_importances_, dtype=float)
        return dict(zip(use, vals))
    return {}


def models_for(pos):
    mods = ["cox", "gbsa"]
    if rsf_permitido(pos):
        mods.append("rsf")
    return mods


def run_importance():
    fold_rows = []
    native_rows = []
    for pos in POSITIONS:
        panel, covs = load_panel(pos)
        print(f"=== importância {pos} ===", flush=True)
        for season in OFFICIAL:
            tr_raw, te_raw = walkforward_split(panel, season)
            tr, te, use, _, _ = preprocess(tr_raw, te_raw, covs)
            te = mask_test_support(tr, te)
            for modelo in models_for(pos):
                model = fit_model(modelo, tr, use, pos, season)
                base, imp = perm_c_drop(modelo, model, te, use)
                imp["posicao"] = pos
                imp["modelo"] = modelo
                imp["teste_em_temporada"] = season
                imp["c_base"] = base
                fold_rows.append(imp)
                nat = native_importance(modelo, model, use)
                for feat, val in nat.items():
                    native_rows.append(
                        {
                            "posicao": pos,
                            "modelo": modelo,
                            "teste_em_temporada": season,
                            "feature": feat,
                            "native": val,
                        }
                    )
                print(f"  {modelo} {season} C={base:.3f}", flush=True)
    folds = pd.concat(fold_rows, ignore_index=True)
    folds.to_csv(RES / "perm_importance_folds.csv", index=False)
    pd.DataFrame(native_rows).to_csv(RES / "native_importance_folds.csv", index=False)
    mean_imp = (
        folds.groupby(["posicao", "modelo", "feature"], as_index=False)["perm_drop"]
        .mean()
        .sort_values(["posicao", "modelo", "perm_drop"], ascending=[True, True, False])
    )
    mean_imp.to_csv(RES / "perm_importance_media.csv", index=False)
    return mean_imp


def concordance_importance(mean_imp: pd.DataFrame):
    rows = []
    overlap_rows = []
    for pos in POSITIONS:
        mods = models_for(pos)
        wide = mean_imp[mean_imp.posicao == pos].pivot(index="feature", columns="modelo", values="perm_drop")
        for a, b in [(i, j) for i in mods for j in mods if i < j]:
            if a not in wide.columns or b not in wide.columns:
                continue
            sub = wide[[a, b]].dropna()
            rho, pval = spearmanr(sub[a], sub[b])
            rows.append({"posicao": pos, "par": f"{a}-{b}", "spearman": float(rho), "p": float(pval), "n": int(len(sub))})
            top_a = set(sub[a].nlargest(8).index)
            top_b = set(sub[b].nlargest(8).index)
            overlap_rows.append(
                {
                    "posicao": pos,
                    "par": f"{a}-{b}",
                    "top8_intersecao": len(top_a & top_b),
                    "jaccard": len(top_a & top_b) / len(top_a | top_b),
                    "comuns": ", ".join(sorted(top_a & top_b)),
                }
            )
    conc = pd.DataFrame(rows)
    ov = pd.DataFrame(overlap_rows)
    conc.to_csv(RES / "concordancia_importance.csv", index=False)
    ov.to_csv(RES / "overlap_top8.csv", index=False)
    return conc, ov


def predict_2024():
    rows = []
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
        print(f"=== 2024 {pos} n={len(te)} ===", flush=True)
        block = te[["player_id", "player_name", "snapshot_season"]].copy()
        block["posicao"] = pos
        if "years_exp" in te.columns:
            block["years_exp"] = te["years_exp"].to_numpy()
        for modelo in models_for(pos):
            model = fit_model(modelo, tr, use, pos, 2024)
            if modelo == "cox":
                exp, med = expected_cox(model, te[use], tau_effective)
                risk = model.predict_partial_hazard(te[use]).to_numpy().reshape(-1)
            else:
                Xte = te[use].to_numpy(dtype=float)
                fns = model.predict_survival_function(Xte, return_array=False)
                exp, med = expected_from_fns(fns, tau_effective, exige_dominio=True)
                risk = model.predict(Xte)
            tmp = block.copy()
            tmp["modelo"] = modelo
            tmp["expected_years"] = exp
            tmp["median_years"] = med
            tmp["tau_requested"] = TMAX_EXPECT
            tmp["tau_effective"] = tau_effective
            tmp["tau_efetivo"] = tau_effective
            tmp["maxT_train"] = max_train_by_position[pos]
            tmp["risk"] = risk
            rows.append(tmp)
    pred = pd.concat(rows, ignore_index=True)
    pred["rank"] = pred.groupby(["posicao", "modelo"])["expected_years"].rank(ascending=False, method="first")
    pred.to_csv(RES / "predicao_2024_modelos.csv", index=False)
    corr_rows = []
    for pos in POSITIONS:
        wide = pred[pred.posicao == pos].pivot(index="player_id", columns="modelo", values="expected_years")
        mods = [c for c in models_for(pos) if c in wide.columns]
        for a, b in [(i, j) for i in mods for j in mods if i < j]:
            sub = wide[[a, b]].dropna()
            rho, pval = spearmanr(sub[a], sub[b])
            corr_rows.append({"posicao": pos, "par": f"{a}-{b}", "spearman": float(rho), "p": float(pval), "n": int(len(sub))})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(RES / "concordancia_predicao_2024.csv", index=False)
    return pred, corr


def retrospect_wf_2021():
    rows = []
    for pos in POSITIONS:
        panel, covs = load_panel(pos)
        tr_raw, te_raw = walkforward_split(panel, 2021)
        tr, te, use, _, _ = preprocess(tr_raw, te_raw, covs)
        te = mask_test_support(tr, te)
        model = fit_model("gbsa", tr, use, pos, 2021)
        Xte = te[use].to_numpy(dtype=float)
        fns = model.predict_survival_function(Xte, return_array=False)
        tau_efetivo = tau_efetivo_fns(fns, tmax=3.0)
        exp, med = expected_from_fns(fns, tmax=3.0, exige_dominio=False)
        risk = model.predict(Xte)
        out = te[["player_id", "player_name", "snapshot_season", "residual_duration_model", "residual_event"]].copy()
        if "years_exp" in te.columns:
            out["years_exp"] = te["years_exp"].to_numpy()
        out["posicao"] = pos
        out["expected_years"] = exp
        out["median_years"] = med
        out["tau_efetivo"] = tau_efetivo
        out["risk"] = risk
        out["rank"] = out["expected_years"].rank(ascending=False, method="first")
        rows.append(out)
        print(f"=== WF 2021 {pos} n={len(out)} eventos={int(out.residual_event.sum())} ===", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(RES / "retrospecto_wf_2021_gbsa.csv", index=False)
    return df


def retrospect_landmark_2023():
    rows = []
    for pos in POSITIONS:
        panel, covs = load_panel(pos)
        tr_raw, te_raw = landmark_split(panel, 2023, 1.0)
        tr, te, use, _, _ = preprocess(tr_raw, te_raw, covs)
        ytr = Surv.from_arrays(tr["lm_event"].astype(bool), tr["lm_time"].astype(float))
        # Landmark com origem t_L = 2023: HP do fold walk-forward de mesma temporada (LANDMARK_HP_ORIGEM).
        model = GradientBoostingSurvivalAnalysis(random_state=SEED, **hp_aninhado(HPS, pos, "gbsa", 2023))
        model.fit(tr[use].to_numpy(dtype=float), ytr)
        Xte = te[use].to_numpy(dtype=float)
        fns = model.predict_survival_function(Xte, return_array=False)
        s1 = np.array([float(np.asarray(fn(1.0)).reshape(-1)[0]) for fn in fns])
        risk = model.predict(Xte)
        out = te[["player_id", "player_name", "snapshot_season", "lm_time", "lm_event"]].copy()
        if "years_exp" in te.columns:
            out["years_exp"] = te["years_exp"].to_numpy()
        out["posicao"] = pos
        out["surv_1a"] = s1
        out["risk"] = risk
        out["rank"] = out["surv_1a"].rank(ascending=False, method="first")
        rows.append(out)
        print(f"=== LM 2023 w=1 {pos} n={len(out)} saidas={int(out.lm_event.sum())} ===", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(RES / "retrospecto_landmark_2023_w1_gbsa.csv", index=False)
    return df


def make_figures(mean_imp: pd.DataFrame, pred: pd.DataFrame):
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 4.2))
    for ax, pos in zip(axes, POSITIONS):
        style_ax(ax)
        sub = mean_imp[mean_imp.posicao == pos]
        top = sub.groupby("feature")["perm_drop"].mean().nlargest(8).index.tolist()
        plot = sub[sub.feature.isin(top)].copy()
        feats = list(reversed(top))
        y = np.arange(len(feats))
        mods = models_for(pos)
        width = 0.22
        for i, m in enumerate(mods):
            vals = []
            for f in feats:
                hit = plot[(plot.feature == f) & (plot.modelo == m)]
                vals.append(float(hit["perm_drop"].iloc[0]) if len(hit) else 0.0)
            offset = (i - (len(mods) - 1) / 2) * width
            color = {0: "0.75", 1: "0.35", 2: "0.10"}[i]
            ax.barh(y + offset, vals, height=width * 0.95, color=color, edgecolor="black", linewidth=0.4, label=m.upper())
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS.get(f, f.replace("_", " ")[:22]) for f in feats], fontsize=7)
        ax.set_title(pos)
        if pos == "QB":
            ax.set_xlabel(r"queda do $C$")
    axes[1].legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "perm_importance_top.pdf", bbox_inches="tight")
    fig.savefig(FIG / "perm_importance_top.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4))
    for ax, pos in zip(axes, POSITIONS):
        style_ax(ax)
        wide = pred[pred.posicao == pos].pivot(index="player_id", columns="modelo", values="expected_years")
        if "cox" in wide.columns and "gbsa" in wide.columns:
            ax.scatter(wide["cox"], wide["gbsa"], s=14, c="0.25", edgecolors="none", alpha=0.75)
            lo = min(wide["cox"].min(), wide["gbsa"].min())
            hi = max(wide["cox"].max(), wide["gbsa"].max())
            ax.plot([lo, hi], [lo, hi], color="0.55", lw=0.7, ls=":")
        ax.set_title(pos)
        ax.set_xlabel("Cox, anos esperados")
        if pos == "QB":
            ax.set_ylabel("GBSA, anos esperados")
    fig.tight_layout()
    fig.savefig(FIG / "pred_2024_cox_vs_gbsa.pdf", bbox_inches="tight")
    fig.savefig(FIG / "pred_2024_cox_vs_gbsa.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def contagem_texto(serie: pd.Series) -> dict:
    """Contagens com chave textual: agrupamentos multiplos geram tuplas, que o JSON nao aceita."""
    return {
        "|".join(k if isinstance(k, tuple) else (k,)): int(v)
        for k, v in serie.to_dict().items()
    }


def main():
    mean_imp = run_importance()
    conc, ov = concordance_importance(mean_imp)
    pred, corr = predict_2024()
    wf = retrospect_wf_2021()
    lm = retrospect_landmark_2023()
    make_figures(mean_imp, pred)
    payload = {
        "concordancia_importance": conc.to_dict(orient="records"),
        "overlap_top8": ov.to_dict(orient="records"),
        "concordancia_predicao_2024": corr.to_dict(orient="records"),
        "n_2024": contagem_texto(pred.groupby(["posicao", "modelo"]).size()),
        "wf_2021_n": contagem_texto(wf.groupby("posicao").size()),
        "lm_2023_saidas": contagem_texto(lm.groupby("posicao")["lm_event"].sum()),
    }
    (RES / "resumo.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(conc.to_string(index=False))
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
