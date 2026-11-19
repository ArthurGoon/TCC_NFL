#!/usr/bin/env python3
"""Pacote preditivo oficial: walk-forward, landmark, listas 2024 e forma das predições.

Hiperparâmetros: por padrão (`--hp nested`), cada fold usa o HP escolhido pela busca
aninhada daquele mesmo fold (estudo_tune_preditivo/resultados/hps_por_fold.csv). Landmark
com origem t_L usa o HP do fold walk-forward de mesma temporada t_L (sem busca por horizonte).
Lista 2024 usa o HP da busca de 2024. `--hp default` reproduz o pacote com os valores
padrão (sensibilidade), gravando em resultados/sensibilidade_hp_default/.
Walk-forward (oficial 2018-2021; 2022-2023 no apêndice), landmark e leitura da forma das
predições entre Cox, GBSA e RSF (QB, RB e WR). Não grava figura da tese no modo default.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import gaussian_kde, skew, spearmanr
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "Predição" / "shared"))
sys.path.insert(0, str(ROOT / "Inferência" / "shared"))

from tgb_protocol import (  # noqa: E402
    HORIZONS,
    LANDMARK_WS,
    LANDMARKS,
    OFFICIAL_SEASONS,
    TEST_SEASONS,
    candidate_times,
    carrega_hps_aninhados,
    common_training_horizon,
    hp_aninhado,
    hp_padrao,
    media_restrita_cox,
    media_restrita_sksurv,
    evaluate_fold,
    km_survival_matrix,
    landmark_completo,
    landmark_split,
    load_panel,
    mask_test_support,
    preprocess,
    rsf_permitido,
    restricted_time_identifiable,
    restricted_time_target,
    walkforward_split,
    y_surv,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "resultados"
FIG = HERE / "figuras"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

POSITIONS = ("QB", "RB", "WR")
SEED = 42
# OMP_NUM_THREADS=1 no topo; o RSF é o único modelo do pacote que aceita n_jobs.
N_JOBS = 1
TMAX_EXPECT = 12.0
# Ultima temporada do painel: fim da janela de observacao do teste.
FIM_JANELA = 2024
# Modo de hiperparâmetros ("nested" = protocolo principal; "default" = sensibilidade).
MODO_HP = "nested"
HPS = None  # tabela da busca aninhada (só em modo nested)


def versao_pacote(nome: str) -> str:
    """Versão instalada da distribuição; 'ausente' se não estiver no ambiente."""
    try:
        return version(nome)
    except PackageNotFoundError:
        return "ausente"


def models_for(pos: str) -> list[str]:
    mods = ["km", "cox", "gbsa"]
    if rsf_permitido(pos):
        mods.append("rsf")
    return mods


def hp_do_fold(pos: str, modelo: str, origem: int) -> dict:
    """HP usado no fold de origem `origem`. Sem fallback: falta de linha levanta KeyError."""
    if MODO_HP == "default":
        return hp_padrao(modelo)
    global HPS
    if HPS is None:  # importado como modulo (estudos de sensibilidade): mesma tabela do oficial
        HPS = carrega_hps_aninhados(ROOT / "documentos" / "estudo_tune_preditivo" / "resultados" / "hps_por_fold.csv")
    return hp_aninhado(HPS, pos, modelo, origem)


def pred_models_for(pos: str) -> list[str]:
    return [m for m in models_for(pos) if m != "km"]


def surv_from_fns(model, X, times: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return np.empty((len(X), 0))
    fns = model.predict_survival_function(X, return_array=False)
    return np.vstack([np.asarray(fn(times), dtype=float) for fn in fns])


def expected_sksurv(model, X, tmax: float = TMAX_EXPECT, n_grid: int = 49, exige_dominio: bool = True):
    fns = model.predict_survival_function(X, return_array=False)
    return media_restrita_sksurv(fns, tmax, exige_dominio=exige_dominio)


def expected_cox(cph, X, tmax: float = TMAX_EXPECT, n_grid: int = 49):
    return media_restrita_cox(cph, X, tmax)


def fit_cox(tr, use, time_col, event_col, pen: float):
    fit_cols = [time_col, event_col] + use
    last = None
    for p in (pen, 0.1) if pen == 0.0 else (pen,):
        try:
            cph = CoxPHFitter(penalizer=float(p))
            cph.fit(tr[fit_cols], duration_col=time_col, event_col=event_col, show_progress=False)
            return cph, float(p)
        except Exception as exc:
            last = exc
    raise RuntimeError(last)


def fit_predict(modelo: str, hp: dict, tr, te, use, time_col: str, event_col: str, times: np.ndarray):
    """Ajusta `modelo` com o HP do fold (`hp`) e devolve risco, sobrevivência e extras."""
    Xtr = tr[use].to_numpy(dtype=float)
    Xte = te[use].to_numpy(dtype=float)
    ytr = Surv.from_arrays(tr[event_col].astype(bool), tr[time_col].astype(float))
    if modelo == "km":
        surv, risk = km_survival_matrix(tr, len(te), times, time_col, event_col)
        return risk, surv, {"penalizer": np.nan}
    if modelo == "cox":
        cph, pen = fit_cox(tr, use, time_col, event_col, float(hp["penalizer"]))
        risk = cph.predict_partial_hazard(te[use]).to_numpy().reshape(-1)
        surv = (
            cph.predict_survival_function(te[use], times=times).T.to_numpy()
            if len(times)
            else np.empty((len(te), 0))
        )
        # hp = o pedido pela busca; `penalizer` = o efetivamente ajustado (recuo para 0.1 se não convergir).
        return risk, surv, {"penalizer": pen, "model": cph, "hp": {"penalizer": float(hp["penalizer"])}}
    if modelo == "gbsa":
        model = GradientBoostingSurvivalAnalysis(random_state=SEED, **hp)
        model.fit(Xtr, ytr)
        return model.predict(Xte), surv_from_fns(model, Xte, times), {"model": model, "hp": dict(hp)}
    if modelo == "rsf":
        model = RandomSurvivalForest(random_state=SEED, n_jobs=N_JOBS, **hp)
        model.fit(Xtr, ytr)
        return model.predict(Xte), surv_from_fns(model, Xte, times), {"model": model, "hp": dict(hp)}
    raise ValueError(modelo)


def score(tr, te, time_col, event_col, risk, surv, times, uno: bool):
    return evaluate_fold(
        y_surv(tr, time_col, event_col),
        y_surv(te, time_col, event_col),
        risk,
        surv,
        times,
        compute_uno=uno,
    )


def _coleta_matriz_sobrevivencia(surv_rows, grid_map, pos, season, modelo, tr, te, times, surv, risk) -> None:
    """Guarda o insumo do bootstrap do passo `ic_preditivo`.

    A matriz de sobrevivencia por individuo, a grade de tempos do fold e as duas
    avaliacoes do Kaplan-Meier de censura que o `sksurv` usa como IPCW: no tempo
    observado de cada sujeito e em cada ponto da grade. Sem elas o bootstrap nao
    reconstroi IBS e AUC media identicos aos publicados. A grade e gravada uma
    vez por (posicao, fold); a matriz, uma vez por modelo.
    """
    # Mesma distribuicao de censura do `evaluate_fold`: a do proprio teste.
    cens = CensoringDistributionEstimator().fit(
        y_surv(te, "residual_duration_model", "residual_event")
    )
    t_teste = te["residual_duration_model"].to_numpy(dtype=float)
    g_grade = cens.predict_proba(times)
    g_sujeito = cens.predict_proba(t_teste)
    chave = (pos, int(season))
    if chave not in grid_map:
        grid_map[chave] = [
            {
                "posicao": pos,
                "teste_em_temporada": int(season),
                "j": int(j),
                "t": float(t),
                "g_t": float(g_grade[j]),
            }
            for j, t in enumerate(times)
        ]
    base = {
        "posicao": pos,
        "teste_em_temporada": int(season),
        "modelo": modelo,
        "player_id": te["player_id"].to_numpy(),
        "tempo": t_teste,
        "evento": te["residual_event"].to_numpy(dtype=int),
        "g_subject": g_sujeito,
        "risk": risk,
    }
    for j in range(surv.shape[1]):
        base[f"s{j}"] = surv[:, j]
    surv_rows.append(pd.DataFrame(base))


def walkforward() -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    pred_rows = []
    # `mask_test_support` descarta do teste observacoes com residual maior ou
    # igual ao maximo do treino. O efeito precisa de artefato proprio: sem ele a
    # decisao metodologica afirma "zero linhas removidas" sem produtor do numero.
    mask_rows: list[dict] = []
    # Insumo do bootstrap (`ic_preditivo`): sobrevivencia por individuo por fold.
    surv_rows: list[pd.DataFrame] = []
    grid_map: dict[tuple[str, int], list[dict]] = {}
    for pos in POSITIONS:
        panel, covs = load_panel(pos)
        print(f"=== WF {pos} ===", flush=True)
        for season in TEST_SEASONS:
            tr_raw, te_raw = walkforward_split(panel, season)
            if len(tr_raw) < 50 or len(te_raw) < 10 or int(te_raw["residual_event"].sum()) < 1:
                continue
            tr, te, use, _, _ = preprocess(tr_raw, te_raw, covs)
            n_antes = int(len(te))
            te = mask_test_support(tr, te)
            mask_rows.append(
                {
                    "posicao": pos,
                    "temporada": int(season),
                    "n_teste_antes": n_antes,
                    "n_teste_depois": int(len(te)),
                    "n_removidas": n_antes - int(len(te)),
                }
            )
            if len(te) < 5 or int(te["residual_event"].sum()) < 1:
                continue
            times = candidate_times(
                tr["residual_duration_model"].to_numpy(), te["residual_duration_model"].to_numpy()
            )
            for modelo in models_for(pos):
                hp = {} if modelo == "km" else hp_do_fold(pos, modelo, season)
                try:
                    risk, surv, extra = fit_predict(
                        modelo, hp, tr, te, use, "residual_duration_model", "residual_event", times
                    )
                except Exception:
                    print(f"  FAIL {modelo} {season}\n{traceback.format_exc()[-300:]}", flush=True)
                    continue
                mets = score(
                    tr, te, "residual_duration_model", "residual_event", risk, surv, times, uno=(modelo != "km")
                )
                mets.update(
                    {
                        "posicao": pos,
                        "modelo": modelo,
                        "protocolo": "walkforward",
                        "teste_em_temporada": int(season),
                        "oficial": int(season in OFFICIAL_SEASONS),
                        "n_train": int(len(tr)),
                        "n_test": int(len(te)),
                        "n_covs": int(len(use)),
                        "cox_penalizer": extra.get("penalizer", np.nan),
                        "hp_modo": MODO_HP,
                        "hp_json": json.dumps(extra["hp"], default=str) if extra.get("hp") else np.nan,
                    }
                )
                fold_rows.append(mets)
                cval = mets.get("c_index", np.nan)
                ctxt = "nan" if not np.isfinite(cval) else f"{cval:.3f}"
                print(f"  {modelo} {season} C={ctxt}", flush=True)
                if modelo == "km" or int(season) not in OFFICIAL_SEASONS:
                    continue
                _coleta_matriz_sobrevivencia(
                    surv_rows, grid_map, pos, season, modelo, tr, te, times, surv, risk
                )
                model = extra.get("model")
                if model is None:
                    continue
                # A predição RMST é limitada a tau; o alvo individual só é pontuável
                # se o evento ocorreu ou se o follow-up alcançou tau.
                tau = float(FIM_JANELA - int(season))
                if modelo == "cox":
                    exp, med = expected_cox(model, te[use], tau)
                else:
                    exp, med = expected_sksurv(model, te[use].to_numpy(dtype=float), tmax=tau, exige_dominio=True)
                block = te[["player_id", "player_name", "snapshot_season", "residual_duration_model", "residual_event"]].copy()
                followup = block["residual_duration_model"].to_numpy(dtype=float)
                event = block["residual_event"].to_numpy(dtype=int)
                identifiable = restricted_time_identifiable(event, followup, tau)
                block["error_identifiable"] = identifiable
                block["error_target_years"] = restricted_time_target(followup, tau, identifiable)
                if "years_exp" in te.columns:
                    block["years_exp"] = te["years_exp"].to_numpy()
                block["posicao"] = pos
                block["modelo"] = modelo
                block["expected_years"] = exp
                block["tau"] = tau
                block["median_years"] = med
                block["risk"] = risk
                pred_rows.append(block)
    folds = pd.DataFrame(fold_rows)
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    mask = pd.DataFrame(
        mask_rows, columns=["posicao", "temporada", "n_teste_antes", "n_teste_depois", "n_removidas"]
    )
    destino_mask = RES / "auditoria" / "mask_support.csv"
    destino_mask.parent.mkdir(parents=True, exist_ok=True)
    mask.to_csv(destino_mask, index=False)
    if surv_rows:
        pd.concat(surv_rows, ignore_index=True).to_csv(RES / "oof_surv_matrix.csv", index=False)
        grade = pd.DataFrame([r for rows in grid_map.values() for r in rows])
        grade.to_csv(RES / "oof_times_grid.csv", index=False)
        print(
            f"matriz de sobrevivencia OOF: {len(surv_rows)} blocos (pos x fold x modelo),"
            f" {len(grade)} pontos de grade em {len(grid_map)} folds",
            flush=True,
        )
    print(
        f"mask_test_support: {int(mask['n_removidas'].sum())} linhas removidas nos folds oficiais"
        f" ({len(mask)} folds)",
        flush=True,
    )
    return folds, preds


def landmark() -> pd.DataFrame:
    rows = []
    for pos in POSITIONS:
        panel, covs = load_panel(pos)
        print(f"=== LM {pos} ===", flush=True)
        for t_l in LANDMARKS:
            for w in LANDMARK_WS:
                tr_raw, te_raw = landmark_split(panel, t_l, float(w))
                if len(tr_raw) < 50 or len(te_raw) < 10 or int(te_raw["lm_event"].sum()) < 1:
                    continue
                tr, te, use, _, _ = preprocess(tr_raw, te_raw, covs)
                if len(te) < 5 or int(te["lm_event"].sum()) < 1:
                    continue
                extra_h = np.array([h for h in HORIZONS if h <= w], dtype=float)
                times = candidate_times(tr["lm_time"].to_numpy(), te["lm_time"].to_numpy(), horizons=extra_h)
                for modelo in models_for(pos):
                    # Landmark com origem t_L: HP do fold walk-forward de mesma temporada t_L.
                    hp = {} if modelo == "km" else hp_do_fold(pos, modelo, t_l)
                    try:
                        risk, surv, extra = fit_predict(modelo, hp, tr, te, use, "lm_time", "lm_event", times)
                    except Exception:
                        print(f"  FAIL {modelo} tL={t_l} w={w}", flush=True)
                        continue
                    mets = score(tr, te, "lm_time", "lm_event", risk, surv, times, uno=(modelo != "km"))
                    mets.update(
                        {
                            "posicao": pos,
                            "modelo": modelo,
                            "protocolo": "landmark",
                            "landmark": int(t_l),
                            "horizon_w": int(w),
                            # acompanhamento completo ate t_L + w + 1 (entra nas medias)
                            "completa": int(landmark_completo(t_l, w)),
                            "n_train": int(len(tr)),
                            "n_test": int(len(te)),
                            "n_covs": int(len(use)),
                            "cox_penalizer": extra.get("penalizer", np.nan),
                            "hp_modo": MODO_HP,
                            "hp_json": json.dumps(extra["hp"], default=str) if extra.get("hp") else np.nan,
                        }
                    )
                    rows.append(mets)
                    cval = mets.get("c_index", np.nan)
                    ctxt = "nan" if not np.isfinite(cval) else f"{cval:.3f}"
                    print(f"  {modelo} tL={t_l} w={w} C={ctxt}", flush=True)
    return pd.DataFrame(rows)


def predict_2024() -> pd.DataFrame:
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
        base = te[["player_id", "player_name", "snapshot_season"]].copy()
        base["posicao"] = pos
        if "years_exp" in te.columns:
            base["years_exp"] = te["years_exp"].to_numpy()
        dummy_times = np.array([], dtype=float)
        for modelo in pred_models_for(pos):
            hp = hp_do_fold(pos, modelo, 2024)
            risk, _, extra = fit_predict(
                modelo, hp, tr, te, use, "residual_duration_model", "residual_event", dummy_times
            )
            model = extra["model"]
            if modelo == "cox":
                exp, med = expected_cox(model, te[use], tau_effective)
            else:
                exp, med = expected_sksurv(
                    model, te[use].to_numpy(dtype=float), tmax=tau_effective, exige_dominio=True
                )
            tmp = base.copy()
            tmp["modelo"] = modelo
            tmp["expected_years"] = exp
            tmp["median_years"] = med
            tmp["tau_requested"] = TMAX_EXPECT
            tmp["tau_effective"] = tau_effective
            tmp["tau_efetivo"] = tau_effective
            tmp["maxT_train"] = max_train_by_position[pos]
            tmp["risk"] = risk
            # Lambda efetivamente usado na linha: o penalizer do Cox e, nas árvores,
            # os hiperparâmetros em JSON (no Cox o lambda já é a coluna cox_penalizer).
            tmp["cox_penalizer"] = extra.get("penalizer", np.nan)
            tmp["hp_json"] = json.dumps(extra["hp"], default=str) if extra.get("hp") else np.nan
            tmp["hp_modo"] = MODO_HP
            rows.append(tmp)
    pred = pd.concat(rows, ignore_index=True)
    pred["rank"] = pred.groupby(["posicao", "modelo"])["expected_years"].rank(ascending=False, method="first")
    # As duas colunas adicionais entram no fim da tabela.
    pred = pred[
        [c for c in pred.columns if c not in ("cox_penalizer", "hp_json")] + ["cox_penalizer", "hp_json"]
    ]
    return pred


def kde_peaks(vals: np.ndarray, grid: np.ndarray, bw: float = 0.28) -> tuple[int, list[float]]:
    vals = vals[np.isfinite(vals)]
    if len(vals) < 8:
        return 0, []
    y = gaussian_kde(vals, bw_method=bw)(grid)
    loc = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > 0.08 * y.max()))[0]
    peaks = [float(grid[i + 1]) for i in loc]
    return len(peaks), peaks


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.45, color="0.65")


def analyze(pred_2024: pd.DataFrame, pred_oof: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42})
    labs = {"cox": "Cox", "gbsa": "GBSA", "rsf": "RSF"}
    ls = {"cox": "-", "gbsa": "--", "rsf": ":"}
    col = {"cox": "0.55", "gbsa": "0.15", "rsf": "0.00"}

    shape_rows = []
    spear_rows = []
    for src, df in (("2024", pred_2024), ("oof_2018_2021", pred_oof)):
        if df is None or df.empty:
            continue
        for pos in POSITIONS:
            mods = pred_models_for(pos)
            sub = df[df.posicao == pos]
            for m in mods:
                v = sub.loc[sub.modelo == m, "expected_years"].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                grid = np.linspace(max(0.1, float(np.min(v)) - 0.3), float(np.max(v)) + 0.3, 240)
                npeak, peaks = kde_peaks(v, grid)
                shape_rows.append(
                    {
                        "fonte": src,
                        "posicao": pos,
                        "modelo": m,
                        "n": int(len(v)),
                        "media": float(np.mean(v)),
                        "mediana": float(np.median(v)),
                        "p25": float(np.quantile(v, 0.25)),
                        "p75": float(np.quantile(v, 0.75)),
                        "iqr": float(np.quantile(v, 0.75) - np.quantile(v, 0.25)),
                        "dp": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
                        "assimetria": float(skew(v, bias=False)) if len(v) > 2 else np.nan,
                        "n_picos_kde": int(npeak),
                        "picos": ",".join(f"{p:.2f}" for p in peaks),
                    }
                )
            idx = ["player_id", "snapshot_season"] if "snapshot_season" in sub.columns and src != "2024" else ["player_id"]
            wide = sub.pivot_table(index=idx, columns="modelo", values="expected_years")
            for i, a in enumerate(mods):
                for b in mods[i + 1 :]:
                    if a not in wide.columns or b not in wide.columns:
                        continue
                    pair = wide[[a, b]].dropna()
                    rho, pval = spearmanr(pair[a], pair[b])
                    spear_rows.append(
                        {
                            "fonte": src,
                            "posicao": pos,
                            "par": f"{a}-{b}",
                            "spearman": float(rho),
                            "p": float(pval),
                            "n": int(len(pair)),
                        }
                    )
    pd.DataFrame(shape_rows).to_csv(RES / "forma_predicoes.csv", index=False)
    pd.DataFrame(spear_rows).to_csv(RES / "concordancia_esperados.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.15), sharey=True)
    for ax, pos in zip(axes, POSITIONS):
        style_ax(ax)
        sub = pred_2024[pred_2024.posicao == pos]
        xmax = 11.2
        grid = np.linspace(0.2, xmax, 220)
        for m in pred_models_for(pos):
            vals = sub.loc[sub.modelo == m, "expected_years"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            kde = gaussian_kde(vals, bw_method=0.28)
            ax.plot(grid, kde(grid), color=col[m], linestyle=ls[m], linewidth=1.7, label=labs[m])
            ax.axvline(np.median(vals), color=col[m], linestyle=ls[m], linewidth=0.7, alpha=0.7)
        ax.set_title(pos)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("anos esperados")
        if pos == "QB":
            ax.set_ylabel("densidade")
        if pos == "WR":
            ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "dist_anos_2024.pdf", bbox_inches="tight")
    fig.savefig(FIG / "dist_anos_2024.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.85))
    for ax, pos in zip(axes, POSITIONS):
        mods = pred_models_for(pos)
        labels = [labs[m] for m in mods]
        wide = (
            pred_2024[pred_2024.posicao == pos]
            .pivot_table(index="player_id", columns="modelo", values="expected_years")
            .dropna()
        )
        n = len(mods)
        mat = np.eye(n)
        for i, a in enumerate(mods):
            for j, b in enumerate(mods):
                if i != j:
                    mat[i, j] = float(spearmanr(wide[a], wide[b]).statistic)
        ax.imshow(mat, cmap="gray_r", vmin=0.75, vmax=1.0)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_title(pos)
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if mat[i, j] > 0.90 else "black",
                )
    fig.tight_layout()
    fig.savefig(FIG / "concordancia_2024.pdf", bbox_inches="tight")
    fig.savefig(FIG / "concordancia_2024.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    for ax, pos in zip(axes, POSITIONS):
        style_ax(ax)
        wide = pred_2024[pred_2024.posicao == pos].pivot_table(
            index="player_id", columns="modelo", values="expected_years"
        )
        if "cox" in wide.columns and "gbsa" in wide.columns:
            ax.scatter(wide["cox"], wide["gbsa"], s=14, c="0.25", edgecolors="none", alpha=0.75)
            lo = float(min(wide["cox"].min(), wide["gbsa"].min()))
            hi = float(max(wide["cox"].max(), wide["gbsa"].max()))
            ax.plot([lo, hi], [lo, hi], color="0.55", lw=0.7, ls=":")
        ax.set_title(pos)
        ax.set_xlabel("Cox, anos esperados")
        if pos == "QB":
            ax.set_ylabel("GBSA, anos esperados")
    fig.tight_layout()
    fig.savefig(FIG / "scatter_cox_gbsa_2024.pdf", bbox_inches="tight")
    fig.savefig(FIG / "scatter_cox_gbsa_2024.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if pred_oof is not None and not pred_oof.empty:
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.15), sharey=True)
        for ax, pos in zip(axes, POSITIONS):
            style_ax(ax)
            sub = pred_oof[pred_oof.posicao == pos]
            xmax = 11.2
            grid = np.linspace(0.2, xmax, 220)
            for m in pred_models_for(pos):
                vals = sub.loc[sub.modelo == m, "expected_years"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) < 8:
                    continue
                kde = gaussian_kde(vals, bw_method=0.28)
                ax.plot(grid, kde(grid), color=col[m], linestyle=ls[m], linewidth=1.7, label=labs[m])
                ax.axvline(np.median(vals), color=col[m], linestyle=ls[m], linewidth=0.7, alpha=0.7)
            ax.set_title(pos)
            ax.set_xlim(0, xmax)
            ax.set_xlabel("anos esperados (OOF)")
            if pos == "QB":
                ax.set_ylabel("densidade")
            if pos == "WR":
                ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(FIG / "dist_anos_oof_2018_2021.pdf", bbox_inches="tight")
        fig.savefig(FIG / "dist_anos_oof_2018_2021.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def summarize_wf(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    off = folds[folds["oficial"] == 1]
    app = folds[folds["oficial"] == 0]

    def agg(df):
        rows = []
        for (modelo, pos), sub in df.groupby(["modelo", "posicao"]):
            rows.append(
                {
                    "modelo": modelo,
                    "posicao": pos,
                    "n_folds": int(len(sub)),
                    "c_index": float(sub["c_index"].mean()),
                    "c_index_sd": float(sub["c_index"].std()) if len(sub) > 1 else np.nan,
                    "ibs": float(sub["ibs"].mean()) if sub["ibs"].notna().any() else np.nan,
                    "mean_time_auc": float(sub["mean_time_auc"].mean())
                    if "mean_time_auc" in sub and sub["mean_time_auc"].notna().any()
                    else np.nan,
                }
            )
        return pd.DataFrame(rows).sort_values(["posicao", "modelo"])

    return agg(off), agg(app)


def summarize_lm(lm: pd.DataFrame) -> pd.DataFrame:
    """Media por (modelo, posicao, w) so nas origens com acompanhamento completo."""
    rows = []
    lm = lm[[landmark_completo(t, w) for t, w in zip(lm["landmark"], lm["horizon_w"])]]
    for (modelo, pos, w), sub in lm.groupby(["modelo", "posicao", "horizon_w"]):
        rows.append(
            {
                "modelo": modelo,
                "posicao": pos,
                "horizon_w": int(w),
                "n_folds": int(len(sub)),
                "c_index": float(sub["c_index"].mean()),
                "c_index_sd": float(sub["c_index"].std()) if len(sub) > 1 else np.nan,
                "ibs": float(sub["ibs"].mean()) if sub["ibs"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["posicao", "horizon_w", "modelo"])


def configura(args: argparse.Namespace) -> None:
    """Define modo de HP, posições, temporadas e pasta de saída antes de rodar."""
    global RES, FIG, MODO_HP, HPS, POSITIONS, TEST_SEASONS
    MODO_HP = args.hp
    POSITIONS = tuple(args.posicoes)
    TEST_SEASONS = list(args.folds)
    if args.saida:
        RES = Path(args.saida).resolve()
    elif MODO_HP == "default":
        RES = HERE / "resultados" / "sensibilidade_hp_default"
    if MODO_HP == "default" or args.saida:
        FIG = RES / "figuras"
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if MODO_HP == "nested":
        HPS = carrega_hps_aninhados(Path(args.hps).resolve())


def main(args: argparse.Namespace) -> None:
    configura(args)
    t0 = time.time()
    print(
        f"HP={MODO_HP} posicoes={list(POSITIONS)} folds={TEST_SEASONS} saida={RES}"
        + (f" tabela={Path(args.hps).resolve()}" if MODO_HP == "nested" else ""),
        flush=True,
    )
    wf, oof = walkforward()
    wf.to_csv(RES / "metricas_walkforward_folds.csv", index=False)
    if args.so_walkforward:
        print(f"so walk-forward: fim {time.time() - t0:.0f}s", flush=True)
        return
    if not oof.empty:
        oof.to_csv(RES / "pred_oof_walkforward_2018_2021.csv", index=False)
    lm = landmark()
    lm.to_csv(RES / "metricas_landmark_folds.csv", index=False)
    pred = predict_2024()
    pred.to_csv(RES / "predicao_2024_modelos.csv", index=False)

    off, app = summarize_wf(wf)
    off.to_csv(RES / "resumo_walkforward_2018_2021.csv", index=False)
    app.to_csv(RES / "resumo_walkforward_2022_2023.csv", index=False)
    lm_sum = summarize_lm(lm)
    lm_sum.to_csv(RES / "resumo_landmark.csv", index=False)

    analyze(pred, oof)
    hp_usado = pd.concat(
        [
            wf[["posicao", "modelo", "teste_em_temporada", "hp_json"]].assign(fonte="walkforward"),
            lm[["posicao", "modelo", "landmark", "hp_json"]]
            .rename(columns={"landmark": "teste_em_temporada"})
            .assign(fonte="landmark(origem=t_L)"),
        ],
        ignore_index=True,
    ).drop_duplicates()
    hp_usado.to_csv(RES / "hp_usado_por_fold.csv", index=False)
    payload = {
        "hp_modo": MODO_HP,
        "hps_fonte": str(Path(args.hps).resolve()) if MODO_HP == "nested" else "tgb_protocol.DEFAULT_*",
        "landmark_hp": "fold walk-forward de mesma temporada t_L (sem busca por horizonte)",
        "seed": SEED,
        "python": sys.version,
        "numpy": versao_pacote("numpy"),
        "pandas": versao_pacote("pandas"),
        "lifelines": versao_pacote("lifelines"),
        "scikit_survival": versao_pacote("scikit-survival"),
        "n_jobs": N_JOBS,
        "oficial_walkforward": off.to_dict(orient="records"),
        "apendice_2022_2023": app.to_dict(orient="records"),
        "landmark": lm_sum.to_dict(orient="records"),
        "secs": round(time.time() - t0, 1),
    }
    (RES / "resumo.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(off.to_string(index=False), flush=True)
    print(lm_sum.to_string(index=False), flush=True)
    print(f"fim {time.time() - t0:.0f}s", flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pacote preditivo oficial (ver docstring).")
    ap.add_argument("--hp", choices=("nested", "default"), default="nested")
    ap.add_argument(
        "--hps",
        default=str(
            ROOT / "documentos" / "estudo_tune_preditivo" / "resultados" / "hps_por_fold.csv"
        ),
        help="tabela da busca aninhada (só com --hp nested)",
    )
    ap.add_argument("--posicoes", default=",".join(POSITIONS))
    ap.add_argument("--folds", default=",".join(str(t) for t in TEST_SEASONS), help="temporadas de teste")
    ap.add_argument("--saida", default=None, help="pasta de saída (teste)")
    ap.add_argument("--so-walkforward", action="store_true", help="só o walk-forward (teste de fumaça)")
    args = ap.parse_args(argv)
    args.posicoes = [x.strip() for x in args.posicoes.split(",") if x.strip()]
    args.folds = [int(x) for x in args.folds.split(",") if x.strip()]
    return args


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
