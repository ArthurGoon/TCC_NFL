#!/usr/bin/env python3
"""Busca aninhada dos hiperparâmetros, fold a fold (protocolo principal).

Para cada posição, modelo (Cox, GBSA, RSF) e temporada S em 2018 a 2024, escolhe o HP
usando só o treino do fold S (retratos anteriores a S, rotulados com o que se sabia em S).
Calibração interna = últimos 2 anos desse treino; o treino interno é rotulado na origem
do início da calibração. Espaço:
  Cox  — 7 penalizers (cartesian)
  RSF  — ntree=500 fixo; cartesian mtry × folha × split × depth (225)
  GBSA — 120 sorteios no espaço largo + default (Bergstra & Bengio)
Seleção no interno por Harrell C. Sem moda, sem congelamento e sem fallback: se a
busca não puder ser feita para um fold, o script para com erro.
A avaliação externa (default x tunado) continua só nos folds oficiais 2018 a 2021.
Saída: resultados/hps_por_fold.csv (lida pelo pacote oficial via tgb_protocol).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "Predição" / "shared"))
sys.path.insert(0, str(ROOT / "Inferência" / "shared"))

from tgb_protocol import (  # noqa: E402
    DEFAULT_GBSA_HP,
    DEFAULT_RSF_HP,
    candidate_times,
    cox_penalizer,
    evaluate_fold,
    load_panel,
    mask_test_support,
    preprocess,
    rotulo_em_tempo_real,
    rsf_permitido,
    walkforward_split,
    y_surv,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "resultados"
FIG = HERE / "figuras"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

OFFICIAL = (2018, 2019, 2020, 2021)
# Busca em todas as temporadas usadas pelo pacote: walk-forward 2018-2023, landmark
# (origens 2020-2023, reusam o HP do fold de mesma origem) e lista 2024.
TUNE_SEASONS = tuple(range(2018, 2025))
POSITIONS = ("QB", "RB", "WR")
MODELOS = ("cox", "gbsa", "rsf")
SEED = 42
CAL_YEARS = 2
GBSA_DRAWS = 120
N_JOBS = max(1, (os.cpu_count() or 2) - 1)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Valores padrão: candidato 0 da grade (referência) e sensibilidade do pacote oficial.
DEFAULTS = {
    "cox": {"penalizer": 0.0},
    "gbsa": dict(DEFAULT_GBSA_HP),
    "rsf": dict(DEFAULT_RSF_HP),
}

SPACE = {
    "cox": {"penalizer": (0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0)},
    "rsf": {
        "n_estimators": (500,),
        "max_features": ("sqrt", "log2", 0.3, 0.5, None),
        "min_samples_leaf": (3, 5, 8, 15, 25),
        "min_samples_split": (6, 12, 20),
        "max_depth": (6, 12, None),
    },
    "gbsa": {
        "learning_rate": (0.01, 0.03, 0.05, 0.1, 0.2),
        "n_estimators": (100, 200, 400, 800),
        "max_depth": (1, 2, 3, 4, 5, 6),
        "subsample": (0.5, 0.7, 0.85, 1.0),
        "min_samples_leaf": (1, 5, 10, 20, 40),
        "max_features": (None, "sqrt"),
        "n_draws": GBSA_DRAWS,
    },
}


def hp_key(hp: dict) -> str:
    return json.dumps(hp, sort_keys=True, default=str)


def _prepend_default(modelo: str, grid: list[dict]) -> list[dict]:
    out = [dict(DEFAULTS[modelo])]
    seen = {hp_key(out[0])}
    for hp in grid:
        k = hp_key(hp)
        if k not in seen:
            out.append(dict(hp))
            seen.add(k)
    return out


def rsf_grid() -> list[dict]:
    spec = SPACE["rsf"]
    grid = []
    for mf in spec["max_features"]:
        for leaf in spec["min_samples_leaf"]:
            for split in spec["min_samples_split"]:
                for depth in spec["max_depth"]:
                    grid.append(
                        {
                            "n_estimators": 500,
                            "max_features": mf,
                            "min_samples_leaf": int(leaf),
                            "min_samples_split": int(split),
                            "max_depth": depth,
                        }
                    )
    return _prepend_default("rsf", grid)


def gbsa_random_grid(n_draws: int = GBSA_DRAWS, seed: int = SEED) -> list[dict]:
    spec = SPACE["gbsa"]
    rng = np.random.default_rng(seed)
    grid = []
    seen: set[str] = set()
    n_lr, n_ne, n_d, n_s, n_l, n_mf = (
        len(spec["learning_rate"]),
        len(spec["n_estimators"]),
        len(spec["max_depth"]),
        len(spec["subsample"]),
        len(spec["min_samples_leaf"]),
        len(spec["max_features"]),
    )
    guard = 0
    while len(grid) < n_draws and guard < n_draws * 50:
        guard += 1
        hp = {
            "learning_rate": float(spec["learning_rate"][int(rng.integers(n_lr))]),
            "n_estimators": int(spec["n_estimators"][int(rng.integers(n_ne))]),
            "max_depth": int(spec["max_depth"][int(rng.integers(n_d))]),
            "subsample": float(spec["subsample"][int(rng.integers(n_s))]),
            "min_samples_leaf": int(spec["min_samples_leaf"][int(rng.integers(n_l))]),
            "max_features": spec["max_features"][int(rng.integers(n_mf))],
        }
        k = hp_key(hp)
        if k in seen:
            continue
        seen.add(k)
        grid.append(hp)
    return _prepend_default("gbsa", grid)


GRIDS = {
    "cox": _prepend_default("cox", [{"penalizer": p} for p in SPACE["cox"]["penalizer"]]),
    "gbsa": gbsa_random_grid(),
    "rsf": rsf_grid(),
}


def surv_from_fns(model, X, times: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return np.empty((len(X), 0))
    fns = model.predict_survival_function(X, return_array=False)
    return np.vstack([np.asarray(fn(times), dtype=float) for fn in fns])


def _ajusta_cox_com_recuo(pen: float, tr: pd.DataFrame, use: list[str]):
    """Mesmo recuo do pacote oficial: se o Cox sem penalizacao nao converge, usa 0,1."""
    cols = ["residual_duration_model", "residual_event"] + list(use)
    ultimo = None
    for p in (pen, 0.1) if pen == 0.0 else (pen,):
        try:
            cph = CoxPHFitter(penalizer=float(p))
            cph.fit(tr[cols], duration_col="residual_duration_model", event_col="residual_event", show_progress=False)
            return cph
        except Exception as exc:  # noqa: BLE001
            ultimo = exc
    raise RuntimeError(ultimo)


def fit_predict(modelo: str, hp: dict, tr: pd.DataFrame, te: pd.DataFrame, use: list[str], times: np.ndarray):
    Xtr = tr[use].to_numpy(dtype=float)
    Xte = te[use].to_numpy(dtype=float)
    ytr = Surv.from_arrays(tr["residual_event"].astype(bool), tr["residual_duration_model"].astype(float))
    if modelo == "cox":
        cph = _ajusta_cox_com_recuo(float(hp["penalizer"]), tr, use)
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
        return model.predict(Xte), surv_from_fns(model, Xte, times)
    if modelo == "rsf":
        model = RandomSurvivalForest(random_state=SEED, n_jobs=1, **hp)
        model.fit(Xtr, ytr)
        return model.predict(Xte), surv_from_fns(model, Xte, times)
    raise ValueError(modelo)


def score_fold(tr, te, risk, surv, times) -> tuple[float, float]:
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


def better(a_c, a_ibs, b_c, b_ibs) -> bool:
    if not np.isfinite(a_c):
        return False
    if not np.isfinite(b_c):
        return True
    if abs(a_c - b_c) > 1e-6:
        return a_c > b_c
    if np.isfinite(a_ibs) and np.isfinite(b_ibs):
        return a_ibs < b_ibs
    return False


def inner_split(train_raw: pd.DataFrame):
    seasons = sorted(int(s) for s in train_raw["snapshot_season"].dropna().unique())
    if len(seasons) < CAL_YEARS + 1:
        return None, None
    cal_seasons = set(seasons[-CAL_YEARS:])
    # O treino interno so pode saber o que se sabia no inicio da validacao interna;
    # a validacao interna fica rotulada ate a origem do fold externo, o que e legitimo.
    inner_tr = rotulo_em_tempo_real(
        train_raw[~train_raw["snapshot_season"].isin(cal_seasons)].copy(), train_raw, min(cal_seasons)
    )
    inner_va = train_raw[train_raw["snapshot_season"].isin(cal_seasons)].copy()
    if len(inner_tr) < 40 or len(inner_va) < 8 or int(inner_va["residual_event"].sum()) < 3:
        return None, None
    return inner_tr, inner_va


def _score_hp_inner(modelo: str, hp: dict, tr, va, use: list[str]) -> dict:
    t0 = time.time()
    err = ""
    cval = np.nan
    try:
        risk, _ = fit_predict(modelo, hp, tr, va, use, times=np.array([], dtype=float))
        cval, _ = score_fold(tr, va, risk, np.empty((len(va), 0)), np.array([], dtype=float))
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    return {
        "hp_json": json.dumps(hp, default=str),
        "inner_c": cval if np.isfinite(cval) else np.nan,
        "ok": bool(np.isfinite(cval)),
        "error": err,
        "secs": round(time.time() - t0, 3),
    }


def choose_hp(modelo: str, train_raw: pd.DataFrame, covs: list[str], pos: str, season: int) -> tuple[dict, dict, list[dict]]:
    inner_tr_raw, inner_va_raw = inner_split(train_raw)
    if inner_tr_raw is None:
        raise RuntimeError(f"busca aninhada impossível ({pos} {modelo} {season}): treino ou validação interna curta")
    tr, va, use, _, _ = preprocess(inner_tr_raw, inner_va_raw, covs)
    va = mask_test_support(tr, va)
    if len(va) < 5 or int(va["residual_event"].sum()) < 2:
        raise RuntimeError(f"busca aninhada impossível ({pos} {modelo} {season}): validação interna com poucos eventos")
    cal_seasons = sorted(int(s) for s in inner_va_raw["snapshot_season"].dropna().unique())
    cands = GRIDS[modelo]
    n_jobs = 1 if modelo == "cox" or len(cands) <= 4 else N_JOBS
    print(
        f"    inner n={len(cands)} jobs={n_jobs} tr={len(tr)} va={len(va)} "
        f"ev_va={int(va['residual_event'].sum())} cal={cal_seasons}",
        flush=True,
    )
    t0 = time.time()
    if n_jobs == 1:
        scored = [_score_hp_inner(modelo, hp, tr, va, use) for hp in cands]
    else:
        scored = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_score_hp_inner)(modelo, hp, tr, va, use) for hp in cands
        )
    best_hp: dict | None = None
    best_c, best_ibs = np.nan, np.nan
    tried = 0
    for hp, row in zip(cands, scored):
        if row["ok"]:
            tried += 1
            cval = float(row["inner_c"])
            if better(cval, np.nan, best_c, best_ibs):
                best_hp, best_c = dict(hp), cval
    if best_hp is None:
        raise RuntimeError(f"busca aninhada falhou em todos os candidatos ({pos} {modelo} {season})")
    trial_rows = []
    for hp, row in zip(cands, scored):
        trial_rows.append(
            {
                "posicao": pos,
                "modelo": modelo,
                "teste_em_temporada": int(season),
                "inner_c": row["inner_c"],
                "ok": row["ok"],
                "error": row["error"],
                "secs": row["secs"],
                "hp_json": row["hp_json"],
                "escolhido": hp_key(hp) == hp_key(best_hp),
            }
        )
    info = {
        "inner_c": best_c,
        "inner_ibs": np.nan,
        "n_tried": tried,
        "n_cands": len(cands),
        "cal_seasons": ",".join(str(s) for s in cal_seasons),
        "n_inner_train": int(len(tr)),
        "n_inner_val": int(len(va)),
        "secs_inner": round(time.time() - t0, 1),
    }
    return best_hp, info, trial_rows


def evaluate_hp(modelo, hp, tr_raw, te_raw, covs, season, tag, pos: str):
    tr, te, use, _, _ = preprocess(tr_raw, te_raw, covs)
    te = mask_test_support(tr, te)
    if len(te) < 5 or int(te["residual_event"].sum()) < 1:
        return None
    times = candidate_times(tr["residual_duration_model"].to_numpy(), te["residual_duration_model"].to_numpy())
    risk, surv = fit_predict(modelo, hp, tr, te, use, times)
    cval, ibs = score_fold(tr, te, risk, surv, times)
    mets = evaluate_fold(
        y_surv(tr, "residual_duration_model", "residual_event"),
        y_surv(te, "residual_duration_model", "residual_event"),
        risk,
        surv,
        times,
        compute_uno=False,
    )
    mets.update(
        {
            "posicao": pos,
            "modelo": modelo,
            "tag": tag,
            "teste_em_temporada": int(season),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_covs": int(len(use)),
            "c_index": cval,
            "ibs": ibs,
            "hp_json": json.dumps(hp, default=str),
        }
    )
    return mets


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.45, color="0.65")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, index=False)


JUSTIFICATIVA_APLICADA = (
    "Protocolo principal: em cada fold, o HP é o escolhido pela busca aninhada com o treino "
    "daquele fold (calibração = 2 últimos anos do treino). Sem moda, sem congelamento e sem "
    "fallback para o padrão. Os valores padrão são a sensibilidade do pacote oficial "
    "(run_oficial.py --hp default)."
)


def lambda_decisao(choice_rows: list[dict]) -> list[dict]:
    """Registro, por (posicao, modelo), do HP usado em cada fold pela busca aninhada."""
    rows = []
    df = pd.DataFrame(choice_rows)
    if df.empty:
        return rows
    for (pos, modelo), sub in df.groupby(["posicao", "modelo"]):
        sub = sub.sort_values("teste_em_temporada")
        hp_por_fold = {str(int(r.teste_em_temporada)): json.loads(r.hp_escolhido) for r in sub.itertuples()}
        rows.append(
            {
                "posicao": pos,
                "modelo": modelo,
                "n_folds": int(len(sub)),
                "temporadas": [int(t) for t in sub["teste_em_temporada"]],
                "hp_por_fold": hp_por_fold,
                "regra_aplicada": "aninhado_por_fold",
                "justificativa": JUSTIFICATIVA_APLICADA,
            }
        )
    return rows


def registra_lambda_decisao(choice_rows: list[dict]) -> list[dict]:
    """Grava resultados/lambda_decisao.json a partir das escolhas por fold."""
    rows = lambda_decisao(choice_rows)
    (RES / "lambda_decisao.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return rows


def resume_oficial(folds: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for (pos, modelo, tag), sub in folds.groupby(["posicao", "modelo", "tag"]):
        summary.append(
            {
                "posicao": pos,
                "modelo": modelo,
                "tag": tag,
                "n_folds": int(len(sub)),
                "c_index": float(sub["c_index"].mean()),
                "c_index_sd": float(sub["c_index"].std()) if len(sub) > 1 else np.nan,
                "ibs": float(sub["ibs"].mean()) if sub["ibs"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(summary).sort_values(["posicao", "modelo", "tag"])


def figura_default_vs_tunado(summ: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.5), sharey=True)
    for ax, pos in zip(axes, POSITIONS):
        style_ax(ax)
        mods = list(MODELOS)
        x = np.arange(len(mods))
        dft = []
        tun = []
        for m in mods:
            sd = summ[(summ.posicao == pos) & (summ.modelo == m) & (summ.tag == "default")]
            st = summ[(summ.posicao == pos) & (summ.modelo == m) & (summ.tag == "tunado")]
            dft.append(float(sd["c_index"].iloc[0]) if len(sd) else np.nan)
            tun.append(float(st["c_index"].iloc[0]) if len(st) else np.nan)
        ax.bar(x - 0.18, dft, 0.36, label="default", color="0.75", edgecolor="black", linewidth=0.55)
        ax.bar(x + 0.18, tun, 0.36, label="tunado", color="0.25", edgecolor="black", linewidth=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in mods])
        ax.set_title(pos)
        ax.set_ylim(0.55, 0.90)
        ax.axhline(0.50, color="0.4", lw=0.5, ls=":")
    axes[0].set_ylabel("Harrell $C$")
    axes[1].legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(FIG / "cindex_default_vs_tunado.pdf", bbox_inches="tight")
    fig.savefig(FIG / "cindex_default_vs_tunado.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    print(
        f"espaço cox={len(GRIDS['cox'])} gbsa={len(GRIDS['gbsa'])} rsf={len(GRIDS['rsf'])} "
        f"cal={CAL_YEARS}a jobs={N_JOBS} temporadas={list(args.folds)} posicoes={list(args.posicoes)} "
        f"modelos={list(args.modelos)} saida={RES}",
        flush=True,
    )
    fold_rows: list[dict] = []
    choice_rows: list[dict] = []
    trial_rows: list[dict] = []
    for pos in args.posicoes:
        panel, covs = load_panel(pos)
        print(f"=== {pos} pool={len(covs)} n={len(panel)} ===", flush=True)
        for modelo in args.modelos:
            # RSF roda em todas as posições (protocolo com busca aninhada por fold).
            if modelo == "rsf" and not rsf_permitido(pos):
                print(f"  rsf {pos}: fora (regra do pacote)", flush=True)
                continue
            for season in args.folds:
                tr_raw, te_raw = walkforward_split(panel, season)
                if len(tr_raw) < 50:
                    raise RuntimeError(f"treino curto para busca: {pos} {modelo} {season} (n={len(tr_raw)})")
                t_fold = time.time()
                best_hp, inner_info, trials = choose_hp(modelo, tr_raw, covs, pos, season)
                trial_rows.extend(trials)
                _write_csv(RES / "inner_trials.csv", trial_rows)
                choice_rows.append(
                    {
                        "posicao": pos,
                        "modelo": modelo,
                        "teste_em_temporada": int(season),
                        "hp_escolhido": json.dumps(best_hp, default=str),
                        **{f"inner_{k}" if not str(k).startswith("inner") else k: v for k, v in inner_info.items()},
                    }
                )
                _write_csv(RES / "hps_por_fold.csv", choice_rows)
                print(f"  {modelo} {season} hp={best_hp} inner_c={inner_info.get('inner_c')}", flush=True)
                # Avaliação externa só nos folds oficiais (2018 a 2021), como antes.
                eh_oficial = int(season) in OFFICIAL
                if not eh_oficial or len(te_raw) < 10 or int(te_raw["residual_event"].sum()) < 1:
                    print(f"    fold {time.time() - t_fold:.0f}s (sem avaliação externa)", flush=True)
                    continue
                for tag, hp in (("default", DEFAULTS[modelo]), ("tunado", best_hp)):
                    try:
                        row = evaluate_hp(modelo, hp, tr_raw, te_raw, covs, season, tag, pos)
                    except Exception as exc:
                        print(f"    FAIL {tag} {exc}", flush=True)
                        traceback.print_exc()
                        continue
                    if row is None:
                        continue
                    fold_rows.append(row)
                    print(f"    {tag}: C={row['c_index']:.3f} IBS={row.get('ibs')}", flush=True)
                _write_csv(RES / "metricas_folds_default_vs_tunado.csv", fold_rows)
                print(f"    fold {time.time() - t_fold:.0f}s", flush=True)

    pd.DataFrame(choice_rows).to_csv(RES / "hps_por_fold.csv", index=False)
    if trial_rows:
        pd.DataFrame(trial_rows).to_csv(RES / "inner_trials.csv", index=False)
    decisao = registra_lambda_decisao(choice_rows)
    if not choice_rows:
        print("nenhum fold buscado", flush=True)
        return

    if fold_rows:
        folds = pd.DataFrame(fold_rows)
        folds.to_csv(RES / "metricas_folds_default_vs_tunado.csv", index=False)
        summ = resume_oficial(folds)
        summ.to_csv(RES / "resumo_oficial_2018_2021.csv", index=False)
        figura_default_vs_tunado(summ)
        print(summ.to_string(index=False), flush=True)

    payload = {
        "regra": (
            "Protocolo principal: busca aninhada em cada fold (treino do fold; calibração = "
            "últimos 2 anos do treino). Cox: 7 penalizers; RSF: 225 combinações (ntree=500), "
            "em QB, RB e WR; GBSA: 120 sorteios + default. Sem moda, congelamento ou fallback."
        ),
        "temporadas_buscadas": [int(t) for t in args.folds],
        "posicoes": list(args.posicoes),
        "modelos": list(args.modelos),
        "cal_years": CAL_YEARS,
        "seed": SEED,
        "n_jobs": N_JOBS,
        "n_cands": {k: len(v) for k, v in GRIDS.items()},
        "space": SPACE,
        "lambda_decisao": decisao,
    }
    (RES / "resumo.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("lambda decisão:", json.dumps(decisao, indent=2, ensure_ascii=False, default=str)[:4000], flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--posicoes", default=",".join(POSITIONS), help="ex.: QB ou QB,RB,WR")
    ap.add_argument("--folds", default=",".join(str(t) for t in TUNE_SEASONS), help="temporadas S da busca")
    ap.add_argument("--modelos", default=",".join(MODELOS), help="ex.: cox,gbsa,rsf")
    ap.add_argument("--jobs", type=int, default=None, help="processos da busca do GBSA e RSF")
    ap.add_argument(
        "--saida",
        default=None,
        help="pasta de saída (teste). Padrão: resultados/ oficial. Nunca sobrescreve o oficial se usada.",
    )
    ap.add_argument(
        "--so-lambda-decisao",
        action="store_true",
        help="só regrava lambda_decisao.json a partir de hps_por_fold.csv (sem re-tunar)",
    )
    args = ap.parse_args(argv)
    args.posicoes = [x.strip() for x in args.posicoes.split(",") if x.strip()]
    args.modelos = [x.strip() for x in args.modelos.split(",") if x.strip()]
    args.folds = [int(x) for x in args.folds.split(",") if x.strip()]
    for pos in args.posicoes:
        if pos not in POSITIONS:
            ap.error(f"posição desconhecida: {pos}")
    for m in args.modelos:
        if m not in MODELOS:
            ap.error(f"modelo desconhecido: {m}")
    for t in args.folds:
        if t not in TUNE_SEASONS:
            ap.error(f"temporada fora de TUNE_SEASONS: {t}")
    return args


if __name__ == "__main__":
    ARGS = parse_args(sys.argv[1:])
    if ARGS.jobs is not None:
        N_JOBS = max(1, int(ARGS.jobs))
    if ARGS.saida:
        RES = Path(ARGS.saida).resolve()
        FIG = RES / "figuras"
        RES.mkdir(parents=True, exist_ok=True)
        FIG.mkdir(parents=True, exist_ok=True)
    if ARGS.so_lambda_decisao:
        # Regenera só o artefato de decisão a partir de hps_por_fold.csv. Não re-tuna.
        registra_lambda_decisao(pd.read_csv(RES / "hps_por_fold.csv").to_dict(orient="records"))
    else:
        main(ARGS)
