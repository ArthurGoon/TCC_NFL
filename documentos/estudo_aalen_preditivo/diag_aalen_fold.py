#!/usr/bin/env python3
"""Diagnostico do fold QB 2018: Y(t), posto e condicao de X(t)'X(t)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figuras"
OUT.mkdir(parents=True, exist_ok=True)
RES = Path(__file__).resolve().parent / "resultados"
RES.mkdir(parents=True, exist_ok=True)

QB = pd.read_csv(ROOT / "Inferência" / "Preparação Modelagem" / "resultados" / "df_cox_QB.csv")
AALEN_CURTA = json.loads(
    (ROOT / "Inferência" / "Modelagem de Aalen" / "resultados" / "aalen_final_covs_QB.json").read_text(
        encoding="utf-8"
    )
)
POOL = json.loads((ROOT / "Predição" / "shared" / "covs_preditivas_QB.json").read_text(encoding="utf-8"))[
    "covariates"
]


def carrega_lag_temporada():
    """Importa ``lag_temporada`` de Predição/shared/add_features.py (defasagem por temporada).

    Só o AST da função é executado: o módulo inteiro gera os pickles de features ao ser importado.
    """
    fonte = ROOT / "Predição" / "shared" / "add_features.py"
    arvore = ast.parse(fonte.read_text(encoding="utf-8"))
    no = next(n for n in arvore.body if isinstance(n, ast.FunctionDef) and n.name == "lag_temporada")
    ns = {"pd": pd}
    exec(compile(ast.Module(body=[no], type_ignores=[]), str(fonte), "exec"), ns)
    return ns["lag_temporada"]


lag_temporada = carrega_lag_temporada()


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["player_id", "season"]).reset_index(drop=True).copy()
    out["depth_lag1"] = lag_temporada(out, "depth_team", 1)
    out["depth_lag2"] = lag_temporada(out, "depth_team", 2)
    out["depth_worsened_lag"] = ((out["depth_lag1"] > out["depth_lag2"]).fillna(0)).astype(int)
    out["off_pct_lag1"] = lag_temporada(out, "offense_pct", 1)
    out["off_pct_lag2"] = lag_temporada(out, "offense_pct", 2)
    out["offense_pct_declined_lag"] = ((out["off_pct_lag1"] < out["off_pct_lag2"]).fillna(0)).astype(int)
    out["gm_lag1"] = lag_temporada(out, "games_missed", 1)
    out["gm_lag2"] = lag_temporada(out, "games_missed", 2)
    out["games_missed_increased_lag"] = ((out["gm_lag1"] > out["gm_lag2"]).fillna(0)).astype(int)
    out["rookie_flag"] = (out["first_observed_season"] == 1).astype(int)
    out["eff_lag1"] = lag_temporada(out, "passer_rating", 1)
    out["eff_lag2"] = lag_temporada(out, "passer_rating", 2)
    out["efficiency_declined_lag"] = ((out["eff_lag1"] < out["eff_lag2"]).fillna(0)).astype(int)
    out["seasons_since_gap"] = 0
    for _, grp in out.groupby("player_id"):
        counter = 0
        for idx in grp.sort_values("season").index:
            if out.at[idx, "had_prior_gap_flag"] == 1:
                counter = 0
            out.at[idx, "seasons_since_gap"] = counter
            counter += 1
    return out


def impute(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        out[col] = s.fillna(s.median())
    return out


def at_risk_matrix(df: pd.DataFrame, t: float, cols: list[str]) -> np.ndarray:
    mask = (df["start"] < t) & (df["stop"] >= t)
    block = df.loc[mask, cols].to_numpy(dtype=float)
    intercept = np.ones((block.shape[0], 1))
    return np.hstack([intercept, block])


def design_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    events = df.loc[df["event"] == 1, "stop"].to_numpy(dtype=float)
    rows = []
    for t in np.unique(events):
        X = at_risk_matrix(df, t, cols)
        y = X.shape[0]
        p = X.shape[1]
        if y == 0:
            continue
        xtx = X.T @ X
        rank = int(np.linalg.matrix_rank(xtx, tol=1e-8))
        try:
            cond = float(np.linalg.cond(xtx))
        except np.linalg.LinAlgError:
            cond = np.inf
        rows.append(
            {
                "t": float(t),
                "Y": y,
                "p_plus_1": p,
                "rank": rank,
                "full_rank": int(rank == p),
                "cond": cond,
                "deficit": p - rank,
            }
        )
    return pd.DataFrame(rows)


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="black")
    ax.yaxis.label.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.set_facecolor("white")


def main() -> None:
    df = add_lag_features(QB)
    train = df[df["season"] < 2018].copy()
    curta = [c for c in AALEN_CURTA if c in train.columns and not c.endswith("_missing")]
    pool = [c for c in POOL if c != "years_exp" and c in train.columns]
    train_curta = impute(train, curta)
    train_pool = impute(train, pool)

    stats_curta = design_stats(train_curta, curta)
    stats_pool = design_stats(train_pool, pool)
    stats_curta["spec"] = "spec_curta"
    stats_pool["spec"] = "pool"
    both = pd.concat([stats_curta, stats_pool], ignore_index=True)
    both.to_csv(RES / "design_stats_qb_2018.csv", index=False)

    summary = {
        "n_train_rows": int(len(train)),
        "n_train_players": int(train["player_id"].nunique()),
        "n_events": int(train["event"].sum()),
        "n_cov_curta": len(curta),
        "n_cov_pool": len(pool),
        "curta_event_times": int(len(stats_curta)),
        "curta_full_rank_frac": float(stats_curta["full_rank"].mean()) if len(stats_curta) else None,
        "pool_full_rank_frac": float(stats_pool["full_rank"].mean()) if len(stats_pool) else None,
        "curta_median_Y": float(stats_curta["Y"].median()) if len(stats_curta) else None,
        "pool_median_Y": float(stats_pool["Y"].median()) if len(stats_pool) else None,
        "curta_min_Y": int(stats_curta["Y"].min()) if len(stats_curta) else None,
        "pool_min_Y": int(stats_pool["Y"].min()) if len(stats_pool) else None,
        "curta_median_cond": float(np.median(stats_curta["cond"])) if len(stats_curta) else None,
        "pool_median_cond": float(np.median(stats_pool["cond"])) if len(stats_pool) else None,
        "pool_times_rank_lt_p": int((stats_pool["full_rank"] == 0).sum()) if len(stats_pool) else None,
        "curta_cols": curta,
        "pool_cols": pool,
    }
    (RES / "resumo_qb_2018.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 11,
            "axes.edgecolor": "black",
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    style_ax(ax)
    ax.plot(stats_curta["t"], stats_curta["Y"], color="black", lw=1.8, label="Em risco Y(t)")
    ax.axhline(len(curta) + 1, color="black", ls=":", lw=1.2, label=f"p+1 da especificação curta ({len(curta)+1})")
    ax.axhline(len(pool) + 1, color="black", ls="--", lw=1.2, label=f"p+1 do pool ({len(pool)+1})")
    ax.set_xlabel("Tempo de carreira t (anos de experiencia)")
    ax.set_ylabel("Jogadores em risco")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "yt_vs_p.pdf")
    fig.savefig(OUT / "yt_vs_p.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    style_ax(ax)
    ax.plot(stats_curta["t"], stats_curta["rank"], color="black", lw=1.8, label="Posto, especificação curta")
    ax.plot(stats_pool["t"], stats_pool["rank"], color="black", lw=1.4, ls="--", label="Posto, pool preditivo")
    ax.axhline(len(curta) + 1, color="black", ls=":", lw=1.0)
    ax.axhline(len(pool) + 1, color="black", ls=":", lw=1.0)
    ax.set_xlabel("Tempo de carreira t")
    ax.set_ylabel("Posto de X(t)'X(t)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "posto_vs_t.pdf")
    fig.savefig(OUT / "posto_vs_t.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    style_ax(ax)
    labels = ["Especificação\ncurta", "Pool preditivo\nsem years_exp"]
    fracs = [
        100 * stats_curta["full_rank"].mean() if len(stats_curta) else 0,
        100 * stats_pool["full_rank"].mean() if len(stats_pool) else 0,
    ]
    ax.bar([0, 1], fracs, color=["black", "black"], width=0.55)
    ax.patches[1].set_facecolor("0.55")
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("% dos tempos de evento com posto completo")
    ax.set_ylim(0, 105)
    for i, v in enumerate(fracs):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "frac_posto_completo.pdf")
    fig.savefig(OUT / "frac_posto_completo.png", dpi=160)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
