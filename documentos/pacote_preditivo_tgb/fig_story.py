#!/usr/bin/env python3
"""Figuras do pacote preditivo (só esta pasta; não toca figuras da tese)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import gaussian_kde
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Predição" / "shared"))
from tgb_protocol import (
    carrega_hps_aninhados,
    hp_aninhado,
    load_panel,
    preprocess,
    restricted_time_identifiable,
    restricted_time_target,
    rsf_permitido,
)  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figuras"
RES = ROOT / "documentos" / "estudo_modelos_finais" / "resultados"
FIG.mkdir(parents=True, exist_ok=True)

LAB = {
    "years_exp": "anos de experiência",
    "passer_rating": "passer rating",
    "team_pass_attempts_share": "share de passes",
    "depth_team": "depth no elenco",
    "offense_pct": "snaps ofensivos",
    "games_missed": "jogos perdidos",
    "team_rushing_yards_share": "share terrestre",
    "team_targets_share": "share de alvos",
    "draft_ovr": "posição no draft",
    "weight": "peso",
    "height": "altura",
    "unobs_career_years": "anos sem estatística antes da 1ª linha",
    "observed_career_lag_ypc": "YPC (histórico)",
    "observed_career_lag_ypc_qb": "YPC do QB (histórico)",
    "observed_career_lag_td_pct": "% TD (histórico)",
    "observed_career_lag_touches": "toques (histórico)",
    "observed_career_lag_ypr": "YPR (histórico)",
    "adot": "ADOT",
    "observed_career_lag_catch_rate": "catch rate (histórico)",
    "pre_window_veteran": "veterano pré-janela",
    "within_window_gap_years_before_t": "gap recente",
    "undrafted_flag": "não draftado",
    "seasons_since_gap": "temporadas desde gap",
    "draft_round": "rodada do draft",
    "observed_career_lag_fumble_lost": "fumble (histórico)",
}


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.45, color="0.65")


def setup():
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42})


def hit_chart(oof: pd.DataFrame) -> None:
    ev = oof[oof.residual_event == 1].copy()
    ev["ae"] = (ev.expected_years - ev.residual_duration_model).abs()
    champs = {"QB": "gbsa", "RB": "gbsa", "WR": "rsf"}
    hs = (1, 2, 3, 5)
    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    style(ax)
    x = np.arange(len(hs))
    w = 0.26
    ls = {"QB": "-", "RB": "--", "WR": ":"}
    mk = {"QB": "o", "RB": "s", "WR": "D"}
    for i, pos in enumerate(("QB", "RB", "WR")):
        g = ev[(ev.posicao == pos) & (ev.modelo == champs[pos])]
        ys = [100 * (g.ae <= h).mean() for h in hs]
        ax.plot(x, ys, color="0.15", linestyle=ls[pos], marker=mk[pos], linewidth=1.7, markersize=6, label=f"{pos} ({champs[pos].upper()})")
        for xi, yi in zip(x, ys):
            ax.text(xi + (i - 1) * 0.08, yi + 2.2, f"{yi:.0f}", fontsize=8, ha="center", color="0.2")
    ax.set_xticks(x)
    ax.set_xticklabels(["1 ano", "2 anos", "3 anos", "5 anos"])
    ax.set_ylim(0, 108)
    ax.set_ylabel("acerto (%)")
    ax.set_xlabel("janela |previsto − observado|")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "acerto_horizontes.pdf", bbox_inches="tight")
    fig.savefig(FIG / "acerto_horizontes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def scatter_obs(oof: pd.DataFrame) -> None:
    ev = oof[oof.residual_event == 1]
    champs = {"QB": "gbsa", "RB": "gbsa", "WR": "rsf"}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.85), sharey=True)
    for ax, pos in zip(axes, ("QB", "RB", "WR")):
        style(ax)
        g = ev[(ev.posicao == pos) & (ev.modelo == champs[pos])]
        ax.scatter(g.residual_duration_model, g.expected_years, s=12, c="0.25", alpha=0.65, edgecolors="none")
        lo, hi = 0, 8.2
        ax.plot([lo, hi], [lo, hi], color="0.55", lw=0.8, ls=":")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, 11)
        ax.set_title(f"{pos} ({champs[pos].upper()})")
        ax.set_xlabel("observado (anos)")
        if pos == "QB":
            ax.set_ylabel("previsto (anos)")
    fig.tight_layout()
    fig.savefig(FIG / "previsto_vs_observado.pdf", bbox_inches="tight")
    fig.savefig(FIG / "previsto_vs_observado.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _lookup(oof, pos, m, name, yr):
    hit = oof[(oof.posicao == pos) & (oof.modelo == m) & (oof.player_name == name) & (oof.snapshot_season == yr)]
    if hit.empty:
        return None
    r = hit.iloc[0]
    identifiable = restricted_time_identifiable(
        [r.residual_event], [r.residual_duration_model], [r.tau]
    )[0]
    if not identifiable:
        return None
    observed = restricted_time_target(
        [r.residual_duration_model], [r.tau], [identifiable]
    )[0]
    return {
        "label": f"{name} {yr}",
        "obs": float(observed),
        "pred": float(r.expected_years),
    }


def _dumbbell(ax, recs, title):
    style(ax)
    y = np.arange(len(recs))
    for i, r in enumerate(recs):
        ax.plot([r["obs"], r["pred"]], [i, i], color="0.35", lw=1.35)
        ax.scatter([r["obs"]], [i], s=26, c="0.75", zorder=3, label="observado" if i == 0 else None)
        ax.scatter([r["pred"]], [i], s=26, c="0.10", zorder=3, label="previsto" if i == 0 else None)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in recs], fontsize=7.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("anos")


def casos(oof: pd.DataFrame) -> None:
    groups = [
        (
            "Acertos (erro pequeno)",
            [
                ("QB", "gbsa", "Tom Brady", 2018),
                ("QB", "gbsa", "Ryan Tannehill", 2018),
                ("QB", "gbsa", "Teddy Bridgewater", 2019),
                ("RB", "gbsa", "Melvin Gordon", 2019),
                ("RB", "gbsa", "David Johnson", 2018),
                ("RB", "gbsa", "Dalvin Cook", 2018),
                ("WR", "rsf", "Sammy Watkins", 2018),
                ("WR", "rsf", "Julio Jones", 2021),
            ],
        ),
        (
            "Casos médios (erro ≈ mediana)",
            [
                ("QB", "gbsa", "Matt Ryan", 2020),
                ("QB", "gbsa", "Eli Manning", 2018),
                ("RB", "gbsa", "Kenyan Drake", 2020),
                ("RB", "gbsa", "Adrian Peterson", 2020),
                ("WR", "rsf", "Sammy Watkins", 2020),
                ("WR", "rsf", "Larry Fitzgerald", 2018),
            ],
        ),
        (
            "Erros (cauda e extra-campo)",
            [
                ("QB", "gbsa", "Drew Brees", 2018),
                ("QB", "cox", "Drew Brees", 2018),
                ("QB", "gbsa", "Ryan Finley", 2019),
                ("QB", "gbsa", "Andrew Luck", 2018),
                ("RB", "gbsa", "Todd Gurley", 2019),
                ("WR", "rsf", "Henry Ruggs", 2020),
                ("WR", "rsf", "Jarvis Landry", 2019),
            ],
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 4.55))
    for ax, (title, rows) in zip(axes, groups):
        recs = []
        for pos, m, name, yr in rows:
            r = _lookup(oof, pos, m, name, yr)
            if r is None:
                continue
            tag = "Cox" if m == "cox" else pos
            r["label"] = f"{r['label']} ({tag})"
            recs.append(r)
        recs = recs[::-1]
        _dumbbell(ax, recs, title)
    axes[0].legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "casos_individuais.pdf", bbox_inches="tight")
    fig.savefig(FIG / "casos_individuais.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def dist_ae(oof: pd.DataFrame) -> None:
    ev = oof[oof.residual_event == 1].copy()
    ev["ae"] = (ev.expected_years - ev.residual_duration_model).abs()
    champs = {"QB": "gbsa", "RB": "gbsa", "WR": "rsf"}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
    bins = np.arange(0, 9.5, 0.5)
    for ax, pos in zip(axes, ("QB", "RB", "WR")):
        style(ax)
        g = ev[(ev.posicao == pos) & (ev.modelo == champs[pos])].ae
        ax.hist(g, bins=bins, color="0.35", edgecolor="white", linewidth=0.4)
        ax.axvline(g.median(), color="0.1", lw=1.2, ls="--")
        ax.axvline(3.0, color="0.55", lw=0.8, ls=":")
        ax.set_title(f"{pos} ({champs[pos].upper()})")
        ax.set_xlabel("|previsto − observado|")
        if pos == "QB":
            ax.set_ylabel("eventos")
        ax.text(0.97, 0.92, f"mediana {g.median():.1f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "dist_erro_abs.pdf", bbox_inches="tight")
    fig.savefig(FIG / "dist_erro_abs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def error_tails(oof: pd.DataFrame) -> None:
    ev = oof[oof.residual_event == 1].copy()
    ev["err"] = ev.expected_years - ev.residual_duration_model
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
    ls = {"cox": "-", "gbsa": "--", "rsf": ":"}
    col = {"cox": "0.15", "gbsa": "0.35", "rsf": "0.55"}
    xs = np.linspace(-3.2, 8.2, 240)
    for ax, pos in zip(axes, ("QB", "RB", "WR")):
        style(ax)
        mods = ["cox", "gbsa"] + (["rsf"] if rsf_permitido(pos) else [])
        for m in mods:
            g = ev.loc[(ev.posicao == pos) & (ev.modelo == m), "err"].to_numpy(dtype=float)
            kde = gaussian_kde(g)
            ax.plot(xs, kde(xs), linestyle=ls[m], color=col[m], linewidth=1.55, label=m.upper())
        ax.axvline(0.0, color="0.7", linewidth=0.7)
        ax.set_title(pos)
        ax.set_xlabel("previsto − observado")
        if pos == "QB":
            ax.set_ylabel("densidade")
            ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "cauda_erro.pdf", bbox_inches="tight")
    fig.savefig(FIG / "cauda_erro.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def native_importance() -> None:
    # Cox: |β| no X padronizado (leitura nativa do linear).
    # GBSA: feature_importances_ de Friedman (leitura nativa do boosting).
    # RSF: scikit-survival não implementa VIMP; uso a permutação do C já
    # calculada nos folds oficiais (estudo de leitura).
    perm_path = ROOT / "documentos" / "estudo_leitura_preditiva" / "resultados" / "perm_importance_media.csv"
    perm = pd.read_csv(perm_path)
    rows = []
    tops = {pos: {} for pos in ("QB", "RB", "WR")}
    for pos in ("QB", "RB", "WR"):
        panel, covs = load_panel(pos)
        tr_raw = panel[panel.snapshot_season < 2022].copy()
        te_raw = panel[panel.snapshot_season == 2021].copy()
        tr, _, use, _, _ = preprocess(tr_raw, te_raw, covs)

        # Treino com origem 2022 (snapshot < 2022): HP da busca aninhada desse fold.
        hps = carrega_hps_aninhados()
        hp_cox = hp_aninhado(hps, pos, "cox", 2022)
        hp_gbsa = hp_aninhado(hps, pos, "gbsa", 2022)
        cph = CoxPHFitter(penalizer=float(hp_cox["penalizer"]))
        cph.fit(
            tr[["residual_duration_model", "residual_event"] + use],
            duration_col="residual_duration_model",
            event_col="residual_event",
            show_progress=False,
        )
        cox_s = cph.params_.abs().reindex(use).sort_values(ascending=False)
        tops[pos]["cox"] = cox_s.head(6).iloc[::-1]

        ytr = Surv.from_arrays(tr.residual_event.astype(bool), tr.residual_duration_model.astype(float))
        model = GradientBoostingSurvivalAnalysis(random_state=42, **hp_gbsa)
        model.fit(tr[use].to_numpy(dtype=float), ytr)
        gbsa_s = pd.Series(np.asarray(model.feature_importances_, dtype=float), index=use).sort_values(ascending=False)
        tops[pos]["gbsa"] = gbsa_s.head(6).iloc[::-1]

        rsf = perm[(perm.posicao == pos) & (perm.modelo == "rsf")].set_index("feature")["perm_drop"]
        if len(rsf):
            tops[pos]["rsf"] = rsf.sort_values(ascending=False).head(6).iloc[::-1]

        for feat, val in cox_s.items():
            rows.append({"posicao": pos, "modelo": "cox", "feature": feat, "valor": float(val), "tipo": "abs_beta"})
        for feat, val in gbsa_s.items():
            rows.append({"posicao": pos, "modelo": "gbsa", "feature": feat, "valor": float(val), "tipo": "friedman"})
        print(pos, "cox", cox_s.head(6).to_string())
        print(pos, "gbsa", gbsa_s.head(6).to_string())

    pd.DataFrame(rows).to_csv(FIG / "importancia_familias.csv", index=False)

    fig, axes = plt.subplots(3, 3, figsize=(8.2, 6.6))
    titles = {"cox": "Cox |β|", "gbsa": "GBSA nativa", "rsf": "RSF permutação do C"}
    xlabels = {"cox": "|β| padronizado", "gbsa": "fração do ganho", "rsf": "queda do C"}
    for i, pos in enumerate(("QB", "RB", "WR")):
        for j, modelo in enumerate(("cox", "gbsa", "rsf")):
            ax = axes[i, j]
            style(ax)
            if modelo not in tops[pos]:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "RSF fora no QB", ha="center", va="center", fontsize=8, transform=ax.transAxes)
                continue
            s = tops[pos][modelo]
            labels = [LAB.get(k, k.replace("_", " ")[:20]) for k in s.index]
            ax.barh(range(len(s)), s.values, color="0.28", height=0.72)
            ax.set_yticks(range(len(s)))
            ax.set_yticklabels(labels, fontsize=6.6)
            if i == 0:
                ax.set_title(titles[modelo], fontsize=9)
            if j == 0:
                ax.set_ylabel(pos, fontsize=9)
            if i == 2:
                ax.set_xlabel(xlabels[modelo], fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "importancia_nativa.pdf", bbox_inches="tight")
    fig.savefig(FIG / "importancia_nativa.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    oof = pd.read_csv(RES / "pred_oof_walkforward_2018_2021.csv")
    hit_chart(oof)
    scatter_obs(oof)
    casos(oof)
    dist_ae(oof)
    error_tails(oof)
    native_importance()
    print("figuras em", FIG)


if __name__ == "__main__":
    main()
