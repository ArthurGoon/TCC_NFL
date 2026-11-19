#!/usr/bin/env python3
"""Figuras do Cap. 5 (prefixo tgb_*)."""
from __future__ import annotations

import os
import textwrap
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Predição" / "shared"))
from thesis_figure_style import (  # noqa: E402
    apply_thesis_rcparams,
    figsize_full,
    save_thesis_figure,
    thesis_model_label,
    thesis_var_label,
)
from tgb_protocol import (
    modelos_destaque,
    restricted_time_identifiable,
    restricted_time_target,
    rsf_permitido,
)  # noqa: E402

RES = ROOT / "documentos" / "estudo_modelos_finais" / "resultados"
PACOTE_FIG = ROOT / "documentos" / "pacote_preditivo_tgb" / "figuras"
PERM = ROOT / "documentos" / "estudo_leitura_preditiva" / "resultados" / "perm_importance_media.csv"
DEST = ROOT / "Escrita Final" / "figuras"

# Modelo de maior C e modelo de contraste por posicao, lidos do resultado (sem lista fixa).
_DESTAQUE = modelos_destaque(pd.read_csv(RES / "resumo_walkforward_2018_2021.csv"))
CHAMPS = {pos: par[0] for pos, par in _DESTAQUE.items()}
POS = ("QB", "RB", "WR")
NAVY = "#2E4057"
GREEN = "#1D9E75"
ORANGE = "#D85A30"
MUTED = "#8A9BB0"



def _lab(name: str) -> str:
    return thesis_var_label(name)


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.45, color="0.65")


def _error_view(oof: pd.DataFrame) -> pd.DataFrame:
    out = oof.copy()
    tau = out["tau"].to_numpy(dtype=float)
    followup = out["residual_duration_model"].to_numpy(dtype=float)
    event = out["residual_event"].to_numpy(dtype=float)
    out["error_identifiable"] = restricted_time_identifiable(event, followup, tau)
    out["error_target_years"] = restricted_time_target(
        followup, tau, out["error_identifiable"].to_numpy(dtype=bool)
    )
    return out[out.error_identifiable].copy()


def _save(fig, name: str) -> None:
    save_thesis_figure(fig, DEST / name, also_png=False)
    plt.close(fig)


def cindex_bars(wf: pd.DataFrame) -> None:
    apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=figsize_full(aspect=0.42))
    _style(ax)
    models = ["km", "cox", "gbsa", "rsf"]
    x = np.arange(len(POS))
    w = 0.18
    colors = {"km": MUTED, "cox": NAVY, "gbsa": GREEN, "rsf": ORANGE}
    for i, m in enumerate(models):
        ys = []
        for pos in POS:
            row = wf[(wf.modelo == m) & (wf.posicao == pos)]
            ys.append(float(row.c_index.iloc[0]) if len(row) else np.nan)
        ax.bar(
            x + (i - 1.5) * w,
            ys,
            width=w,
            color=colors[m],
            edgecolor="white",
            linewidth=0.4,
            label=thesis_model_label(m),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(POS)
    ax.set_ylim(0.45, 0.90)
    ax.set_ylabel("$C$-index de Harrell")
    ax.axhline(0.5, color="0.55", lw=0.7, ls=":")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    fig.tight_layout()
    _save(fig, "tgb_cindex_walkforward.pdf")


def hit_chart(hits: pd.DataFrame) -> None:
    apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=figsize_full(aspect=0.42))
    _style(ax)
    hs = (1, 2, 3, 5)
    cols = {1: "acerto_1a", 2: "acerto_2a", 3: "acerto_3a", 5: "acerto_5a"}
    x = np.arange(len(hs))
    ls = {"QB": "-", "RB": "--", "WR": ":"}
    mk = {"QB": "o", "RB": "s", "WR": "D"}
    pos_col = {"QB": NAVY, "RB": GREEN, "WR": ORANGE}
    for i, pos in enumerate(POS):
        g = hits[(hits.posicao == pos) & (hits.modelo == CHAMPS[pos])].iloc[0]
        ys = [float(g[cols[h]]) for h in hs]
        ax.plot(
            x,
            ys,
            color=pos_col[pos],
            linestyle=ls[pos],
            marker=mk[pos],
            linewidth=1.7,
            markersize=6,
            label=f"{pos} ({thesis_model_label(CHAMPS[pos])})",
        )
        for xi, yi in zip(x, ys):
            # Separacao horizontal maior e fundo branco: em "5 anos" as tres
            # series ficam entre 94% e 100% e os rotulos se sobrepunham.
            ax.text(
                xi + (i - 1) * 0.14,
                yi + 2.4,
                f"{yi:.0f}",
                fontsize=7.5,
                ha="center",
                va="bottom",
                color=pos_col[pos],
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.85),
            )
    ax.set_xticks(x)
    ax.set_xticklabels(["1 ano", "2 anos", "3 anos", "5 anos"])
    ax.set_ylim(0, 112)
    ax.set_ylabel("acerto (%)")
    ax.set_xlabel(r"janela $|\mathrm{previsto}-\mathrm{observado}|$")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "tgb_acerto_horizontes.pdf")


def scatter_obs(oof: pd.DataFrame) -> None:
    apply_thesis_rcparams()
    ev = _error_view(oof)
    fig, axes = plt.subplots(1, 3, figsize=figsize_full(aspect=0.38), sharey=True)
    for ax, pos in zip(axes, POS):
        _style(ax)
        g = ev[(ev.posicao == pos) & (ev.modelo == CHAMPS[pos])]
        ax.scatter(g.error_target_years, g.expected_years, s=12, c=NAVY, alpha=0.65, edgecolors="none")
        # previsto e observado vao ate tau = 2024 - S, no maximo 6 anos nos folds de referencia
        ax.plot([0, 6.4], [0, 6.4], color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(-0.2, 6.4)
        ax.set_ylim(0, 6.4)
        ax.set_xticks([0, 2, 4, 6])
        ax.set_title(f"{pos} ({thesis_model_label(CHAMPS[pos])})")
        ax.set_xlabel("observado (anos)")
        if pos == "QB":
            ax.set_ylabel("previsto (anos)")
    fig.tight_layout()
    _save(fig, "tgb_previsto_vs_observado.pdf")


# Casos da figura tgb_casos_individuais: regra, sem lista fixa.
# Faixas sobre o erro (previsto - observado, em anos) do modelo de maior C de cada
# posicao, as mesmas do auditor (reprodutibilidade/audit_escrita_vs_pipeline.py, 5.6):
#   Erro pequeno:             |erro| <= 0,6                       referencia 0
#   Erro proximo da mediana:  0,6 < erro <= mediana + 0,5        referencia: ponto medio
#                             (mediana = mediana do |erro| dos tres campeoes juntos)
#   Erros graves:             erro > 2 (faixa aberta)            referencia 2,0
# Em cada painel, para cada posicao, entram os CASOS_POR_POSICAO pares (jogador,
# temporada de origem) com erro dentro da faixa e mais proximos da referencia;
# empate por player_id e temporada de origem.
FAIXAS_CASOS = ("Erro pequeno", "Erro próximo da mediana", "Erros graves")
CASOS_POR_POSICAO = 2


def _na_faixa(faixa: str, erro: pd.Series, mediana: float) -> pd.Series:
    if faixa == "Erro pequeno":
        return erro.abs() <= 0.6
    if faixa == "Erro próximo da mediana":
        return (erro > 0.6) & (erro <= mediana + 0.5)
    return erro > 2.0


def _referencia(faixa: str, mediana: float) -> float:
    if faixa == "Erro pequeno":
        return 0.0
    if faixa == "Erro próximo da mediana":
        return (0.6 + mediana + 0.5) / 2
    return 2.0


def selecao_casos(oof: pd.DataFrame, champs: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Painel -> linhas escolhidas (CASOS_POR_POSICAO por posicao), pela regra acima."""
    eligible = _error_view(oof)
    campeoes = pd.concat([eligible[(eligible.posicao == p) & (eligible.modelo == champs[p])] for p in POS])
    mediana = float((campeoes.expected_years - campeoes.error_target_years).abs().median())
    paineis: dict[str, pd.DataFrame] = {}
    for faixa in FAIXAS_CASOS:
        partes = []
        for pos in POS:
            g = eligible[(eligible.posicao == pos) & (eligible.modelo == champs[pos])].copy()
            g["erro"] = g.expected_years - g.error_target_years
            g = g[_na_faixa(faixa, g.erro, mediana)].copy()
            g["dist"] = (g.erro - _referencia(faixa, mediana)).abs()
            partes.append(g.sort_values(["dist", "player_id", "snapshot_season"]).head(CASOS_POR_POSICAO))
        paineis[faixa] = pd.concat(partes)
    return paineis


def casos(oof: pd.DataFrame) -> None:
    apply_thesis_rcparams()
    paineis = selecao_casos(oof, CHAMPS)
    fig, axes = plt.subplots(1, 3, figsize=figsize_full(aspect=0.58), sharex=True)
    legend_done = False
    for ax, title in zip(axes, FAIXAS_CASOS):
        recs = []
        for r in paineis[title].itertuples():
            last = r.player_name.split()[-1]
            recs.append(
                {
                    "label": f"{last} {r.snapshot_season} ({r.posicao})",
                    "obs": float(r.error_target_years),
                    "pred": float(r.expected_years),
                }
            )
        recs = recs[::-1]
        _style(ax)
        y = np.arange(len(recs))
        for i, r in enumerate(recs):
            ax.plot([r["obs"], r["pred"]], [i, i], color=MUTED, lw=1.2)
            ax.scatter(
                [r["obs"]],
                [i],
                s=28,
                c=ORANGE,
                zorder=3,
                label="observado" if not legend_done else None,
            )
            ax.scatter(
                [r["pred"]],
                [i],
                s=28,
                c=NAVY,
                zorder=3,
                label="previsto" if not legend_done else None,
            )
            legend_done = True
        ax.set_yticks(y)
        ax.set_yticklabels([r["label"] for r in recs], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 6.3)
        ax.set_xticks([0, 2, 4, 6])
        ax.tick_params(axis="x", labelsize=7.5)
    fig.supxlabel("anos", fontsize=9)
    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "tgb_casos_individuais.pdf")


def dist_ae(oof: pd.DataFrame) -> None:
    apply_thesis_rcparams()
    ev = _error_view(oof)
    ev["ae"] = (ev.expected_years - ev.error_target_years).abs()
    fig, axes = plt.subplots(1, 3, figsize=figsize_full(aspect=0.38), sharey=True)
    bins = np.arange(0, 9.5, 0.5)
    for ax, pos in zip(axes, POS):
        _style(ax)
        g = ev[(ev.posicao == pos) & (ev.modelo == CHAMPS[pos])].ae
        ax.hist(g, bins=bins, color=NAVY, edgecolor="white", linewidth=0.4)
        ax.axvline(g.median(), color=ORANGE, lw=1.2, ls="--")
        ax.axvline(3.0, color=MUTED, lw=0.8, ls=":")
        ax.set_title(f"{pos} ({thesis_model_label(CHAMPS[pos])})")
        if pos == "QB":
            ax.set_ylabel("eventos")
        ax.text(0.97, 0.92, f"mediana {g.median():.1f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    # Um rotulo unico embaixo: repetido por painel, o texto invadia o vizinho.
    fig.supxlabel(r"$|\mathrm{previsto}-\mathrm{observado}|$", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "tgb_dist_erro_abs.pdf")


def error_tails(oof: pd.DataFrame) -> None:
    apply_thesis_rcparams()
    ev = _error_view(oof)
    ev["err"] = ev.expected_years - ev.error_target_years
    fig, axes = plt.subplots(1, 3, figsize=figsize_full(aspect=0.38), sharey=True)
    ls = {"cox": "-", "gbsa": "--", "rsf": ":"}
    col = {"cox": NAVY, "gbsa": GREEN, "rsf": ORANGE}
    xs = np.linspace(-3.2, 8.2, 240)
    for ax, pos in zip(axes, POS):
        _style(ax)
        mods = ["cox", "gbsa"] + (["rsf"] if rsf_permitido(pos) else [])
        for m in mods:
            g = ev.loc[(ev.posicao == pos) & (ev.modelo == m), "err"].to_numpy(dtype=float)
            kde = gaussian_kde(g)
            ax.plot(xs, kde(xs), linestyle=ls[m], color=col[m], linewidth=1.55, label=thesis_model_label(m))
        ax.axvline(0.0, color="0.7", linewidth=0.7)
        ax.set_title(pos)
        if pos == "QB":
            ax.set_ylabel("densidade")
            ax.legend(frameon=False, fontsize=7.5)
    fig.supxlabel("previsto $-$ observado", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "tgb_cauda_erro.pdf")


def native_importance() -> None:
    apply_thesis_rcparams()
    fam = pd.read_csv(PACOTE_FIG / "importancia_familias.csv")
    perm = pd.read_csv(PERM)
    fig, axes = plt.subplots(3, 3, figsize=figsize_full(aspect=1.16))
    titles = {"cox": r"Cox $|\beta|$", "gbsa": "GBSA (Friedman)", "rsf": "RSF (permutação do $C$)"}
    xlabels = {"cox": r"$|\beta|$ padronizado", "gbsa": "fração do ganho", "rsf": "queda do $C$"}
    for i, pos in enumerate(POS):
        for j, modelo in enumerate(("cox", "gbsa", "rsf")):
            ax = axes[i, j]
            _style(ax)
            if modelo == "rsf" and not rsf_permitido(pos):
                ax.set_axis_off()
                ax.text(0.5, 0.5, "RSF não ajustado", ha="center", va="center", fontsize=8, transform=ax.transAxes)
                continue
            if modelo == "rsf":
                s = (
                    perm[(perm.posicao == pos) & (perm.modelo == "rsf")]
                    .set_index("feature")["perm_drop"]
                    .sort_values(ascending=False)
                    .head(6)
                    .iloc[::-1]
                )
            else:
                s = (
                    fam[(fam.posicao == pos) & (fam.modelo == modelo)]
                    .set_index("feature")["valor"]
                    .sort_values(ascending=False)
                    .head(6)
                    .iloc[::-1]
                )
            ax.barh(range(len(s)), s.values, color=NAVY, height=0.72)
            upper_limit = MaxNLocator(nbins=1).tick_values(0, float(s.max()))[-1]
            ax.set_xlim(0, upper_limit)
            ax.set_xticks([0, upper_limit])
            ax.set_xticklabels(["0", format(upper_limit, ".3g").replace(".", ",")])
            ax.tick_params(axis="x", labelsize=8, labelrotation=35, pad=2)
            ax.set_yticks(range(len(s)))
            ax.set_yticklabels(["\n".join(textwrap.wrap(_lab(k), 20)) for k in s.index], fontsize=8, linespacing=1.05)
            if i == 0:
                ax.set_title(titles[modelo], fontsize=9.5)
            if j == 0:
                ax.set_ylabel(pos, fontsize=10)
            if i == 2:
                ax.set_xlabel(xlabels[modelo], fontsize=8.5)
    fig.tight_layout(h_pad=1.1, w_pad=0.6)
    _save(fig, "tgb_importancia_nativa.pdf")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(RES / "pred_oof_walkforward_2018_2021.csv")
    wf = pd.read_csv(RES / "resumo_walkforward_2018_2021.csv")
    hits = pd.read_csv(RES / "auditoria" / "acerto_horizontes.csv")
    cindex_bars(wf)
    hit_chart(hits)
    scatter_obs(oof)
    casos(oof)
    dist_ae(oof)
    error_tails(oof)
    native_importance()
    print("figuras em", DEST)


if __name__ == "__main__":
    main()
