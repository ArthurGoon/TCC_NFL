"""Figuras do pacote preditivo: distribuição 2024 e concordância entre modelos."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "Predição" / "shared"))
from tgb_protocol import rsf_permitido  # noqa: E402

FIG = HERE / "figuras"
(HERE / "resultados").mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PRED_CSV = HERE.parents[1] / "documentos" / "estudo_leitura_preditiva" / "resultados" / "predicao_2024_modelos.csv"
df = pd.read_csv(PRED_CSV)

POS = ["QB", "RB", "WR"]
MODS = {"cox": "Cox", "gbsa": "GBSA", "rsf": "RSF"}
LS = {"cox": "-", "gbsa": "--", "rsf": ":"}
COL = {"cox": "0.55", "gbsa": "0.15", "rsf": "0.00"}
LW = {"cox": 1.6, "gbsa": 1.8, "rsf": 1.7}


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.45, color="0.65")


# --- distribuição ---
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.15), sharey=True)
for ax, pos in zip(axes, POS):
    style(ax)
    sub = df[df.posicao == pos]
    mods = ["cox", "gbsa"] + (["rsf"] if rsf_permitido(pos) else [])
    xmax = 11.2
    grid = np.linspace(0.2, xmax, 220)
    for m in mods:
        vals = sub.loc[sub.modelo == m, "expected_years"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        kde = gaussian_kde(vals, bw_method=0.28)
        ax.plot(grid, kde(grid), color=COL[m], linestyle=LS[m], linewidth=LW[m], label=MODS[m])
        ax.axvline(np.median(vals), color=COL[m], linestyle=LS[m], linewidth=0.7, alpha=0.7)
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
plt.close()

# --- concordância (Spearman dos anos esperados) ---
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.85))
for ax, pos in zip(axes, POS):
    mods = ["cox", "gbsa"] + (["rsf"] if rsf_permitido(pos) else [])
    labels = [MODS[m] for m in mods]
    wide = (
        df[df.posicao == pos]
        .pivot_table(index="player_id", columns="modelo", values="expected_years")
        .dropna()
    )
    n = len(mods)
    mat = np.eye(n)
    for i, a in enumerate(mods):
        for j, b in enumerate(mods):
            if i == j:
                continue
            mat[i, j] = float(spearmanr(wide[a], wide[b]).statistic)
    im = ax.imshow(mat, cmap="gray_r", vmin=0.75, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(pos)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > 0.90 else "black")
fig.tight_layout()
fig.savefig(FIG / "concordancia_2024.pdf", bbox_inches="tight")
fig.savefig(FIG / "concordancia_2024.png", dpi=220, bbox_inches="tight")
plt.close()

# --- resumo numérico ---
rows = []
for pos in POS:
    for m, lab in MODS.items():
        if pos == "QB" and m == "rsf":
            continue
        v = df[(df.posicao == pos) & (df.modelo == m)]["expected_years"]
        rows.append({
            "posicao": pos, "modelo": lab, "n": int(v.size),
            "p25": float(v.quantile(0.25)), "mediana": float(v.median()),
            "p75": float(v.quantile(0.75)), "media": float(v.mean()),
        })
pd.DataFrame(rows).to_csv(HERE / "resultados" / "dist_anos_2024.csv", index=False)
print("ok", FIG)
print(pd.DataFrame(rows).to_string(index=False))
