"""Figura do Harrell C no walk-forward oficial (2018 a 2021).

Le os valores do artefato oficial em vez de os fixar no codigo. A versao
anterior trazia `cox = [0.78, 0.71, 0.73]`, com o RB ainda no valor PRE-freeze
(0,71), enquanto o oficial e 0,72 — a figura contradizia a escrita.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

out = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
RESUMO = ROOT / "documentos" / "estudo_modelos_finais" / "resultados" / "resumo_walkforward_2018_2021.csv"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

pos = ["QB", "RB", "WR"]
if not RESUMO.exists():
    raise FileNotFoundError(f"Artefato oficial ausente: {RESUMO}. Rode tgb_pacote_oficial antes.")
d = pd.read_csv(RESUMO)


def serie(modelo: str) -> list:
    """C-index do modelo por posicao; NaN quando o modelo nao foi ajustado."""
    valores = []
    for p in pos:
        linha = d[(d["modelo"] == modelo) & (d["posicao"] == p)]
        valores.append(float(linha["c_index"].iloc[0]) if not linha.empty else np.nan)
    return valores


km = serie("km")
cox = serie("cox")
gbsa = serie("gbsa")
rsf = serie("rsf")
print("C-index lido do artefato oficial:")
for nome, vals in (("km", km), ("cox", cox), ("gbsa", gbsa), ("rsf", rsf)):
    print(f"  {nome:5s} " + " ".join(f"{p}={v:.6f}" for p, v in zip(pos, vals)))

x = np.arange(len(pos))
w = 0.18

fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.bar(x - 1.5 * w, km, w, label="Kaplan-Meier", color="0.88", edgecolor="black", linewidth=0.6)
ax.bar(x - 0.5 * w, cox, w, label="Cox", color="0.68", edgecolor="black", linewidth=0.6)
ax.bar(x + 0.5 * w, gbsa, w, label="GBSA", color="0.40", edgecolor="black", linewidth=0.6)
ax.bar(x + 1.5 * w, rsf, w, label="RSF", color="0.12", edgecolor="black", linewidth=0.6)

ax.set_xticks(x)
ax.set_xticklabels(pos)
ax.set_ylabel("Harrell $C$")
ax.set_ylim(0.45, 0.92)
ax.axhline(0.50, color="0.35", linewidth=0.6, linestyle=":")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.45, color="0.65")
fig.tight_layout()
fig.savefig(out / "cindex_oficial.pdf", bbox_inches="tight")
fig.savefig(out / "cindex_oficial.png", dpi=220, bbox_inches="tight")
plt.close()
print("ok")
