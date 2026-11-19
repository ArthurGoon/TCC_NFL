#!/usr/bin/env python3
"""Figura do walk-forward 2018-2021: C de Harrell por origem, por posicao e modelo. Passo 63.

Le os artefatos oficiais e nao reajusta modelo algum:

- ``documentos/estudo_modelos_finais/resultados/metricas_walkforward_folds.csv``: C de cada origem;
- ``documentos/estudo_modelos_finais/resultados/resumo_walkforward_2018_2021.csv``: C medio;
- ``documentos/estudo_ic_preditivo/resultados/ic_metricas.csv``: IC 95% do C medio por modelo.

Grava ``Escrita Final/figuras/c_walkforward_folds.pdf``: um painel por posicao, com um ponto
por origem (2018 a 2021) e a media com IC 95%. A linha vertical marca C = 0,5 (Kaplan-Meier).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "run_all.py").exists() and (candidate / "DataBase").exists():
            return candidate
    raise RuntimeError("Project root not found")


ROOT = find_project_root()
os.chdir(ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(ROOT / "Predição" / "shared"))
from thesis_figure_style import apply_thesis_rcparams, num_br  # noqa: E402

OFICIAL = ROOT / "documentos" / "estudo_modelos_finais" / "resultados"
IC = ROOT / "documentos" / "estudo_ic_preditivo" / "resultados"
SAIDA = ROOT / "Escrita Final" / "figuras" / "c_walkforward_folds.pdf"

POSICOES = ("QB", "RB", "WR")
MODELOS = (("cox", "Cox penalizado"), ("gbsa", "GBSA"), ("rsf", "RSF"))
ORIGENS = {2018: "o", 2019: "s", 2020: "^", 2021: "D"}
COR = "0.25"


def formata(x: float, _pos=None) -> str:
    return num_br(x, 1)


def carrega() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = pd.read_csv(OFICIAL / "metricas_walkforward_folds.csv")
    folds = folds[(folds["protocolo"] == "walkforward") & (folds["oficial"] == 1)]
    resumo = pd.read_csv(OFICIAL / "resumo_walkforward_2018_2021.csv")
    ic = pd.read_csv(IC / "ic_metricas.csv")
    ic = ic[ic["metrica"] == "c_index"]
    return folds, resumo, ic


def main() -> None:
    folds, resumo, ic = carrega()
    apply_thesis_rcparams(font_scale=0.9)
    # Figura baixa e larga: largura do texto e altura ~0,28 da pagina (escala ~1 no LaTeX).

    fig, eixos = plt.subplots(1, 3, figsize=(6.3, 2.45), sharey=True)
    for eixo, pos in zip(eixos, POSICOES):
        for linha, (modelo, rotulo) in enumerate(MODELOS):
            y = len(MODELOS) - 1 - linha
            sub = folds[(folds["posicao"] == pos) & (folds["modelo"] == modelo)]
            for ano, marca in ORIGENS.items():
                c = sub.loc[sub["teste_em_temporada"] == ano, "c_index"]
                if c.empty:
                    raise ValueError(f"origem {ano} ausente para {pos} {modelo}")
                eixo.scatter(float(c.iloc[0]), y, marker=marca, s=20, color=COR, alpha=0.6, linewidths=0, zorder=2)

            media = resumo.loc[(resumo["posicao"] == pos) & (resumo["modelo"] == modelo), "c_index"]
            if media.empty:
                raise ValueError(f"media ausente para {pos} {modelo}")
            ref = ic[(ic["posicao"] == pos) & (ic["modelo"] == modelo)]
            if ref.empty:
                raise ValueError(f"IC ausente para {pos} {modelo}")
            m = float(media.iloc[0])
            baixo, alto = float(ref.iloc[0]["ic_baixo"]), float(ref.iloc[0]["ic_alto"])
            eixo.plot([baixo, alto], [y, y], color="black", lw=1.4, zorder=3, solid_capstyle="butt")
            eixo.scatter(m, y, marker="o", s=42, color="black", zorder=4)

        eixo.axvline(0.5, color="black", ls=(0, (4, 3)), lw=1.3, zorder=1)  # C = 0,5 (Kaplan-Meier), nos tres paineis
        eixo.set_title(pos, fontsize=11.5, pad=3)
        eixo.set_xlim(0.40, 0.82)
        eixo.set_xticks([0.5, 0.6, 0.7, 0.8])
        eixo.xaxis.set_major_formatter(plt.FuncFormatter(formata))
        eixo.set_yticks(range(len(MODELOS)))
        eixo.set_yticklabels([rotulo for _, rotulo in reversed(MODELOS)])
        eixo.set_ylim(-0.6, len(MODELOS) - 0.4)
        eixo.grid(axis="x", color="0.9", lw=0.6, zorder=0)
        eixo.set_xlabel("C de Harrell")
        for lado in ("top", "right"):
            eixo.spines[lado].set_visible(False)

    alcas = [
        plt.Line2D([], [], marker=marca, ls="", color=COR, alpha=0.7, markersize=4, label=f"Origem {ano}")
        for ano, marca in ORIGENS.items()
    ]
    alcas.append(plt.Line2D([], [], marker="o", ls="-", color="black", markersize=5, lw=1.4, label="Média e IC 95%"))
    fig.legend(handles=alcas, loc="upper center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.09), columnspacing=0.8, handletextpad=0.2, handlelength=1.6)

    fig.subplots_adjust(left=0.17, right=0.995, top=0.84, bottom=0.2, wspace=0.12)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA, bbox_inches="tight")
    plt.close(fig)
    print(f"figura gravada: {SAIDA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
