#!/usr/bin/env python3
"""Leitura do landmarking jogador a jogador, com o modelo principal de cada posição.

Refaz os ajustes do landmarking oficial (mesmo recorte, pré-processamento e
hiperparâmetros de `run_oficial.landmark`) só para o modelo de maior C em cada
posição, e guarda o que o pacote oficial não guarda: a probabilidade prevista de
saída em w anos por jogador. Confere se o C reproduz `metricas_landmark_folds.csv`.

Saídas em documentos/estudo_leitura_aplicada/resultados/:
  landmark_jogadores.csv  uma linha por jogador, origem e horizonte
  landmark_quintis.csv    saída prevista e observada por quintil de risco
  landmark_casos.csv      extremos da origem 2023 com w = 1
  tgb_landmark_quintis.pdf  saida prevista e observada por quintil de risco
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

ROOT = Path(__file__).resolve().parent.parent
OF = ROOT / "documentos" / "estudo_modelos_finais"
SAIDA = ROOT / "documentos" / "estudo_leitura_aplicada" / "resultados"
ULTIMA_TEMPORADA = 2024


def _modelo_principal() -> dict[str, str]:
    """Modelo de maior C por posicao, lido do resumo oficial, sem lista fixa."""
    sys.path.insert(0, str(ROOT / "Predição" / "shared"))
    from tgb_protocol import modelos_destaque

    resumo = pd.read_csv(OF / "resultados" / "resumo_walkforward_2018_2021.csv")
    return {pos: par[0] for pos, par in modelos_destaque(resumo).items()}


PRINCIPAL = _modelo_principal()


def carrega_oficial():
    sys.path.insert(0, str(OF))
    spec = importlib.util.spec_from_file_location("run_oficial", OF / "run_oficial.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_oficial"] = mod
    spec.loader.exec_module(mod)
    return mod


def jogadores(of) -> pd.DataFrame:
    linhas = []
    oficial = pd.read_csv(OF / "resultados" / "metricas_landmark_folds.csv")
    for pos, modelo in PRINCIPAL.items():
        panel, covs = of.load_panel(pos)
        for t_l in of.LANDMARKS:
            for w in of.LANDMARK_WS:
                if t_l + w > ULTIMA_TEMPORADA:  # sem w anos de acompanhamento depois da origem
                    continue
                tr_raw, te_raw = of.landmark_split(panel, t_l, float(w))
                tr, te, use, _, _ = of.preprocess(tr_raw, te_raw, covs)
                hp = of.hp_do_fold(pos, modelo, t_l)
                risk, surv, _ = of.fit_predict(modelo, hp, tr, te, use, "lm_time", "lm_event", np.array([float(w)]))
                c = concordance_index_censored(te["lm_event"].astype(bool).to_numpy(), te["lm_time"].astype(float).to_numpy(), risk)[0]
                ref = oficial[(oficial.posicao == pos) & (oficial.modelo == modelo) & (oficial.landmark == t_l) & (oficial.horizon_w == w)].c_index.iloc[0]
                if abs(c - ref) > 1e-3:
                    raise RuntimeError(f"{pos} {modelo} {t_l} w={w}: C {c:.4f} difere do oficial {ref:.4f}")
                out = te[["player_id", "player_name", "lm_time", "lm_event"]].copy()
                out["posicao"], out["modelo"], out["origem"], out["w"] = pos, modelo, t_l, w
                # so as origens com acompanhamento ate t_L + w + 1 entram nos quintis
                out["completa"] = int(of.landmark_completo(t_l, w))
                out["risco"] = risk
                out["p_saida_w"] = 1.0 - surv[:, 0]
                # quintil dentro da origem e do horizonte: 1 = menor risco
                out["quintil"] = pd.qcut(out["p_saida_w"].rank(method="first"), 5, labels=False) + 1
                out["saiu_em_w"] = (out["lm_event"].astype(bool) & (out["lm_time"] <= w)).astype(int)
                # nao voltou na temporada seguinte a origem (T = 0, registrado como 0,001 no ajuste)
                out["nao_voltou"] = (out["lm_event"].astype(bool) & (out["lm_time"] < 0.5)).astype(int)
                linhas.append(out)
                print(f"{pos} {modelo} {t_l} w={w} n={len(out)} C={c:.3f}", flush=True)
    return pd.concat(linhas, ignore_index=True)


def figura_quintis(q: pd.DataFrame) -> Path:
    """Saida prevista e observada em w anos por quintil de risco, um painel por posicao."""
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sys.path.insert(0, str(ROOT / "Predição" / "shared"))
    from thesis_figure_style import apply_thesis_rcparams, eixos_virgula, figsize_full, save_thesis_figure

    apply_thesis_rcparams()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
    cor_w = {1: "#2E4057", 2: "#D85A30", 3: "#1D9E75"}
    fw = figsize_full()[0]
    fig, axes = plt.subplots(1, 3, figsize=(fw, fw * 0.4), sharey=True)
    for ax, pos in zip(axes, ("QB", "RB", "WR")):
        modelo = PRINCIPAL[pos]
        for w in (1, 2, 3):
            s = q[(q.posicao == pos) & (q.w == w)].sort_values("quintil")
            ax.plot(s.quintil, 100 * s.prevista, color=cor_w[w], linestyle="--", linewidth=1.3)
            ax.plot(s.quintil, 100 * s.observada, color=cor_w[w], linestyle="-", linewidth=1.3,
                    marker="o", markersize=4)
        ax.set_title(f"{pos} ({'RSF' if modelo == 'rsf' else 'Cox'})", loc="left", fontweight="bold")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        eixos_virgula(ax, x=False, y=True)
    axes[0].set_ylabel("saída em $w$ anos (%)")
    fig.supxlabel("quintil de risco previsto (1 = menor risco, 5 = maior risco)", y=0.13, fontsize=9)
    handles = [Line2D([], [], color=cor_w[w], linewidth=1.6, label=f"$w = {w}$") for w in (1, 2, 3)]
    handles += [Line2D([], [], color="0.3", linestyle="-", marker="o", markersize=4, label="observada"),
                Line2D([], [], color="0.3", linestyle="--", label="prevista")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=1.0)
    caminho = SAIDA / "tgb_landmark_quintis.pdf"
    save_thesis_figure(fig, caminho, also_png=False)
    plt.close(fig)
    return caminho


def main() -> None:
    of = carrega_oficial()
    d = jogadores(of)
    SAIDA.mkdir(parents=True, exist_ok=True)
    d.to_csv(SAIDA / "landmark_jogadores.csv", index=False)
    comp = d[d.completa == 1]
    if not all(of.landmark_completo(o, w) for o, w in zip(comp.origem, comp.w)):
        raise RuntimeError("origem sem acompanhamento completo entrou nos quintis")
    q = (
        comp.groupby(["posicao", "modelo", "w", "quintil"])
        .agg(n=("saiu_em_w", "size"), saidas=("saiu_em_w", "sum"), prevista=("p_saida_w", "mean"), observada=("saiu_em_w", "mean"), nao_voltou=("nao_voltou", "mean"))
        .reset_index()
    )
    q.to_csv(SAIDA / "landmark_quintis.csv", index=False)
    figura_quintis(q)
    # Origem 2023 com w = 1: fora das medias, mas "nao voltou em 2024" e observavel nela.
    ult = d[(d.origem == 2023) & (d.w == 1)].sort_values(["posicao", "p_saida_w"])
    ult.to_csv(SAIDA / "landmark_casos.csv", index=False)


if __name__ == "__main__":
    main()
