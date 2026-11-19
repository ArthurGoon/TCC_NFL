#!/usr/bin/env python3
"""Aalen em bloco como modelo discreto: confere a identidade com a regressao linear de cada ano
e compara o Aalen com um modelo de probabilidade linear agregado (efeito fixo por ano, erro por jogador).

Saidas em Inferência/sensibilidade/resultados/: aalen_discreto_resumo.json e aalen_discreto_lpm.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
AALEN = ROOT / "Inferência" / "Modelagem de Aalen" / "resultados"
SAIDA = ROOT / "Inferência" / "sensibilidade" / "resultados"


def main() -> None:
    linhas, resumo = [], {}
    for pos in ("QB", "RB", "WR"):
        d = pd.read_csv(AALEN / f"aalen_input_{pos}.csv", low_memory=False)
        s = json.loads((AALEN / f"aalen_summary_{pos}.json").read_text(encoding="utf-8"))
        covs, ini, fim = s["covariates_tested"], int(s["janela_inicio"]), int(s["janela_fim"])
        cum = pd.read_csv(AALEN / f"aalen_cum_coefs_{pos}.csv")
        tcol = next(c for c in cum.columns if c.lower() in ("time", "times", "t"))
        cum = cum.set_index(tcol)
        janela = d[(d["stop"] > ini) & (d["stop"] <= fim)].copy()
        maxdif = 0.0
        anteriores = pd.Series(0.0, index=covs)
        for t in sorted(janela["stop"].unique()):
            g = janela[janela["stop"] == t]
            X = np.column_stack([np.ones(len(g)), g[covs].to_numpy(float)])
            beta = np.linalg.lstsq(X, g["event"].to_numpy(float), rcond=None)[0][1:]
            cum_t = cum.loc[t, covs].astype(float)
            maxdif = max(maxdif, float(np.abs(beta - (cum_t - anteriores).to_numpy()).max()))
            anteriores = cum_t
        anos = pd.get_dummies(janela["stop"].astype(int), drop_first=True).to_numpy(float)
        X = np.column_stack([np.ones(len(janela)), janela[covs].to_numpy(float), anos])
        y = janela["event"].to_numpy(float)
        XtXi = np.linalg.inv(X.T @ X)
        b = XtXi @ X.T @ y
        u = (y - X @ b)[:, None] * X
        grupos = pd.Series(janela["player_id"].to_numpy())
        meio = pd.DataFrame(u).groupby(grupos.to_numpy()).sum().to_numpy()
        G, n, k = meio.shape[0], X.shape[0], X.shape[1]
        V = XtXi @ (meio.T @ meio) @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
        for j, c in enumerate(covs, start=1):
            t_ = b[j] / np.sqrt(V[j, j])
            p_l = float(2 * stats.t.sf(abs(t_), G - 1))
            p_a = float(s["tests_null_effect"][c]["p_value"])
            linhas.append({"posicao": pos, "covariavel": c, "coef_lpm": float(b[j]), "p_lpm": p_l, "p_aalen": p_a,
                           "bhat_fim": float(cum.loc[cum.index.max(), c])})
        resumo[pos] = {"max_dif_ols_aalen": maxdif, "janela": [ini, fim], "n_covariaveis": len(covs)}
    t = pd.DataFrame(linhas)
    t["mesmo_lado_5pct"] = (t.p_lpm < 0.05) == (t.p_aalen < 0.05)
    SAIDA.mkdir(parents=True, exist_ok=True)
    t.to_csv(SAIDA / "aalen_discreto_lpm.csv", index=False)
    resumo["total"] = {"n": int(len(t)), "mesmo_lado_5pct": int(t.mesmo_lado_5pct.sum()), "max_dif": max(resumo[p]["max_dif_ols_aalen"] for p in ("QB", "RB", "WR"))}
    (SAIDA / "aalen_discreto_resumo.json").write_text(json.dumps(resumo, indent=2), encoding="utf-8")
    print(json.dumps(resumo, indent=1))


if __name__ == "__main__":
    main()
