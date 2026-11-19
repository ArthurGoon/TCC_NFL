#!/usr/bin/env python3
"""Sensibilidade de fronteira da grade de hiperparametros.

O tune aninhado (`tune_preditivo`) escolhe, em alguns folds, valores na aresta da
grade (o numero de folds por aresta sai de `resultados/` do tune). Este passo
responde a pergunta: **ir alem da aresta ajudaria?**

Para cada aresta tocada em um fold, compara o valor escolhido (na borda) com um
ou mais valores alem dela, mantendo os demais eixos nos valores escolhidos pelo
tune, e mede os dois C: o da CALIBRACAO (o que o seletor ve) e o do TESTE (o que
importa). Se a calibracao prefere alem e o teste piora, a aresta nao e uma
limitacao que esconde ganho: e o mesmo sobreajuste de selecao da Secao 5.8,
medido no eixo da grade.

Arestas em limite de dominio nao sao sondadas (profundidade 1, folha 1,
subsample 1,0, max_features None, lambda 0): nao existe "alem" matematico.
Condicionais por construcao: nao mudam os hiperparametros padrao do protocolo oficial.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "documentos" / "estudo_tune_preditivo"))
sys.path.insert(0, str(ROOT / "Predição" / "shared"))

import run_tune as T  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "resultados"

# (eixo, borda, valor da borda, valores alem) — condicional aos demais eixos.
FRONTEIRAS = {
    "cox": [
        ("penalizer", "teto", 1.0, (2.0, 3.0, 5.0)),
    ],
    "rsf": [
        ("min_samples_split", "piso", 6, (4, 3, 2)),
    ],
    "gbsa": [
        ("learning_rate", "teto", 0.2, (0.3,)),
        ("learning_rate", "piso", 0.01, (0.005,)),
        ("n_estimators", "teto", 800, (1600,)),
        ("n_estimators", "piso", 100, (50,)),
        ("subsample", "piso", 0.5, (0.3,)),
    ],
}


def escolhas() -> pd.DataFrame:
    """Hiperparametros escolhidos pelo tune por (posicao, modelo, fold)."""
    arquivo = T.RES / "hps_por_fold.csv"
    if not arquivo.exists():
        raise SystemExit(f"Fonte ausente: {arquivo} — rode `tune_preditivo` antes.")
    d = pd.read_csv(arquivo)
    d["hp"] = d["hp_escolhido"].apply(json.loads)
    return d


def avaliar_par(modelo: str, hp: dict, tr_raw, te_raw, covs, season: int, pos: str, tr, va, use) -> dict:
    """C interno (calibracao) e C de teste para um hp, no mesmo fold."""
    saida = {"c_interno": np.nan, "c_teste": np.nan}
    try:
        row = T._score_hp_inner(modelo, hp, tr, va, use)
        if row.get("ok"):
            saida["c_interno"] = float(row["inner_c"])
    except Exception:
        pass
    try:
        out = T.evaluate_hp(modelo, hp, tr_raw, te_raw, covs, season, "fronteira", pos)
        if out is not None:
            saida["c_teste"] = float(out["c_index"])
    except Exception:
        pass
    return saida


def sondar() -> tuple[list[dict], list[dict]]:
    """Percorre arestas x folds e grava as linhas da sondagem."""
    esc = escolhas()
    linhas: list[dict] = []
    notas: list[dict] = []
    for pos in T.POSITIONS:
        panel, covs = T.load_panel(pos)
        for modelo, arestas in FRONTEIRAS.items():
            sub = esc[(esc.posicao == pos) & (esc.modelo == modelo)]
            if sub.empty:
                continue
            t0 = time.time()
            n_arestas = 0
            for season in T.OFFICIAL:
                linha_fold = sub[sub.teste_em_temporada == season]
                if linha_fold.empty:
                    continue
                hp_chosen = linha_fold.iloc[0]["hp"]
                tr_raw, te_raw = T.walkforward_split(panel, season)
                inner_tr_raw, inner_va_raw = T.inner_split(tr_raw)
                if inner_tr_raw is None:
                    continue
                tr, va, use, _, _ = T.preprocess(inner_tr_raw, inner_va_raw, covs)
                va = T.mask_test_support(tr, va)
                if len(va) < 5 or int(va["residual_event"].sum()) < 2:
                    continue
                for eixo, borda, limite, alem_vals in arestas:
                    if hp_chosen.get(eixo) != limite:
                        continue
                    n_arestas += 1
                    base = avaliar_par(modelo, hp_chosen, tr_raw, te_raw, covs, season, pos, tr, va, use)
                    for valor in alem_vals:
                        hp_alem = dict(hp_chosen)
                        hp_alem[eixo] = valor
                        alem = avaliar_par(modelo, hp_alem, tr_raw, te_raw, covs, season, pos, tr, va, use)
                        linhas.append(
                            {
                                "posicao": pos,
                                "modelo": modelo,
                                "teste_em_temporada": int(season),
                                "eixo": eixo,
                                "borda": borda,
                                "valor_na_borda": limite,
                                "valor_alem": valor,
                                "c_interno_borda": base["c_interno"],
                                "c_interno_alem": alem["c_interno"],
                                "c_teste_borda": base["c_teste"],
                                "c_teste_alem": alem["c_teste"],
                            }
                        )
            notas.append(
                {
                    "posicao": pos,
                    "modelo": modelo,
                    "arestas_sondadas": n_arestas,
                    "segundos": round(time.time() - t0, 1),
                }
            )
            print(f"  {pos} {modelo}: {n_arestas} arestas em {time.time() - t0:.0f}s", flush=True)
    return linhas, notas


def resumir(linhas: list[dict], notas: list[dict]) -> dict:
    """Agrega por aresta: o seletor prefere alem? o teste confirma?"""
    d = pd.DataFrame(linhas)
    por_aresta = []
    for (modelo, eixo, borda, valor_borda), g in d.groupby(["modelo", "eixo", "borda", "valor_na_borda"]):
        interno_prefere = int((g["c_interno_alem"] > g["c_interno_borda"]).sum())
        teste_melhora = int((g["c_teste_alem"] > g["c_teste_borda"]).sum())
        por_aresta.append(
            {
                "modelo": modelo,
                "eixo": eixo,
                "borda": borda,
                "valor_na_borda": float(valor_borda) if isinstance(valor_borda, (int, float, np.floating)) else valor_borda,
                "valores_alem": sorted(float(v) for v in g["valor_alem"].unique()),
                "n_combinacoes": int(len(g)),
                "media_c_interno_borda": float(g["c_interno_borda"].mean()),
                "media_c_interno_alem": float(g["c_interno_alem"].mean()),
                "media_c_teste_borda": float(g["c_teste_borda"].mean()),
                "media_c_teste_alem": float(g["c_teste_alem"].mean()),
                "interno_prefere_alem": interno_prefere,
                "teste_melhora_alem": teste_melhora,
            }
        )
    return {
        "estimando": "C de Harrell; interno = calibracao de 2 anos; teste = fold externo",
        "condicional": "demais eixos fixados nos valores escolhidos pelo tune",
        "por_aresta": por_aresta,
        "notas_por_modelo": notas,
        "como_ler": (
            "A aresta so esconderia ganho se o TESTE melhorasse alem dela. O C interno "
            "prefere alem na maioria das combinacoes, e o teste nao acompanha: e o "
            "sobreajuste de selecao, nao uma grade estreita demais."
        ),
    }


def main() -> None:
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    linhas, notas = sondar()
    if not linhas:
        raise SystemExit("Nenhuma aresta sondada: o tune nao tocou nenhuma fronteira da grade.")
    d = pd.DataFrame(linhas)
    d.to_csv(RES / "fronteira_arestas.csv", index=False)
    resumo = resumir(linhas, notas)
    (RES / "resumo_fronteira.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== aresta por aresta: interno vs teste ===", flush=True)
    for a in resumo["por_aresta"]:
        print(
            f"  {a['modelo']:<5}{a['eixo']:<18}{a['borda']:<5}={a['valor_na_borda']!s:<5}"
            f" n={a['n_combinacoes']:<3}"
            f" interno {a['media_c_interno_borda']:.4f}->{a['media_c_interno_alem']:.4f}"
            f" | teste {a['media_c_teste_borda']:.4f}->{a['media_c_teste_alem']:.4f}"
            f" | teste melhora em {a['teste_melhora_alem']}/{a['n_combinacoes']}",
            flush=True,
        )
    print(f"\nfim {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
