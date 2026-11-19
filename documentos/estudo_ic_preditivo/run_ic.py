#!/usr/bin/env python3
"""Intervalos de confianca e comparacao pareada das metricas preditivas.

O pacote oficial compara Cox, GBSA e RSF pela media dos quatro folds, sem IC e
sem teste pareado. Este passo fecha a lacuna **sem tocar nos numeros
publicados**: reconstroi $C$, IBS e AUC media pelas mesmas formulas do `sksurv`
e exige igualdade com `metricas_walkforward_folds.csv` antes de reamostrar
(`auditoria/replica_metricas.csv`).

Estimando do IC: a media das metricas dos folds oficiais, o mesmo da escrita.

Reamostragem oficial: **bootstrap por cluster (jogador)**. O mesmo jogador entra
em mais de um fold de teste, de modo que as quatro metricas por fold nao sao
independentes. A unidade sorteada e o jogador, com reposicao, sobre o conjunto
de jogadores da posicao; cada jogador sorteado leva todas as suas linhas, em
todos os folds em que aparece (Davison e Hinkley, 1997, sec. 3.8; Field e
Welsh, 2007). O esquema que sorteia jogadores **dentro de cada fold e de forma
independente entre folds** fica como comparacao em
`*_intra_fold.csv`: ela ignora essa correlacao e tende a estreitar o IC.

Nos dois esquemas a composicao dos folds fica fixa: o IC descreve a incerteza
sobre a amostra de jogadores, nao sobre a escolha das temporadas. A comparacao
entre modelos usa os mesmos sorteios (numeros aleatorios comuns), o que torna o
IC da diferenca pareado por construcao. A diferenca reportada e a estimativa
pontual; o IC e o percentil da distribuicao bootstrap. `p_bootstrap_aproximado`
usa caudas com empates em ambas e correcao finita, 2 * min((n(delta<=tol)+1)/(B+1),
(n(delta>=-tol)+1)/(B+1)), limitado a 1. Empates numericos usam tolerancia absoluta
1e-12. Esse resumo bootstrap nao e um teste exato.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import SurvivalFunctionEstimator
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "Predição" / "shared"))

from tgb_protocol import valid_horizon_mask  # noqa: E402

ESTUDO = ROOT / "documentos" / "estudo_modelos_finais" / "resultados"
HERE = Path(__file__).resolve().parent
RES = HERE / "resultados"
AUD = RES / "auditoria"

POSICOES = ("QB", "RB", "WR")
MODELOS = ("cox", "gbsa", "rsf")
METRICAS = ("c_index", "ibs", "mean_time_auc")
# (melhor, pior): a diferenca e sempre "melhor menos pior".
PARES = (("gbsa", "cox"), ("rsf", "cox"), ("gbsa", "rsf"))
FOLDS_OFICIAIS = (2018, 2019, 2020, 2021)
SEED = 42
B_BOOT = 2000
ALFA = 5.0
# Abaixo disso a replica e a implementacao publicada sao a mesma coisa.
TOL_REPLICA = 1e-9
TOL_EMPATE = 1e-12


def auc_medio(T: np.ndarray, E: np.ndarray, risco: np.ndarray, tv: np.ndarray, ipcw: np.ndarray) -> float:
    """AUC media do `sksurv`, vetorizada no tempo.

    Reproduz `cumulative_dynamic_auc`, inclusive com empates no escore (a
    verificacao de replica cobre o caso real): ROC por cumsum na ordem do escore,
    trapezio em (fp, tp) e media ponderada pela sobrevivencia observada no teste.
    """
    if len(tv) == 0 or not np.any(E):
        return float("nan")
    o = np.argsort(-risco, kind="stable")  # decrescente em risco, como o sksurv
    T_o, E_o, w_o = T[o], E[o], ipcw[o]
    caso = (T_o[:, None] <= tv[None, :]) & E_o[:, None]
    ctrl = T_o[:, None] > tv[None, :]
    n_ctrl = ctrl.sum(axis=0)
    tp_total = (caso * w_o[:, None]).sum(axis=0)
    if np.any(n_ctrl == 0) or np.any(tp_total <= 0):
        return float("nan")
    tp = np.cumsum(caso * w_o[:, None], axis=0) / tp_total
    fp = np.cumsum(ctrl, axis=0) / n_ctrl
    # Empates no escore: a curva ROC so ganha um ponto no fim de cada grupo empatado,
    # como em `cumulative_dynamic_auc` (sem isto, escores repetidos contam degraus a mais).
    fim_de_grupo = np.r_[risco[o][1:] != risco[o][:-1], True]
    tp, fp = tp[fim_de_grupo], fp[fim_de_grupo]
    zero = np.zeros((1, len(tv)))
    tp = np.vstack([zero, tp])
    fp = np.vstack([zero, fp])
    # trapezio em x=fp ao longo do eixo 0 (equivale a np.trapezoid(tp, fp, axis=0))
    scores = ((fp[1:] - fp[:-1]) * (tp[1:] + tp[:-1]) / 2.0).sum(axis=0)
    km = SurvivalFunctionEstimator().fit(Surv.from_arrays(E.astype(bool), T))
    s = km.predict_proba(tv)
    d = -np.diff(np.r_[1.0, s])
    denom = 1.0 - s[-1]
    if denom <= 0:
        return float("nan")
    return float((scores * d).sum() / denom)


def metricas_rapidas(T, E, risco, S, t, g_t, g_sub) -> dict:
    """C, IBS e AUC media pelas formulas do `sksurv.evaluate_fold`."""
    Eb = E.astype(bool)
    y = Surv.from_arrays(Eb, T)
    mask = valid_horizon_mask(y, t)
    row = {"c_index": np.nan, "ibs": np.nan, "mean_time_auc": np.nan, "n_horizontes": int(mask.sum())}
    if len(T) < 5 or int(Eb.sum()) < 1:
        return row
    try:
        row["c_index"] = float(concordance_index_censored(Eb, T, risco)[0])
    except Exception:
        pass
    if row["n_horizontes"] == 0:
        return row
    tv = t[mask]
    Sv = S[:, mask]
    # O sksurv troca probabilidade zero de censura por infinito (peso nulo).
    gt = np.where(g_t[mask] == 0, np.inf, g_t[mask])
    gs = np.where(g_sub == 0, np.inf, g_sub)
    caso = (T[:, None] <= tv[None, :]) & Eb[:, None]
    ctrl = T[:, None] > tv[None, :]
    contrib = np.square(Sv) * caso / gs[:, None] + np.square(1.0 - Sv) * ctrl / gt[None, :]
    bs = contrib.mean(axis=0)
    if len(tv) >= 2:
        row["ibs"] = float(np.trapezoid(bs, tv) / (tv[-1] - tv[0]))
    ipcw = np.where(Eb, 1.0 / gs, 0.0)
    row["mean_time_auc"] = auc_medio(T, Eb, risco, tv, ipcw)
    return row


def carrega_celulas() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Le a matriz de sobrevivencia, a grade e as metricas publicadas."""
    mat = pd.read_csv(ESTUDO / "oof_surv_matrix.csv")
    grade = pd.read_csv(ESTUDO / "oof_times_grid.csv")
    folds = pd.read_csv(ESTUDO / "metricas_walkforward_folds.csv")
    celulas: dict[tuple[str, int], dict] = {}
    for (pos, fold), bloco in mat.groupby(["posicao", "teste_em_temporada"], sort=True):
        g = (
            grade[(grade.posicao == pos) & (grade.teste_em_temporada == fold)]
            .drop_duplicates(subset=["j"])
            .sort_values("j")
        )
        t = g["t"].to_numpy(dtype=float)
        g_t = g["g_t"].to_numpy(dtype=float)
        cols = [f"s{j}" for j in range(len(t))]
        modelos: dict[str, dict] = {}
        referencia: np.ndarray | None = None
        for modelo, sub in bloco.groupby("modelo", sort=True):
            sub = sub.sort_values("player_id", kind="stable")
            ids = sub["player_id"].to_numpy()
            # O bootstrap pareado exige que a linha i seja o mesmo jogador nos
            # dois modelos; a ordenacao por player_id garante isso, e o check
            # abaixo reprova o contrario.
            if referencia is None:
                referencia = ids
            elif not np.array_equal(referencia, ids):
                raise SystemExit(f"jogadores desalinhados entre modelos em {pos} {fold} ({modelo})")
            modelos[modelo] = {
                "player_id": ids,
                "T": sub["tempo"].to_numpy(dtype=float),
                "E": sub["evento"].to_numpy(dtype=int),
                "risco": sub["risk"].to_numpy(dtype=float),
                "S": sub[cols].to_numpy(dtype=float),
                "g_sub": sub["g_subject"].to_numpy(dtype=float),
                "t": t,
                "g_t": g_t,
            }
        celulas[(str(pos), int(fold))] = modelos
    return celulas, mat, folds


def checa_replica(celulas: dict, folds: pd.DataFrame) -> pd.DataFrame:
    """Exige que a replica reproduza as metricas publicadas, celula a celula."""
    linhas = []
    for (pos, fold), modelos in sorted(celulas.items()):
        for modelo, d in sorted(modelos.items()):
            rap = metricas_rapidas(d["T"], d["E"], d["risco"], d["S"], d["t"], d["g_t"], d["g_sub"])
            pub = folds[
                (folds.posicao == pos)
                & (folds.modelo == modelo)
                & (folds.teste_em_temporada == fold)
                & (folds.protocolo == "walkforward")
            ]
            if pub.empty:
                continue
            pub = pub.iloc[0]
            for metrica in METRICAS:
                a, b = float(rap[metrica]), float(pub[metrica])
                ambos_nan = not np.isfinite(a) and not np.isfinite(b)
                linhas.append(
                    {
                        "posicao": pos,
                        "teste_em_temporada": fold,
                        "modelo": modelo,
                        "metrica": metrica,
                        "replica": a,
                        "publicado": b,
                        "dif_abs": 0.0 if ambos_nan else abs(a - b),
                        "ok": bool(ambos_nan or abs(a - b) < TOL_REPLICA),
                        "n_horizontes": int(rap["n_horizontes"]),
                    }
                )
    return pd.DataFrame(linhas)


def ic_percentil(amostras: np.ndarray) -> tuple[float, float]:
    return (
        float(np.percentile(amostras, ALFA / 2.0)),
        float(np.percentile(amostras, 100.0 - ALFA / 2.0)),
    )


def p_bootstrap_aproximado(deltas: np.ndarray, atol: float = TOL_EMPATE) -> float:
    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("deltas must be a non-empty finite vector")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative")
    lower = int(np.count_nonzero(values <= atol))
    upper = int(np.count_nonzero(values >= -atol))
    return float(min(1.0, 2.0 * min((lower + 1) / (values.size + 1), (upper + 1) / (values.size + 1))))


def holm_adjust(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("p-values must be a non-empty finite vector")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be in [0, 1]")
    order = np.argsort(values, kind="stable")
    ranks = np.arange(values.size, dtype=float)
    ordered_adjusted = np.minimum(1.0, np.maximum.accumulate((values.size - ranks) * values[order]))
    adjusted = np.empty_like(ordered_adjusted)
    adjusted[order] = ordered_adjusted
    return adjusted, adjusted <= alpha


def _sorteio_intra_fold(rng, celulas, pos, folds_usados, presentes) -> dict[int, np.ndarray]:
    """Indices por fold, sorteados de forma independente em cada fold."""
    idx = {}
    for fold in folds_usados:
        if not presentes[fold]:
            continue
        n = len(celulas[(pos, fold)][presentes[fold][0]]["T"])
        idx[fold] = rng.integers(0, n, n)
    return idx


def _sorteio_cluster(rng, jogadores, posicao_no_fold, folds_usados) -> dict[int, np.ndarray]:
    """Indices por fold a partir de um unico sorteio de jogadores (com reposicao).

    Um jogador sorteado k vezes entra k vezes em cada fold em que aparece; um
    jogador ausente de um fold nao contribui para ele. O tamanho de cada fold
    reamostrado varia em torno do original, como e proprio do bootstrap por cluster.
    """
    sorteados = jogadores[rng.integers(0, len(jogadores), len(jogadores))]
    idx = {}
    for fold in folds_usados:
        mapa = posicao_no_fold[fold]
        linhas = [mapa[j] for j in sorteados if j in mapa]
        idx[fold] = np.asarray(linhas, dtype=int)
    return idx


def bootstrap(celulas: dict, esquema: str = "cluster") -> tuple[pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame]:
    """IC por metrica e IC/p da diferenca pareada, por posicao."""
    if esquema not in ("cluster", "intra_fold"):
        raise ValueError(esquema)
    rng = np.random.default_rng(SEED)
    linhas_ic: list[dict] = []
    linhas_par: list[dict] = []
    linhas_replicas: list[dict] = []
    notas: list[dict] = []
    folds_usados = sorted({f for (_, f) in celulas if f in FOLDS_OFICIAIS})
    for pos in POSICOES:
        modelos = sorted({m for (p, f) in celulas if p == pos for m in celulas[(p, f)]})
        # Estimativa pontual: media das metricas dos folds oficiais.
        base = {m: {k: [] for k in METRICAS} for m in modelos}
        for fold in folds_usados:
            for m in modelos:
                if m not in celulas[(pos, fold)]:
                    continue
                d = celulas[(pos, fold)][m]
                rap = metricas_rapidas(d["T"], d["E"], d["risco"], d["S"], d["t"], d["g_t"], d["g_sub"])
                for k in METRICAS:
                    base[m][k].append(float(rap[k]))
        pontual = {m: {k: float(np.mean(v)) if v else float("nan") for k, v in base[m].items()} for m in modelos}
        amostras = {m: {k: [] for k in METRICAS} for m in modelos}
        difs = {par: [] for par in PARES if par[0] in modelos and par[1] in modelos}
        descartadas = 0
        presentes = {f: [m for m in modelos if m in celulas[(pos, f)]] for f in folds_usados}
        posicao_no_fold = {
            f: {j: i for i, j in enumerate(celulas[(pos, f)][presentes[f][0]]["player_id"])}
            for f in folds_usados
            if presentes[f]
        }
        jogadores = np.array(sorted({j for mapa in posicao_no_fold.values() for j in mapa}))
        for b in range(B_BOOT):
            por_modelo = {m: {k: [] for k in METRICAS} for m in modelos}
            valido = True
            # Mesmos indices para todos os modelos: IC da diferenca e pareado.
            if esquema == "cluster":
                indices = _sorteio_cluster(rng, jogadores, posicao_no_fold, folds_usados)
            else:
                indices = _sorteio_intra_fold(rng, celulas, pos, folds_usados, presentes)
            for fold in folds_usados:
                presente = presentes[fold]
                if not presente:
                    continue
                idx = indices[fold]
                for m in presente:
                    d = celulas[(pos, fold)][m]
                    rap = metricas_rapidas(
                        d["T"][idx], d["E"][idx], d["risco"][idx], d["S"][idx], d["t"], d["g_t"], d["g_sub"][idx]
                    )
                    for k in METRICAS:
                        por_modelo[m][k].append(float(rap[k]))
            for m in modelos:
                for k in METRICAS:
                    v = [x for x in por_modelo[m][k] if np.isfinite(x)]
                    if len(v) != len(folds_usados):
                        valido = False
            if not valido:
                descartadas += 1
                continue
            for m in modelos:
                for k in METRICAS:
                    amostras[m][k].append(float(np.mean(por_modelo[m][k])))
            for par in difs:
                melhor, pior = par
                d_c = float(np.mean(por_modelo[melhor]["c_index"]) - np.mean(por_modelo[pior]["c_index"]))
                difs[par].append(d_c)
                linhas_replicas.append(
                    {
                        "posicao": pos,
                        "esquema": esquema,
                        "par": f"{par[0]}-{par[1]}",
                        "replicacao": b + 1,
                        "delta_c": d_c,
                    }
                )
        for m in modelos:
            for k in METRICAS:
                arr = np.array(amostras[m][k], dtype=float)
                if arr.size == 0:
                    continue
                lo, hi = ic_percentil(arr)
                linhas_ic.append(
                    {
                        "posicao": pos,
                        "modelo": m,
                        "metrica": k,
                        "estimativa": pontual[m][k],
                        "ic_baixo": lo,
                        "ic_alto": hi,
                        "n_boot": int(arr.size),
                        "descartadas": int(descartadas),
                    }
                )
        for par, arr in difs.items():
            a = np.array(arr, dtype=float)
            if a.size == 0:
                continue
            lo, hi = ic_percentil(a)
            prop_pos = float((a > 0).mean())
            p = p_bootstrap_aproximado(a)
            p_simples = max(2.0 * min(prop_pos, 1.0 - prop_pos), 1.0 / a.size)
            linhas_par.append(
                {
                    "posicao": pos,
                    "par": f"{par[0]}-{par[1]}",
                    "delta_c": pontual[par[0]]["c_index"] - pontual[par[1]]["c_index"],
                    "delta_media_boot": float(a.mean()),
                    "ic_baixo": lo,
                    "ic_alto": hi,
                    "p_bootstrap_aproximado": p,
                    "p_bilateral": p,
                    "p_bilateral_simples": p_simples,
                    "prop_delta_positivo": prop_pos,
                    "n_boot": int(a.size),
                }
            )
        notas.append(
            {
                "posicao": pos,
                "esquema": esquema,
                "modelos": modelos,
                "folds": folds_usados,
                "jogadores_distintos": int(len(jogadores)),
                "descartadas": descartadas,
            }
        )
    return pd.DataFrame(linhas_ic), pd.DataFrame(linhas_par), notas, pd.DataFrame(linhas_replicas)


def sensibilidade_folds(folds: pd.DataFrame) -> pd.DataFrame:
    """O campeao muda se o conjunto de folds mudar?"""
    wf = folds[(folds.protocolo == "walkforward") & folds.c_index.notna()].copy()
    disponiveis = sorted(int(s) for s in wf.teste_em_temporada.unique())
    conjuntos: dict[str, list[int]] = {
        "oficial_2018_2021": [s for s in disponiveis if s in FOLDS_OFICIAIS],
        "disponivel_2018_2023": disponiveis,
    }
    for s in disponiveis:
        resto = [x for x in disponiveis if x != s]
        conjuntos[f"sem_{s}"] = resto
    linhas = []
    for nome, seas in conjuntos.items():
        sub = wf[wf.teste_em_temporada.isin(seas)]
        for pos in POSICOES:
            medias = sub[sub.posicao == pos].groupby("modelo").c_index.mean()
            medias = medias[medias.index.isin(MODELOS)]
            if medias.empty:
                continue
            for modelo, valor in medias.items():
                linhas.append(
                    {
                        "conjunto": nome,
                        "n_folds": len(set(seas) & set(sub[sub.posicao == pos].teste_em_temporada)),
                        "posicao": pos,
                        "modelo": modelo,
                        "c_medio": float(valor),
                        "campeao": bool(modelo == medias.idxmax()),
                    }
                )
    return pd.DataFrame(linhas)


def main() -> None:
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    AUD.mkdir(parents=True, exist_ok=True)
    celulas, mat, folds = carrega_celulas()
    print(f"celulas carregadas: {len(celulas)} (pos x fold); linhas de matriz: {len(mat)}", flush=True)

    rep = checa_replica(celulas, folds)
    rep.to_csv(AUD / "replica_metricas.csv", index=False)
    n_ok = int(rep.ok.sum())
    pior = float(rep.dif_abs.max()) if len(rep) else 0.0
    print(f"replica vs publicado: {n_ok}/{len(rep)} celulas-metria ok (pior dif {pior:.2e})", flush=True)
    if n_ok != len(rep):
        ruins = rep[~rep.ok]
        raise SystemExit(
            "Replica divergente do publicado:\n" + ruins.head(10).to_string(index=False)
        )

    ic, par, notas, replicas_cluster = bootstrap(celulas, "cluster")
    ic.to_csv(RES / "ic_metricas.csv", index=False)
    if len(par) != 9 or par.duplicated(["posicao", "par"]).any():
        raise SystemExit("Holm exige exatamente os nove contrastes posição-par previstos")
    par["p_holm"], par["rejeita_5pct_holm"] = holm_adjust(
        par["p_bootstrap_aproximado"].to_numpy(dtype=float)
    )
    par.to_csv(RES / "delta_c_pareado.csv", index=False)
    ic_intra, par_intra, notas_intra, replicas_intra = bootstrap(celulas, "intra_fold")
    ic_intra.to_csv(RES / "ic_metricas_intra_fold.csv", index=False)
    par_intra.to_csv(RES / "delta_c_pareado_intra_fold.csv", index=False)
    pd.concat([replicas_cluster, replicas_intra], ignore_index=True).to_csv(
        RES / "bootstrap_delta_replicas.csv", index=False
    )
    comp = par.merge(par_intra, on=["posicao", "par"], suffixes=("_cluster", "_intra"))
    comp["amplitude_cluster"] = comp.ic_alto_cluster - comp.ic_baixo_cluster
    comp["amplitude_intra"] = comp.ic_alto_intra - comp.ic_baixo_intra
    comp["razao_amplitude"] = comp.amplitude_cluster / comp.amplitude_intra
    comp.to_csv(RES / "comparacao_esquemas_bootstrap.csv", index=False)
    sens = sensibilidade_folds(folds)
    sens.to_csv(RES / "sensibilidade_folds.csv", index=False)

    resumo = {
        "seed": SEED,
        "n_boot": B_BOOT,
        "alfa": ALFA,
        "estimando": "media das metricas dos folds oficiais (2018-2021)",
        "holm": "Ajuste step-down de Holm sobre os nove p_bootstrap_aproximado do esquema cluster; p_bilateral e alias de p_bootstrap_aproximado e p_bilateral_simples usa a regra 2*min(prop, 1-prop), apenas para comparacao.",
        "p_bootstrap_aproximado": "Caudas bilaterais com empates em ambas, tolerancia absoluta 1e-12 e correcao finita (count+1)/(B+1); nao e teste exato.",
        "replicas_bootstrap": "bootstrap_delta_replicas.csv preserva cada delta de C nos esquemas cluster e intra_fold para auditoria de IC, p e Holm.",
        "reamostragem": (
            "bootstrap por cluster: jogador sorteado com reposicao sobre os quatro folds, "
            "levando todas as suas linhas; numeros aleatorios comuns entre modelos"
        ),
        "comparacao": "intra_fold (jogador sorteado dentro de cada fold, independente entre folds)",
        "replica_ok": n_ok,
        "replica_total": int(len(rep)),
        "replica_pior_dif": pior,
        "notas_por_posicao": notas,
        "notas_intra_fold": notas_intra,
        "como_ler": (
            "O IC cobre a amostra de jogadores com a composicao dos folds fixa; nao cobre a "
            "escolha das temporadas. A diferenca de C e a estimativa pontual; o IC e o p vem "
            "da distribuicao bootstrap pareada (mesmos jogadores sorteados nos dois modelos)."
        ),
        "segundos": round(time.time() - t0, 1),
    }
    (RES / "resumo_ic.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== C-index com IC 95% ===", flush=True)
    c = ic[ic.metrica == "c_index"]
    for _, r in c.iterrows():
        print(
            f"  {r.posicao} {r.modelo:<5} {r.estimativa:.3f} [{r.ic_baixo:.3f}; {r.ic_alto:.3f}]",
            flush=True,
        )
    print("\n=== diferenca de C (pareada) ===", flush=True)
    for _, r in par.sort_values(["posicao", "par"]).iterrows():
        print(
            f"  {r.posicao} {r.par:<10} {r.delta_c:+.3f} [{r.ic_baixo:+.3f}; {r.ic_alto:+.3f}]"
            f"  p_aprox={r.p_bootstrap_aproximado:.3f} p_Holm={r.p_holm:.3f}",
            flush=True,
        )
    trocas = sens[sens.campeao].groupby("posicao").campeao.nunique()
    print("\n=== campeao por conjunto de folds ===", flush=True)
    for pos in POSICOES:
        sub = sens[(sens.posicao == pos) & sens.campeao]
        print(f"  {pos}: " + ", ".join(f"{r.conjunto}->{r.modelo}" for _, r in sub.iterrows()), flush=True)
    print(f"\nfim {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
