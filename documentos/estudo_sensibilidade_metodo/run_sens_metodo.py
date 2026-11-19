#!/usr/bin/env python3
"""Duas sensibilidades metodologicas rapidas (ramo inferencial e ramo preditivo).

1. Penalizacao do Cox inferencial. O Cox oficial (passos 11 a 13) e ajustado sem penalizacao,
   depois de retirar as colunas nao identificaveis (covariaveis_identificaveis_cox). Aqui a mesma cadeia (imputacao,
   padronizacao, filtro de colinearidade, backward, teto EPV) e refeita com
   lambda em {0; 0,001; 0,01; 0,1}. Com lambda = 0 a replica tem de devolver o
   conjunto e os HRs oficiais (`cox_hr_{POS}.csv`); a checagem reprova o contrario.
2. Janela de congelamento do X preditivo. O oficial filtra a colinearidade nas temporadas
   2012-2017. Aqui o filtro e refeito em 2012-2015, 2012-2016, 2012-2018, 2012-2019 e no
   painel inteiro; para cada janela que muda o X, o walk-forward oficial e refeito com o X
   alternativo e a diferenca de C e registrada.

Nada oficial e sobrescrito: as saidas ficam em `resultados/` deste estudo.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RES = HERE / "resultados"
RES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "Inferência" / "shared"))
sys.path.insert(0, str(ROOT / "Predição" / "shared"))

from lifelines import CoxTimeVaryingFitter  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tgb_protocol import (  # noqa: E402
    META_COLS,
    OFFICIAL_SEASONS,
    TEST_SEASONS,
    build_residual_snapshot_panel,
    candidate_times,
    load_panel,
    mask_test_support,
    preprocess,
    walkforward_split,
)
from survival_utils import (  # noqa: E402
    apply_imputation_stats,
    filter_collinear_covariates,
    fit_imputation_stats,
    force_missing_flags,
    covariaveis_identificaveis_cox,
)

POSICOES = ("QB", "RB", "WR")
# Escala do lifelines: a penalidade e n * lambda * beta^2 / 2, com n = linhas do painel
# (cerca de 850 a 2 200), de modo que lambda = 0,1 ja equivale a uma ridge forte.
LAMBDAS = (0.0, 0.001, 0.01, 0.1)
LAMBDA_OFICIAL = 0.0
PREP = ROOT / "Inferência" / "Preparação Modelagem" / "resultados"
COX = ROOT / "Inferência" / "Modelagem de COX" / "resultados"


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


# ------------------------------------------------------------------ 1. lambda inferencial
def prepara(pos: str) -> tuple[pd.DataFrame, list[str]]:
    """Cells 4 a 8 do passo 11: imputacao, escala e filtro de colinearidade."""
    df_cox = pd.read_pickle(PREP / f"df_cox_{pos}.pkl")
    id_cols, time_cols = ["player_id", "player_name", "season"], ["start", "stop", "event", "entry"]
    cov_cols = [c for c in df_cox.columns if c not in id_cols + time_cols and c != "draft_round"]
    stats = fit_imputation_stats(df_cox, cov_cols)
    df_model = apply_imputation_stats(df_cox, cov_cols, stats)
    df_model[cov_cols] = StandardScaler().fit_transform(df_model[cov_cols])
    df_fit = df_model[["player_id", "start", "stop", "event"] + cov_cols].copy()
    # Mesma ordem dos passos 11-13: identificaveis saem ANTES do filtro |r|>0,85.
    ident = covariaveis_identificaveis_cox(df_fit, cov_cols, verbose=False)
    filtradas = filter_collinear_covariates(df_fit, ident, threshold=0.85, verbose=False)
    return df_fit, covariaveis_identificaveis_cox(df_fit, force_missing_flags(filtradas, df_fit.columns.tolist()), verbose=False)


def ajusta(df_fit, covs, lam):
    m = CoxTimeVaryingFitter(penalizer=lam)
    m.fit(df_fit[["player_id", "start", "stop", "event"] + list(covs)], id_col="player_id",
          start_col="start", stop_col="stop", event_col="event", show_progress=False)
    return m


def backward(df_fit, covs, lam, limiar=0.05):
    atuais = list(covs)
    while True:
        m = ajusta(df_fit, atuais, lam)
        p = m.summary["p"]
        if p.max() > limiar:
            atuais.remove(p.idxmax())
        else:
            return m, atuais


def sens_lambda() -> pd.DataFrame:
    linhas, conjuntos = [], []
    for pos in POSICOES:
        df_fit, covs = prepara(pos)
        n_ev = int(df_fit["event"].sum())
        max_vars = max(2, int(n_ev / 10))
        oficial = pd.read_csv(COX / f"cox_hr_{pos}.csv").set_index("covariate")
        for lam in LAMBDAS:
            try:
                m, final = backward(df_fit, covs, lam)
                if len(final) > max_vars:
                    final = m.summary["p"].nsmallest(max_vars).index.tolist()
                    m = ajusta(df_fit, final, lam)
                erro = ""
            except Exception as exc:  # noqa: BLE001
                final, m, erro = [], None, f"{type(exc).__name__}: {exc}"
            conjuntos.append({
                "posicao": pos, "lambda": lam, "n_final": len(final), "final": ";".join(sorted(final)),
                "jaccard_vs_oficial": jaccard(final, oficial.index), "erro": erro,
            })
            if m is None:
                continue
            s = m.summary
            for cov in s.index:
                linhas.append({"posicao": pos, "lambda": lam, "covariate": cov,
                               "HR": float(s.loc[cov, "exp(coef)"]), "p": float(s.loc[cov, "p"]),
                               "no_oficial": cov in oficial.index})
    return pd.DataFrame(conjuntos), pd.DataFrame(linhas)


# ------------------------------------------------------------- 2. janela do X preditivo
def sens_janela_x() -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = importlib.util.spec_from_file_location("covs38", ROOT / "pipeline" / "38_covs_preditivas.py")
    covs38 = importlib.util.module_from_spec(spec)
    sys.modules["covs38"] = covs38
    spec.loader.exec_module(covs38)
    janelas = {"2012-2015": (2012, 2015), "2012-2016": (2012, 2016), "2012-2017": (2012, 2017),
               "2012-2018": (2012, 2018), "2012-2019": (2012, 2019), "painel_inteiro": (2012, 2024)}
    linhas, x_por_janela = [], {}
    for pos in POSICOES:
        df = pd.read_pickle(ROOT / "Predição" / "shared" / f"df_cox_{pos}_features.pkl")
        cand = covs38.candidate_covariates(df)
        oficial = json.loads((ROOT / "Predição" / "shared" / f"covs_preditivas_{pos}.json").read_text(encoding="utf-8"))["covariates"]
        for nome, (a, b) in janelas.items():
            sel = list(filter_collinear_covariates(df[df["season"].between(a, b)], cand, threshold=0.85, verbose=False)) + covs38.indicadores_faltantes(df)
            x_por_janela[(pos, nome)] = sel
            linhas.append({"posicao": pos, "janela": nome, "n": len(sel), "jaccard_vs_oficial": jaccard(sel, oficial),
                           "sai": ";".join(sorted(set(oficial) - set(sel))), "entra": ";".join(sorted(set(sel) - set(oficial)))})
    estab = pd.DataFrame(linhas)

    # walk-forward com o X alternativo, so nas janelas que mudam o X de alguma posicao
    spec = importlib.util.spec_from_file_location("ofi_x", ROOT / "documentos" / "estudo_modelos_finais" / "run_oficial.py")
    ofi = importlib.util.module_from_spec(spec)
    sys.modules["ofi_x"] = ofi
    spec.loader.exec_module(ofi)
    ofi.RES = Path(tempfile.mkdtemp())
    efeitos = []
    for nome in ["2012-2017"] + [j for j in janelas if j != "2012-2017"]:
        if nome != "2012-2017" and estab[(estab.janela == nome)]["jaccard_vs_oficial"].min() == 1.0:
            continue

        def carrega_alt(pos, _nome=nome):
            covs = x_por_janela[(pos, _nome)]
            df = pd.read_pickle(ROOT / "Predição" / "shared" / f"df_cox_{pos}_features.pkl")
            return build_residual_snapshot_panel(df, covs, META_COLS), covs

        ofi.load_panel = carrega_alt
        folds, _ = ofi.walkforward()
        f = folds[(folds.protocolo == "walkforward") & (folds.oficial == 1) & (folds.modelo != "km")]
        for (pos, mod), g in f.groupby(["posicao", "modelo"]):
            efeitos.append({"janela": nome, "posicao": pos, "modelo": mod, "c_index": float(g.c_index.mean())})
    ef = pd.DataFrame(efeitos)
    base = ef[ef.janela == "2012-2017"].set_index(["posicao", "modelo"])["c_index"]
    ef["dif_c_vs_oficial"] = ef.apply(lambda r: r.c_index - base[(r.posicao, r.modelo)], axis=1)
    return estab, ef


# ----------------------------------------------- 3. saidas em 2023 tratadas como censura
def sens_saidas_2023() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quem nao joga 2024 aparece como saida em 2023, mas pode voltar depois (hiato).

    Nos anos anteriores o retorno e visivel; em 2023 nao. Aqui as saidas na temporada
    2023 viram censura, (1) no Cox inferencial, com o mesmo conjunto final e o mesmo
    lambda oficial, e (2) nos rotulos de teste do walk-forward, recalculando o C dos
    modelos ja ajustados (o ajuste nao muda: so o rotulo de avaliacao).
    """
    from sksurv.metrics import concordance_index_censored

    hrs = []
    for pos in POSICOES:
        df_fit, _ = prepara(pos)
        df_cox = pd.read_pickle(PREP / f"df_cox_{pos}.pkl")
        oficial = pd.read_csv(COX / f"cox_hr_{pos}.csv").set_index("covariate")
        covs = list(oficial.index)
        for versao in ("oficial", "saidas_2023_censuradas"):
            d = df_fit.copy()
            if versao != "oficial":
                d.loc[(df_cox["season"].to_numpy() == 2023) & (d["event"].to_numpy() == 1), "event"] = 0
            m = ajusta(d, covs, LAMBDA_OFICIAL)
            for cov in covs:
                hrs.append({"posicao": pos, "versao": versao, "covariate": cov,
                            "HR": float(m.summary.loc[cov, "exp(coef)"]), "p": float(m.summary.loc[cov, "p"]),
                            "n_eventos": int(d["event"].sum())})
    oof = pd.read_csv(ROOT / "documentos" / "estudo_modelos_finais" / "resultados" / "pred_oof_walkforward_2018_2021.csv")
    oof = oof[oof.modelo != "km"]
    ultima = oof["snapshot_season"] + oof["residual_duration_model"].round()
    linhas = []
    for versao in ("oficial", "saidas_2023_censuradas"):
        ev = oof["residual_event"].astype(bool).to_numpy().copy()
        if versao != "oficial":
            ev[(ultima == 2023).to_numpy() & ev] = False
        o = oof.assign(_ev=ev)
        for (pos, mod, fold), g in o.groupby(["posicao", "modelo", "snapshot_season"]):
            c = concordance_index_censored(g["_ev"].to_numpy(), g["residual_duration_model"].to_numpy(float), g["risk"].to_numpy(float))[0]
            linhas.append({"versao": versao, "posicao": pos, "modelo": mod, "fold": int(fold), "c": c})
    cs = pd.DataFrame(linhas).groupby(["versao", "posicao", "modelo"])["c"].mean().unstack("versao").reset_index()
    cs["dif"] = cs["saidas_2023_censuradas"] - cs["oficial"]
    return pd.DataFrame(hrs), cs


def main() -> None:
    conj, hr = sens_lambda()
    conj.to_csv(RES / "lambda_inferencial_selecao.csv", index=False)
    hr.to_csv(RES / "lambda_inferencial_hr.csv", index=False)
    # a replica com o lambda oficial reproduz o conjunto e os HRs publicados
    replica = {}
    for pos in POSICOES:
        oficial = pd.read_csv(COX / f"cox_hr_{pos}.csv").set_index("covariate")
        r = hr[(hr.posicao == pos) & (hr["lambda"] == LAMBDA_OFICIAL)].set_index("covariate")
        mesmo_conjunto = set(r.index) == set(oficial.index)
        dif = float((r.loc[oficial.index, "HR"] - oficial["HR"]).abs().max()) if mesmo_conjunto else float("nan")
        replica[pos] = {"mesmo_conjunto": mesmo_conjunto, "max_dif_hr": dif}
    estab, ef = sens_janela_x()
    estab.to_csv(RES / "janela_x_estabilidade.csv", index=False)
    ef.to_csv(RES / "janela_x_efeito_c.csv", index=False)
    resumo = {
        "replica_lambda_oficial": replica,
        "lambda_jaccard_min": {pos: float(conj[conj.posicao == pos].jaccard_vs_oficial.min()) for pos in POSICOES},
        "janela_x_jaccard_min": {pos: float(estab[estab.posicao == pos].jaccard_vs_oficial.min()) for pos in POSICOES},
        "janela_x_max_abs_dif_c": float(ef.dif_c_vs_oficial.abs().max()) if len(ef) else 0.0,
    }
    hr23, c23 = sens_saidas_2023()
    hr23.to_csv(RES / "saidas_2023_cox_inferencial.csv", index=False)
    c23.to_csv(RES / "saidas_2023_c_preditivo.csv", index=False)
    piv = hr23.pivot_table(index=["posicao", "covariate"], columns="versao", values="p")
    resumo["saidas_2023"] = {
        "covariaveis_que_perdem_significancia": [f"{i[0]} {i[1]}" for i, r in piv.iterrows()
                                                 if r["oficial"] < 0.05 <= r["saidas_2023_censuradas"]],
        "max_abs_dif_hr": float(hr23.pivot_table(index=["posicao", "covariate"], columns="versao", values="HR").diff(axis=1).abs().max().max()),
        "max_abs_dif_c": float(c23["dif"].abs().max()),
    }
    print(hr23.to_string(index=False))
    print(c23.round(4).to_string(index=False))
    (RES / "resumo_sens_metodo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(conj.to_string(index=False))
    print(estab.to_string(index=False))
    print(ef.round(4).to_string(index=False))
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    if not all(v["mesmo_conjunto"] and v["max_dif_hr"] < 1e-6 for v in replica.values()):
        raise SystemExit("a replica com lambda = 0 nao reproduz o Cox oficial")


if __name__ == "__main__":
    main()
