#!/usr/bin/env python3
"""Variacoes recentes de trajetoria: efeito de acrescenta-las aos dois ramos.

As cinco variacoes de t-2 para t-1 (piora na profundidade, queda na participacao, aumento de
jogos perdidos, queda na eficiencia e temporadas desde o hiato) nao fazem parte do painel.
Inferencia: selecao do Cox, Aalen e Cox-Aalen com elas entre as candidatas.
Predicao: walk-forward do modelo principal com elas e com a rodada do draft somadas ao X.

Saidas em Inferência/sensibilidade/resultados/variacoes_recentes_*.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
SAIDA = ROOT / "Inferência" / "sensibilidade" / "resultados"
COX = ROOT / "Inferência" / "Modelagem de COX" / "resultados"
AALEN = ROOT / "Inferência" / "Modelagem de Aalen"
OF = ROOT / "documentos" / "estudo_modelos_finais"
FEATURES = ROOT / "Predição" / "shared"
POS = ("QB", "RB", "WR")
VARIACOES = ["depth_worsened_lag", "offense_pct_declined_lag", "games_missed_increased_lag",
             "efficiency_declined_lag", "seasons_since_gap"]
NOVE = {"QB": ["passer_rating", "team_pass_attempts_share", "offense_pct"],
        "RB": ["games_missed", "depth_team", "observed_career_lag_ypc"],
        "WR": ["games_missed", "age", "team_targets_share"]}
FORMA = ["player_id", "start", "stop"]


def carrega(nome: str, caminho: Path):
    spec = importlib.util.spec_from_file_location(nome, caminho)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome] = mod
    spec.loader.exec_module(mod)
    return mod


def lista_final(pos: str) -> list[str]:
    dados = json.loads((COX / f"final_covs_{pos}.json").read_text(encoding="utf-8"))
    return dados if isinstance(dados, list) else dados["covariates"]


def padroniza(painel: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    z = painel[FORMA].copy()
    for c in colunas:
        v = pd.to_numeric(painel[c], errors="coerce")
        v = v.fillna(v.median())
        z[c] = (v - v.mean()) / v.std()
    return z


def ajusta(s51, dados: pd.DataFrame, covs: list[str]) -> CoxTimeVaryingFitter:
    return CoxTimeVaryingFitter(penalizer=s51.PENALIZER).fit(
        dados[["player_id", "start", "stop", "event"] + covs], id_col="player_id",
        event_col="event", start_col="start", stop_col="stop", show_progress=False)


def cox(s51, backward, pos: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    painel = pd.read_pickle(FEATURES / f"df_cox_{pos}_features.pkl")
    base = lista_final(pos)
    selecao = s51.rodar_selecao(painel.drop(columns=["contract_year_flag", "rookie_flag"], errors="ignore"), backward)
    nova = list(selecao["selecionadas"])

    cols_painel = [c for c in pd.read_pickle(ROOT / "Inferência" / "Preparação Modelagem" / "resultados" / f"df_cox_{pos}.pkl").columns
                   if c not in s51.ID_COLS + s51.TIME_COLS and c != "draft_round"]
    imputado = s51.apply_imputation_stats(painel, cols_painel, s51.fit_imputation_stats(painel, cols_painel))
    dados = imputado[["player_id", "start", "stop", "event"]].copy()
    for c in sorted(set(base) | set(VARIACOES)):
        v = imputado[c].astype(float)
        dados[c] = (v - v.mean()) / v.std()
    sem, com = ajusta(s51, dados, base), ajusta(s51, dados, base + VARIACOES)
    lr = 2 * (com.log_likelihood_ - sem.log_likelihood_)
    linhas = []
    for c in base + VARIACOES:
        linhas.append({"posicao": pos, "covariavel": c, "hr_sem": float(np.exp(sem.params_[c])) if c in base else np.nan,
                       "p_sem": float(sem.summary.loc[c, "p"]) if c in base else np.nan,
                       "hr_com": float(np.exp(com.params_[c])), "p_com": float(com.summary.loc[c, "p"])})
    resumo = {"lista_base": base, "lista_nova": nova, "entram": sorted(set(nova) - set(base)), "saem": sorted(set(base) - set(nova)),
              "lr": float(lr), "gl": len(VARIACOES), "p_lr": float(stats.chi2.sf(lr, len(VARIACOES)))}
    return pd.DataFrame(linhas), resumo, padroniza(painel, sorted(set(VARIACOES) | {c for c in nova if c in painel}))


def aalen(pos: str, z: pd.DataFrame, tmp: Path) -> tuple[pd.DataFrame, dict]:
    entrada = pd.read_csv(AALEN / "resultados" / f"aalen_input_{pos}.csv")
    novas = [c for c in z.columns if c not in FORMA and c not in entrada.columns]
    fundida = entrada.merge(z[FORMA + novas], on=FORMA, how="left", validate="1:1")
    if fundida[VARIACOES].isna().any().any():
        raise RuntimeError(f"{pos}: variacoes sem correspondencia na entrada do Aalen")
    arquivo = tmp / f"aalen_{pos}.csv"
    fundida.to_csv(arquivo, index=False)
    triagem = tmp / f"triagem_{pos}.csv"
    subprocess.run(["Rscript", str(AALEN / "shared" / "run_aalen_univariate_screening.R"), str(arquivo), str(triagem), ",".join(VARIACOES), pos],
                   check=True, capture_output=True)
    tri = pd.read_csv(triagem).set_index("covariate").p_value_null
    entram = sorted(tri[tri < 0.05].index)
    fixadas = json.loads((AALEN / "shared" / "listas_aalen_fixadas.json").read_text(encoding="utf-8"))[pos]
    pasta = tmp / f"aalen_{pos}"
    subprocess.run(["Rscript", str(AALEN / "shared" / "run_aalen_tied_inference.R"), str(arquivo), str(pasta), pos, ",".join(fixadas + entram)],
                   check=True, capture_output=True)
    resumo = json.loads(next(pasta.rglob(f"aalen_summary_{pos}.json")).read_text(encoding="utf-8"))
    oficial = json.loads((AALEN / "resultados" / f"aalen_summary_{pos}.json").read_text(encoding="utf-8"))
    linhas = [{"posicao": pos, "covariavel": c, "p_sem": float(oficial["tests_null_effect"][c]["p_value"]) if c in oficial["tests_null_effect"] else np.nan,
               "p_com": float(resumo["tests_null_effect"][c]["p_value"]), "p_triagem": float(tri[c]) if c in tri.index else np.nan}
              for c in NOVE[pos] + entram]
    return pd.DataFrame(linhas), {"triagem": {c: float(v) for c, v in tri.items()}, "entram": entram,
                                  "janela": [resumo["janela_inicio"], resumo["janela_fim"]],
                                  "janela_oficial": [oficial["janela_inicio"], oficial["janela_fim"]]}


def cox_aalen(s36, pos: str, nova: list[str], z: pd.DataFrame, tmp: Path) -> pd.DataFrame:
    espec = s36.spec_da_posicao(pos)
    aditivo = [c for c in espec["additive"] if c in nova]
    multiplicativo = [c for c in nova if c not in aditivo]
    entrada = pd.read_csv(COX / f"schoenfeld_input_{pos}.csv")
    faltam = [c for c in nova if c not in entrada.columns]
    fundida = entrada.merge(z[FORMA + faltam], on=FORMA, how="left", validate="1:1")
    arquivo, pasta = tmp / f"ca_{pos}.csv", tmp / f"ca_{pos}"
    fundida.to_csv(arquivo, index=False)
    pasta.mkdir(exist_ok=True)
    ini, fim = s36.janela_aalen(pos)
    subprocess.run(["Rscript", str(s36.RSCRIPT), str(arquivo), str(pasta), pos, s36._join_covs(aditivo), s36._join_covs(multiplicativo), ini, fim],
                   check=True, capture_output=True)
    beta = pd.read_csv(pasta / f"cox_aalen_beta_{pos}.csv")
    coluna_p = next(c for c in beta.columns if c.lower().startswith("p"))
    linhas = [{"posicao": pos, "covariavel": r.covariate, "bloco": "multiplicativo", "hr": float(r.HR), "p": float(r[coluna_p])} for _, r in beta.iterrows()]
    resumo = json.loads((pasta / f"cox_aalen_summary_{pos}.json").read_text(encoding="utf-8"))
    for c, v in (resumo.get("tests_null_additive") or resumo.get("tests_null_effect") or {}).items():
        if "Intercept" not in c:
            linhas.append({"posicao": pos, "covariavel": c, "bloco": "aditivo", "hr": np.nan, "p": float(v["p_value"])})
    return pd.DataFrame(linhas)


def predicao(of, tp) -> pd.DataFrame:
    principal = {p: par[0] for p, par in tp.modelos_destaque(pd.read_csv(OF / "resultados" / "resumo_walkforward_2018_2021.csv")).items()}
    linhas = []
    for pos in POS:
        painel, covs = of.load_panel(pos)
        ampliado = covs + [c for c in VARIACOES + ["rookie_flag", "draft_round"] if c not in covs]
        bruto = pd.read_pickle(FEATURES / f"df_cox_{pos}_features.pkl")
        painel_ampliado = tp.build_residual_snapshot_panel(bruto, ampliado, tp.META_COLS)
        for nome, base, x in (("painel", painel, covs), ("painel_mais_variacoes", painel_ampliado, ampliado)):
            for origem in (2018, 2019, 2020, 2021):
                tr_raw, te_raw = of.walkforward_split(base, origem)
                tr, te, use, _, _ = of.preprocess(tr_raw, te_raw, x)
                te = of.mask_test_support(tr, te)
                t = of.candidate_times(tr["residual_duration_model"].to_numpy(), te["residual_duration_model"].to_numpy())
                risco, surv, _ = of.fit_predict(principal[pos], of.hp_do_fold(pos, principal[pos], origem), tr, te, use,
                                                "residual_duration_model", "residual_event", t)
                m = of.score(tr, te, "residual_duration_model", "residual_event", risco, surv, t, uno=False)
                linhas.append({"posicao": pos, "modelo": principal[pos], "conjunto": nome, "origem": origem, "n_covs": len(use),
                               "c": m["c_index"], "ibs": m["ibs"]})
    return pd.DataFrame(linhas)


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    s51 = carrega("_passo51", ROOT / "pipeline" / "51_sensibilidade_inferencial.py")
    s36 = carrega("_passo36", ROOT / "pipeline" / "36_cox_aalen_inferencial.py")
    backward = s51.carregar_backward_cox_tdv()
    sys.path.insert(0, str(OF))
    sys.path.insert(0, str(FEATURES))
    of = carrega("run_oficial", OF / "run_oficial.py")
    import tgb_protocol as tp

    t_cox, t_aalen, t_ca, resumo = [], [], [], {}
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for pos in POS:
            hrs, r_cox, z = cox(s51, backward, pos)
            p_aalen, r_aalen = aalen(pos, z, tmp)
            t_cox.append(hrs)
            t_aalen.append(p_aalen)
            t_ca.append(cox_aalen(s36, pos, r_cox["lista_nova"], z, tmp))
            resumo[pos] = {"cox": r_cox, "aalen": r_aalen}
    pd.concat(t_cox).to_csv(SAIDA / "variacoes_recentes_cox.csv", index=False)
    pd.concat(t_aalen).to_csv(SAIDA / "variacoes_recentes_aalen.csv", index=False)
    pd.concat(t_ca).to_csv(SAIDA / "variacoes_recentes_cox_aalen.csv", index=False)
    predicao(of, tp).to_csv(SAIDA / "variacoes_recentes_predicao.csv", index=False)
    (SAIDA / "variacoes_recentes_resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
