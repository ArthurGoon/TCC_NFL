#!/usr/bin/env python3
"""Sensibilidades da selecao do Cox: linhas finais censuradas e frequencia bootstrap.

1. Linhas finais censuradas. Quem e censurado jogou ate t, mas nao se sabe se saiu
   depois de t. O painel o mantem no conjunto de risco em t. A alternativa censura
   no inicio do intervalo (Prentice e Gloeckler, 1978, p. 59): retira a ultima linha
   de cada censurado. Compara HR e p do Cox final, refaz a selecao e o Kaplan-Meier.
2. Frequencia de selecao. Repete a cadeia de selecao em B reamostras de jogadores
   (Harrell, 2001, p. 59).
3. Testes globais. Candidatas contra o modelo nulo (Harrell, 2001, p. 58).

A cadeia de selecao e a do passo 51 (que reaproveita a dos passos 11/12/13).

Saidas (``Inferência/sensibilidade/resultados/``):
- ``sens_censura_inicio_{POS}.csv``: HR e p do Cox final nas duas regras;
- ``frequencia_selecao_{POS}.csv``: frequencia de cada covariavel nas reamostras;
- ``schoenfeld_direcao_{POS}.csv``: sinal de beta(t) ao longo do tempo;
- ``sens_selecao_resumo.json``: testes globais, medianas e Jaccard.
"""
from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter
from scipy import stats
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
PASSO_51 = ROOT / "pipeline" / "51_sensibilidade_inferencial.py"
COX_DIR = ROOT / "Inferência" / "Modelagem de COX" / "resultados"
SAIDA = ROOT / "Inferência" / "sensibilidade" / "resultados"

B_REAMOSTRAS = 50
SEMENTE = 20241012
LIMIAR_PADRAO = 50

spec = importlib.util.spec_from_file_location("_passo_sens_inferencial", PASSO_51)
s51 = importlib.util.module_from_spec(spec)
sys.modules["_passo_sens_inferencial"] = s51
spec.loader.exec_module(s51)


def candidatas_da_cadeia(df_cox: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Mesma ordem de ``rodar_selecao``: devolve o painel padronizado e as candidatas."""
    cov_cols = [c for c in df_cox.columns if c not in s51.ID_COLS + s51.TIME_COLS and c != "draft_round"]
    estat = s51.fit_imputation_stats(df_cox, cov_cols)
    imputado = s51.apply_imputation_stats(df_cox, cov_cols, estat)
    modelo = imputado.copy()
    modelo[cov_cols] = StandardScaler().fit_transform(modelo[cov_cols])
    df_fit = modelo[["player_id", "start", "stop", "event"] + cov_cols].copy()
    identificaveis = s51.covariaveis_identificaveis_cox(df_fit, cov_cols, verbose=False)
    filtradas = s51.filter_collinear_covariates(
        df_fit, identificaveis, threshold=s51.LIMIAR_COLINEARIDADE, verbose=False
    )
    candidatas = s51.force_missing_flags(filtradas, df_fit.columns.tolist())
    return df_fit, s51.covariaveis_identificaveis_cox(df_fit, candidatas, verbose=False)


def teste_global(df_cox: pd.DataFrame) -> dict:
    """Razao de verossimilhanca das candidatas contra o modelo nulo."""
    df_fit, candidatas = candidatas_da_cadeia(df_cox)
    fit = CoxTimeVaryingFitter(penalizer=0.0)
    fit.fit(df_fit[["player_id", "start", "stop", "event"] + candidatas], id_col="player_id",
            start_col="start", stop_col="stop", event_col="event", show_progress=False)
    resultado = fit.log_likelihood_ratio_test()
    return {
        "n_candidatas": len(candidatas),
        "qui2": float(resultado.test_statistic),
        "gl": int(len(candidatas)),
        "p": float(resultado.p_value),
    }


def sem_linha_final_censurada(painel: pd.DataFrame) -> pd.DataFrame:
    ordenado = painel.sort_values(["player_id", "stop"])
    ultima = ~ordenado.duplicated("player_id", keep="last")
    return ordenado[~(ultima & (ordenado["event"] == 0))].copy()


def mediana_km(painel: pd.DataFrame) -> float:
    """Mesmo Kaplan-Meier por jogador dos passos 08/09/10, com entrada tardia."""
    jogador = painel.groupby("player_id").agg(
        duration=("stop", "max"), entry=("start", "min"), event=("event", "max")
    )
    kmf = KaplanMeierFitter()
    kmf.fit(jogador["duration"], event_observed=jogador["event"], entry=jogador["entry"])
    return float(kmf.median_survival_time_)


def comparar_cox_final(painel: pd.DataFrame, covariaveis: list[str]) -> pd.DataFrame:
    """Cox final com a padronizacao do painel completo, com e sem as linhas finais."""
    cov_cols = [c for c in painel.columns if c not in s51.ID_COLS + s51.TIME_COLS and c != "draft_round"]
    estat = s51.fit_imputation_stats(painel, cov_cols)
    imputado = s51.apply_imputation_stats(painel, cov_cols, estat)
    imputado[covariaveis] = StandardScaler().fit_transform(imputado[covariaveis])
    reduzido = sem_linha_final_censurada(imputado)
    base = s51.ajustar_cox(imputado, covariaveis).summary
    alt = s51.ajustar_cox(reduzido, covariaveis).summary
    return pd.DataFrame({
        "covariavel": covariaveis,
        "hr": base.loc[covariaveis, "exp(coef)"].to_numpy(),
        "p": base.loc[covariaveis, "p"].to_numpy(),
        "hr_censura_inicio": alt.loc[covariaveis, "exp(coef)"].to_numpy(),
        "p_censura_inicio": alt.loc[covariaveis, "p"].to_numpy(),
    })


def reamostra(painel: pd.DataFrame, semente: int) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    jogadores = painel["player_id"].unique()
    sorteados = rng.choice(jogadores, size=len(jogadores), replace=True)
    grupos = dict(tuple(painel.groupby("player_id")))
    partes = []
    for i, jogador in enumerate(sorteados):
        copia = grupos[jogador].copy()
        copia["player_id"] = f"{jogador}__{i}"
        partes.append(copia)
    return pd.concat(partes, ignore_index=True)


# Herdados pelos processos filhos via fork: a funcao backward vem de exec e nao e serializavel.
_PAINEL = None
_BACKWARD = None


def selecao_em_reamostra(semente: int) -> dict:
    try:
        return {"semente": semente, "selecionadas": s51.rodar_selecao(reamostra(_PAINEL, semente), _BACKWARD)["selecionadas"]}
    except Exception as erro:  # reamostra pode ficar sem variacao em alguma coluna
        return {"semente": semente, "erro": f"{type(erro).__name__}: {erro}"}


R_DIRECAO = """
suppressMessages(library(survival))
a <- commandArgs(TRUE)
x <- read.csv(a[1], check.names = FALSE)
covs <- setdiff(names(x), c("player_id", "start", "stop", "event"))
f <- as.formula(paste("Surv(start, stop, event) ~", paste(sprintf("`%s`", covs), collapse = "+")))
fit <- coxph(f, data = x, ties = "efron")
z <- cox.zph(fit, transform = "rank")
incl <- sapply(seq_along(covs), function(j) coef(lm(z$y[, j] ~ z$x))[2])
write.csv(data.frame(covariavel = covs, beta = coef(fit), inclinacao_beta_t = incl,
                     p_schoenfeld = z$table[seq_along(covs), "p"]), a[2], row.names = FALSE)
"""


def direcao_schoenfeld(posicao: str) -> pd.DataFrame:
    """Sinal da inclinacao de beta(t): com beta e inclinacao de sinais opostos, o efeito enfraquece."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "direcao.R"
        script.write_text(R_DIRECAO, encoding="utf-8")
        saida = SAIDA / f"schoenfeld_direcao_{posicao}.csv"
        subprocess.run([s51.rscript(), str(script), str(COX_DIR / f"schoenfeld_input_{posicao}.csv"), str(saida)], check=True)
    tabela = pd.read_csv(saida)
    tabela["efeito_enfraquece"] = np.sign(tabela["beta"]) != np.sign(tabela["inclinacao_beta_t"])
    tabela.to_csv(saida, index=False)
    return tabela


def taxa_saida_por_ano(painel: pd.DataFrame) -> dict:
    """Saidas sobre linhas em risco em cada ano de carreira."""
    grupo = painel.groupby("stop")["event"].agg(["sum", "count"])
    grupo = grupo[grupo["count"] >= 30]
    return {str(int(k)): round(float(v["sum"] / v["count"]), 3) for k, v in grupo.iterrows()}


def analisar(posicao: str, backward) -> dict:
    base = s51.carregar_base(posicao)
    painel = s51.construir_painel(base, posicao, LIMIAR_PADRAO, "A")
    selecao = s51.rodar_selecao(painel, backward)
    oficial = json.loads((COX_DIR / f"final_covs_{posicao}.json").read_text(encoding="utf-8"))
    oficial = oficial["final_covs"] if isinstance(oficial, dict) and "final_covs" in oficial else oficial
    if s51.substantivas(selecao["selecionadas"]) != s51.substantivas(oficial):
        raise RuntimeError(f"{posicao}: a selecao nao reproduz final_covs")

    comparacao = comparar_cox_final(painel, list(selecao["selecionadas"]))
    comparacao.to_csv(SAIDA / f"sens_censura_inicio_{posicao}.csv", index=False)

    reduzido = sem_linha_final_censurada(painel)
    selecao_alt = s51.rodar_selecao(reduzido, backward)

    global _PAINEL, _BACKWARD
    _PAINEL, _BACKWARD = painel, backward
    with mp.get_context("fork").Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        resultados = pool.map(selecao_em_reamostra, [SEMENTE + b for b in range(B_REAMOSTRAS)])
    validas = [r["selecionadas"] for r in resultados if "selecionadas" in r]
    contagem: dict[str, int] = {}
    for lista in validas:
        for cov in lista:
            contagem[cov] = contagem.get(cov, 0) + 1
    frequencia = pd.DataFrame(
        [{"covariavel": c, "frequencia": n / len(validas), "no_cox_final": c in selecao["selecionadas"]}
         for c, n in contagem.items()]
    ).sort_values("frequencia", ascending=False)
    frequencia.to_csv(SAIDA / f"frequencia_selecao_{posicao}.csv", index=False)

    return {
        "linhas_retiradas": int(len(painel) - len(reduzido)),
        "taxa_saida_por_ano": taxa_saida_por_ano(painel),
        "schoenfeld_enfraquece": direcao_schoenfeld(posicao).query("p_schoenfeld < 0.05")[["covariavel", "efeito_enfraquece"]].to_dict("records"),
        "mediana_km": mediana_km(painel),
        "mediana_km_censura_inicio": mediana_km(reduzido),
        "selecao_censura_inicio": selecao_alt["selecionadas"],
        "jaccard_censura_inicio": s51.jaccard(selecao["selecionadas"], selecao_alt["selecionadas"]),
        "teste_global_candidatas": teste_global(painel),
        "reamostras_validas": len(validas),
        "reamostras_com_erro": len(resultados) - len(validas),
        "tamanho_mediano_lista": float(np.median([len(v) for v in validas])),
    }


def main() -> int:
    SAIDA.mkdir(parents=True, exist_ok=True)
    backward = s51.carregar_backward_cox_tdv()
    resumo = {"b_reamostras": B_REAMOSTRAS, "semente": SEMENTE}
    for posicao in s51.POSICOES:
        resumo[posicao] = analisar(posicao, backward)
        print(posicao, json.dumps(resumo[posicao], ensure_ascii=False), flush=True)
    (SAIDA / "sens_selecao_resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
