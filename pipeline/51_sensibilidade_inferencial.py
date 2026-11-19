#!/usr/bin/env python3
"""Sensibilidade inferencial do Cox: limiar da taxa, nao draftados e imputacao multipla.

O capitulo 3 (``Escrita Final/texto/c3_material_tg.tex``) afirma
tres analises de sensibilidade; este passo as produz:

1. linha ~461: "Analises de sensibilidade com cortes entre 25 e 100 mantiveram
   estavel a selecao de covariaveis do Cox entre limiares adjacentes" (o corte
   e o denominador das taxas de carreira, fixo em ``den_lag > 50`` nos
   passos 04/05/06);
2. linha ~158: "Analises de sensibilidade com codificacoes alternativas para
   nao draftados nao alteraram a selecao final ... em QB, RB ou WR";
3. imputacao multipla como analise exploratoria condicional; ela nao comprova
   compatibilidade formal com White–Royston nem robustez inferencial.

Este passo reconstroi o painel Cox a partir do CSV bruto, repete exatamente a
cadeia de selecao dos passos 11/12/13 e grava a evidencia em
``Inferência/sensibilidade/resultados/``. Nenhum arquivo do pipeline e alterado:
a logica de painel dos passos 04/05/06 e a funcao ``backward_cox_tdv`` dos
passos 11/12/13 sao reaproveitadas sem modificacao (a funcao e executada a
partir do proprio codigo-fonte do passo 11, ver ``carregar_backward_cox_tdv``).

Teste de aceitacao (obrigatorio): a reconstrucao com limiar 50 tem de reproduzir
a selecao do proprio pipeline, ou seja, o conjunto de covariaveis substantivas
selecionadas tem de ser igual ao conteudo de ``final_covs_{POS}.json`` ignorando
as entradas terminadas em ``_missing``. Se nao reproduzir, o passo falha.

Saidas (``Inferência/sensibilidade/resultados/``):

- ``limiares_taxa.csv``: um registro por limiar e posicao, com tamanho do
  painel, eventos, covariaveis selecionadas e Jaccard contra o limiar 50;
- ``undrafted_cenarios.csv``: um registro por cenario de codificacao de nao
  draftados (A a D) e posicao, com Jaccard contra o cenario A;
- ``imputacao_multipla.csv``: um registro por covariavel do modelo final de
  cada posicao, com HR/p do ajuste unico like-for-like (sem a flag _missing),
  HR/p do ajuste unico original (cox_hr, com flag quando houver) e HR/p agrupados por Rubin;
- ``sens_ep_robusto_{POS}.csv``: HR, EP model-based, EP robusto por jogador
  (cluster(player_id), survival::coxph) e p de cada covariavel do Cox final;
- ``sens_empates_{POS}.csv``: coeficientes e p do Cox final com Efron e Breslow
  (survival::coxph); a % de eventos empatados vai em ``resumo.json``;
- ``sens_tempo_discreto_{POS}.csv``: HR e p do Cox final com Efron, cloglog de tempo
  discreto (GLM binomial, efeito fixo por ano; principal, colunas HR_cloglog/p_cloglog;
  se nao convergir, o agrupado vira principal naquela posicao), cloglog agrupado
  (tempos com menos de 3 eventos somados ao vizinho; hr_cloglog_agrupado/p_cloglog_agrupado)
  e Breslow; os grupos e contagens vao em ``resumo.json``;
- ``sens_passer_rating_zero_QB.csv``: HR do passer_rating no Cox final do QB com
  as 26 linhas sem tentativas medianizadas, zeradas ou excluidas;
- ``sens_covariaveis_defasadas_{POS}.csv``: tipo (temporada corrente ou fixa/defasada)
  de cada covariavel final; HR/IC/p do Cox final com as correntes em t e com as
  correntes trocadas por t-1 (amostra comum: linhas com t-1 do mesmo jogador); HR/IC/p
  oficial (cox_hr) e n de linhas e eventos das duas amostras;
- ``resumo.json``: destaques, bloco ``metodo`` com o que foi calculado, tempos
  e limitacoes.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler


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

INFER_DIR = ROOT / "Inferência"
COX_DIR = INFER_DIR / "Modelagem de COX" / "resultados"
SAIDA = INFER_DIR / "sensibilidade" / "resultados"
PASSO_COX = ROOT / "pipeline" / "11_cox_tdc_qb.py"

sys.path.append(str((INFER_DIR / "shared").resolve()))
from survival_utils import (  # noqa: E402
    add_gap_aware_survival_structure,
    covariaveis_identificaveis_cox,
    apply_imputation_stats,
    filter_collinear_covariates,
    fit_imputation_stats,
    force_missing_flags,
    load_clean_position_data,
)

POSICOES = ("QB", "RB", "WR")

# Limiares do denominador das taxas de carreira (o pipeline usa 50). O texto da
# tese fala em "cortes entre 25 e 100".
LIMIARES_TAXA = (25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100)
LIMIAR_PADRAO = 50

LIMIAR_COLINEARIDADE = 0.85
PENALIZER = 0.0
ALFA = 0.05
M_IMPUTACOES = 10
SEMENTE_BASE = 42

ID_COLS = ["player_id", "player_name", "season"]
TIME_COLS = ["start", "stop", "event", "entry"]

# Palavras-chave de taxa usadas pelo proprio prep (passos 04/05/06) para criar
# as flags _missing.
CHAVES_TAXA = (
    "comp_pct", "ypa", "int_pct", "td_pct", "ypc", "yptarget", "ypr",
    "catch_rate", "adot_acc", "rush_td_pct", "yptouch", "reception_rate",
)

# Configuracao por posicao, copiada literalmente dos passos 04/05/06
# (CAREER_ADITIVAS, COMPONENTES_TAXA, CONTEXTO_POSICAO).
CONFIG_POSICOES: dict[str, dict[str, list]] = {
    "QB": {
        "aditivas": [
            "career_qb_dropback", "career_pass_attempts", "career_complete_pass",
            "career_passing_yards", "career_pass_touchdown", "career_interception",
            "career_qb_scramble", "career_rush_attempts", "career_offense_snaps",
            "career_fantasy_points_ppr", "career_rushing_yards",
            "career_pass_attempts_redzone", "career_pass_touchdown_redzone",
            "career_fumble", "career_fumble_lost",
        ],
        "taxas": [
            ("career_complete_pass", "career_pass_attempts", "career_lag_comp_pct"),
            ("career_passing_yards", "career_pass_attempts", "career_lag_ypa"),
            ("career_interception", "career_pass_attempts", "career_lag_int_pct"),
            ("career_pass_touchdown", "career_pass_attempts", "career_lag_td_pct"),
            ("career_rushing_yards", "career_rush_attempts", "career_lag_ypc_qb"),
        ],
        "contexto": [
            "depth_team", "offense_pct", "games_missed", "age", "years_exp",
            "unobs_career_years", "pre_window_veteran", "first_observed_season",
            "unobserved_history_flag", "within_window_gap_years_before_t", "had_prior_gap_flag",
            "passer_rating", "team_pass_attempts_share",
        ],
    },
    "RB": {
        "aditivas": [
            "career_rush_attempts", "career_rushing_yards", "career_rush_touchdown",
            "career_touches", "career_total_yards", "career_offense_snaps",
            "career_fantasy_points_ppr", "career_receptions", "career_receiving_yards",
            "career_targets", "career_rush_attempts_redzone", "career_rush_touchdown_redzone",
            "career_fumble", "career_fumble_lost", "career_yards_after_catch",
        ],
        "taxas": [
            ("career_rushing_yards", "career_rush_attempts", "career_lag_ypc"),
            ("career_rush_touchdown", "career_rush_attempts", "career_lag_rush_td_pct"),
        ],
        "contexto": [
            "depth_team", "offense_pct", "games_missed", "age", "years_exp",
            "unobs_career_years", "pre_window_veteran", "first_observed_season",
            "unobserved_history_flag", "within_window_gap_years_before_t", "had_prior_gap_flag",
            "team_rush_attempts_share", "team_rushing_yards_share",
        ],
    },
    "WR": {
        "aditivas": [
            "career_targets", "career_receptions", "career_receiving_yards",
            "career_receiving_air_yards", "career_offense_snaps",
            "career_fantasy_points_ppr", "career_rush_attempts", "career_rushing_yards",
            "career_receiving_touchdown", "career_targets_redzone",
            "career_receptions_redzone", "career_yards_after_catch",
            "career_fumble", "career_fumble_lost",
        ],
        "taxas": [
            ("career_receiving_yards", "career_targets", "career_lag_yptarget"),
            ("career_receiving_yards", "career_receptions", "career_lag_ypr"),
            ("career_receptions", "career_targets", "career_lag_catch_rate"),
            ("career_receiving_air_yards", "career_targets", "career_lag_adot_acc"),
        ],
        "contexto": [
            "depth_team", "offense_pct", "games_missed", "age", "years_exp",
            "unobs_career_years", "pre_window_veteran", "first_observed_season",
            "unobserved_history_flag", "within_window_gap_years_before_t", "had_prior_gap_flag",
            "team_targets_share", "adot",
        ],
    },
}

DEMOGRAFICAS = ["height", "weight", "draft_ovr", "draft_round", "undrafted_flag"]

DESCRICAO_CENARIOS = {
    "A": (
        "codificacao do pipeline: draft_round.fillna(8), "
        "undrafted_flag = (draft_round == 8) e draft_ovr = 0 para nao draftados"
    ),
    "B": (
        "draft_ovr permanece ausente para nao draftados (sem zerar); "
        "draft_round.fillna(8) e undrafted_flag identicos a A, de modo que a "
        "imputacao por mediana dos draftados do proprio pipeline preenche draft_ovr"
    ),
    "C": (
        "draft_round.fillna(9) no lugar de 8, com undrafted_flag mantendo os "
        "valores de A (o indicador de nao draftado continua informativo) e "
        "draft_ovr = 0 para nao draftados como em A; muda apenas o codigo sentinela"
    ),
    "D": (
        "draft_ovr e draft_round fora do conjunto candidato; permanece apenas "
        "undrafted_flag, com a codificacao de A"
    ),
}


def substantivas(covariaveis) -> list[str]:
    """Selecao sem as flags mecanicas ``_missing`` (adicionadas por forcamento)."""
    return sorted(c for c in covariaveis if not str(c).endswith("_missing"))


def jaccard(a, b) -> float:
    conjunto_a, conjunto_b = set(a), set(b)
    if not conjunto_a and not conjunto_b:
        return 1.0
    return len(conjunto_a & conjunto_b) / len(conjunto_a | conjunto_b)


def carregar_base(posicao: str) -> pd.DataFrame:
    """Camada limpa do prep (agregacao + sanidade), carregada uma vez por posicao.

    ``load_clean_position_data`` e a unica etapa cara da reconstrucao. Ela e uma
    funcao pura do CSV bruto, entao carregar uma vez e reaplicar as etapas
    seguintes por limiar/cenario reproduz exatamente o que os passos 04/05/06
    fazem a cada execucao.
    """
    return load_clean_position_data(ROOT, posicao)


def construir_painel(base: pd.DataFrame, posicao: str, limiar: float, cenario: str = "A") -> pd.DataFrame:
    """Reconstroi ``df_cox_{POS}`` como nos passos 04/05/06.

    Reproduz, na mesma ordem: conversao numerica, codificacao de draft (que
    varia por cenario), filtro de inclusao minima, estrutura de sobrevivencia
    gap-aware, lags de carreira, taxas de carreira com corte no denominador,
    flags ``_missing`` e remocao de linhas sem tempo/evento.
    """
    if cenario not in DESCRICAO_CENARIOS:
        raise ValueError(f"cenario invalido: {cenario}")
    cfg = CONFIG_POSICOES[posicao]
    df = base.copy()

    for col in ["years_exp", "draft_round", "age", "height", "weight", "draft_ovr",
                "depth_team", "offense_pct", "games_missed", "season", "games_played_season"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    nao_draftado = df["draft_round"].isna()
    if cenario == "C":
        df["draft_round"] = df["draft_round"].fillna(9)
        df["undrafted_flag"] = nao_draftado.astype(int)
        df.loc[df["undrafted_flag"] == 1, "draft_ovr"] = 0
    else:
        df["draft_round"] = df["draft_round"].fillna(8)
        df["undrafted_flag"] = (df["draft_round"] == 8).astype(int)
        if cenario != "B":
            df.loc[df["undrafted_flag"] == 1, "draft_ovr"] = 0
    df = df.sort_values(["player_id", "season"]).reset_index(drop=True)

    if "games_played_season" in df.columns:
        df = df[(df["games_played_season"] >= 1) | (df["offense_snaps"] > 0)].copy()
    elif "offense_snaps" in df.columns:
        df = df[df["offense_snaps"] > 0].copy()

    df = add_gap_aware_survival_structure(df)
    assert (df["stop"] > df["start"]).all(), "ERRO: existem registros com stop <= start"

    lag_cols: list[str] = []
    for cv in cfg["aditivas"]:
        sv = cv.replace("career_", "season_")
        lag = cv.replace("career_", "observed_career_lag_")
        if cv in df.columns and sv in df.columns:
            df[lag] = (df[cv] - df[sv]).clip(lower=0)
            lag_cols.append(lag)

    for num_col, den_col, lag_name in cfg["taxas"]:
        lag_name = lag_name.replace("career_lag_", "observed_career_lag_")
        num_lag = num_col.replace("career_", "observed_career_lag_")
        den_lag = den_col.replace("career_", "observed_career_lag_")
        if num_lag in df.columns and den_lag in df.columns:
            df[lag_name] = np.where(df[den_lag] > limiar, df[num_lag] / df[den_lag], np.nan)
            lag_cols.append(lag_name)

    contexto = [c for c in cfg["contexto"] if c in df.columns]
    demograficas = [c for c in DEMOGRAFICAS if c in df.columns]
    cov_cols = list(dict.fromkeys(lag_cols + contexto + demograficas))
    cov_cols = [c for c in cov_cols if c in df.columns]
    if cenario == "D":
        cov_cols = [c for c in cov_cols if c not in ("draft_ovr", "draft_round")]

    df_cox = df[ID_COLS + TIME_COLS + cov_cols].copy()

    taxa_cols = [c for c in cov_cols if any(x in c for x in CHAVES_TAXA)]
    for col in taxa_cols:
        if col in df_cox.columns:
            df_cox[f"{col}_missing"] = df_cox[col].isna().astype(int)

    return df_cox.dropna(subset=TIME_COLS)


def carregar_backward_cox_tdv():
    """Devolve a ``backward_cox_tdv`` dos passos 11/12/13, sem reimplementar.

    A funcao nao esta em ``Inferência/shared/survival_utils.py``: ela e definida
    dentro do codigo embutido dos passos 11/12/13 (identica nos tres passos).
    Para nao duplicar a implementacao, o modulo ``pipeline/11_cox_tdc_qb.py`` e
    importado por caminho (sem executar ``main``) e apenas o no da funcao e
    compilado e executado aqui.
    """
    if not PASSO_COX.exists():
        raise FileNotFoundError(f"Passo ausente: {PASSO_COX.relative_to(ROOT)}")
    spec = importlib.util.spec_from_file_location("_passo_cox_tdc_referencia", PASSO_COX)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar {PASSO_COX.name}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)

    arvore = ast.parse(modulo.CODIGO_PASSO)
    no = next(
        (item for item in arvore.body
         if isinstance(item, ast.FunctionDef) and item.name == "backward_cox_tdv"),
        None,
    )
    if no is None:
        raise RuntimeError("backward_cox_tdv nao encontrado no codigo do passo 11")
    trecho = ast.Module(body=[no], type_ignores=[])
    ast.fix_missing_locations(trecho)
    escopo = {"CoxTimeVaryingFitter": CoxTimeVaryingFitter}
    exec(compile(trecho, str(PASSO_COX), "exec"), escopo)
    return escopo["backward_cox_tdv"]


def rodar_selecao(df_cox: pd.DataFrame, backward) -> dict:
    """Cadeia de selecao dos passos 11/12/13 sobre um painel Cox ja construido."""
    cov_cols = [c for c in df_cox.columns if c not in ID_COLS + TIME_COLS]
    cov_cols = [c for c in cov_cols if c != "draft_round"]

    imputer_stats = fit_imputation_stats(df_cox, cov_cols)
    df_imputado = apply_imputation_stats(df_cox, cov_cols, imputer_stats)
    scaler = StandardScaler()
    df_model = df_imputado.copy()
    df_model[cov_cols] = scaler.fit_transform(df_model[cov_cols])
    df_fit = df_model[["player_id", "start", "stop", "event"] + cov_cols].copy()

    # Mesma ordem dos passos 11-13: identificaveis saem ANTES do filtro |r|>0,85.
    cov_identificaveis = covariaveis_identificaveis_cox(df_fit, cov_cols, verbose=False)
    cov_filtradas = filter_collinear_covariates(
        df_fit, cov_identificaveis, threshold=LIMIAR_COLINEARIDADE, verbose=False
    )
    candidatas = force_missing_flags(cov_filtradas, df_fit.columns.tolist())
    candidatas = covariaveis_identificaveis_cox(df_fit, candidatas, verbose=False)

    n_eventos = int(df_cox["event"].sum())
    max_vars = max(2, int(n_eventos / 10))
    limiar_p = ALFA if n_eventos >= 50 else 0.10

    modelo, selecionadas = backward(
        df_fit, id_col="player_id", start_col="start", stop_col="stop",
        event_col="event", covariates=candidatas, penalizer=PENALIZER,
        threshold=limiar_p, verbose=False,
    )
    if len(selecionadas) > max_vars:
        selecionadas = modelo.summary["p"].nsmallest(max_vars).index.tolist()
        cols_refit = ["player_id", "start", "stop", "event"] + selecionadas
        modelo = CoxTimeVaryingFitter(penalizer=PENALIZER)
        modelo.fit(df_fit[cols_refit], id_col="player_id", start_col="start",
                   stop_col="stop", event_col="event", show_progress=False)

    return {
        "df_imputado": df_imputado,
        "selecionadas": list(selecionadas),
        "substantivas": substantivas(selecionadas),
        "n_candidatas": len(candidatas),
        "n_eventos": n_eventos,
        "n_linhas": len(df_cox),
        "limiar_p": limiar_p,
        "max_vars": max_vars,
    }


def ajustar_cox(dados: pd.DataFrame, covariadas: list[str]):
    modelo = CoxTimeVaryingFitter(penalizer=PENALIZER)
    modelo.fit(dados[["player_id", "start", "stop", "event"] + covariadas],
               id_col="player_id", start_col="start", stop_col="stop",
               event_col="event", show_progress=False)
    return modelo


def p_valor_wald(coeficiente: float, erro_padrao: float, graus_liberdade: float) -> float:
    """p bilateral do teste de Wald: t com ``graus_liberdade`` finitos, normal quando gl = inf."""
    estatistica = coeficiente / erro_padrao if erro_padrao > 0 else float("nan")
    if np.isfinite(graus_liberdade):
        return float(2 * stats.t.sf(abs(estatistica), graus_liberdade))
    return float(2 * stats.norm.sf(abs(estatistica)))


def rubin(coeficientes: list[float], erros: list[float]) -> dict:
    """Regras de Rubin para coeficiente e erro-padrao agrupados.

    Graus de liberdade classicos de Rubin (1987): nu = (m - 1)(1 + Ubar / ((1 + 1/m) B))^2,
    com nu = inf quando B = 0 (nesse caso o p usa a normal). Nao e a correcao de
    Barnard & Rubin (1999), que depende do numero de observacoes.
    """
    m = len(coeficientes)
    q_barra = float(np.mean(coeficientes))
    u_barra = float(np.mean(np.square(erros)))
    if m > 1:
        b = float(np.var(coeficientes, ddof=1))
    else:
        b = 0.0
    variancia_total = u_barra + (1.0 + 1.0 / m) * b
    erro_padrao = float(np.sqrt(variancia_total))
    if b > 0:
        graus_liberdade = (m - 1) * (1.0 + u_barra / ((1.0 + 1.0 / m) * b)) ** 2
    else:
        graus_liberdade = float("inf")
    estatistica = q_barra / erro_padrao if erro_padrao > 0 else float("nan")
    return {
        "coef_pooled": q_barra,
        "se_pooled": erro_padrao,
        "variancia_entre": b,
        "variancia_intra": u_barra,
        "graus_liberdade": graus_liberdade,
        "p_pooled": p_valor_wald(q_barra, erro_padrao, graus_liberdade),
        "p_pooled_normal": float(2 * stats.norm.sf(abs(estatistica))),
    }


def auxiliares_imputacao(df_cox: pd.DataFrame) -> pd.DataFrame:
    """Colunas que entram SO no modelo de imputacao (nao no Cox).

    ``aux_evento`` e o indicador de evento do intervalo; ``aux_nelson_aalen_stop``
    e o risco acumulado marginal H(stop) de Nelson-Aalen com entrada tardia,
    avaliado no fim de cada intervalo. Os auxiliares sao inspirados em White &
    Royston (2009), mas este uso em painel temporal nao e uma implementacao ou
    validacao formal do metodo; a leitura da MI e exploratoria e condicional.
    H(t) = soma de d(s)/n(s) sobre tempos de evento s <= t, com d(s) = eventos com
    stop = s e n(s) = linhas com start < s <= stop (quem entra em s nao esta em risco).
    """
    eventos_por_tempo = df_cox.loc[df_cox["event"] == 1].groupby("stop").size()
    tempos_evento = eventos_por_tempo.index.to_numpy(dtype=float)
    incrementos = []
    for s, d in zip(tempos_evento, eventos_por_tempo.to_numpy()):
        em_risco = int(((df_cox["start"] < s) & (df_cox["stop"] >= s)).sum())
        incrementos.append(d / em_risco)
    h_acum = np.concatenate([[0.0], np.cumsum(incrementos)])
    posicao_em_stop = np.searchsorted(tempos_evento, df_cox["stop"].to_numpy(dtype=float), side="right")
    h_stop = h_acum[posicao_em_stop]
    if len(h_stop) != len(df_cox) or not np.isfinite(h_stop).all():
        raise RuntimeError("Nelson-Aalen em stop nao alinhado ao painel")
    return pd.DataFrame({
        "aux_evento": df_cox["event"].astype(float).to_numpy(),
        "aux_nelson_aalen_stop": h_stop,
    }, index=df_cox.index)


def imputacao_multipla(df_cox: pd.DataFrame, selecao: dict, posicao: str) -> dict:
    """Imputacao multipla (m = 10) do modelo final, com pooling de Rubin.

    As colunas imputadas sao as covariaveis finais com ao menos um valor ausente
    no painel (as unicas que a imputacao pode afetar). O imputador inclui o
    desfecho (``auxiliares_imputacao``) como preditor. O ajuste usa as mesmas
    linhas do pipeline e um ``StandardScaler`` restrito as covariaveis finais,
    que reproduz o escalonamento coluna a coluna do passo 11.
    """
    finalistas = list(selecao["substantivas"])
    df_imputado = selecao["df_imputado"]
    scaler_final = StandardScaler().fit(df_imputado[finalistas])
    tabela_hr = ler_cox_hr(posicao)

    colunas_ausentes = [c for c in finalistas if df_cox[c].isna().any()]
    bruto = df_cox[finalistas].astype(float)
    mascara_ausente = {c: bruto[c].isna().to_numpy() for c in colunas_ausentes}
    desenhos: dict[str, list[np.ndarray]] = {c: [] for c in colunas_ausentes}
    entrada_imputador = pd.concat([bruto, auxiliares_imputacao(df_cox)], axis=1)
    colunas_imputador = finalistas + ["aux_evento", "aux_nelson_aalen_stop"]

    coeficientes: dict[str, list[float]] = {c: [] for c in finalistas}
    erros: dict[str, list[float]] = {c: [] for c in finalistas}
    p_por_imputacao: dict[str, list[float]] = {c: [] for c in finalistas}
    for m in range(1, M_IMPUTACOES + 1):
        valores = bruto.copy()
        if colunas_ausentes:
            imputador = IterativeImputer(
                random_state=SEMENTE_BASE + m,
                sample_posterior=True,
                initial_strategy="median",
                max_iter=10,
            )
            preenchido = imputador.fit_transform(entrada_imputador[colunas_imputador])
            valores[finalistas] = preenchido[:, : len(finalistas)]
            for c in colunas_ausentes:
                desenhos[c].append(valores[c].to_numpy()[mascara_ausente[c]].copy())
        escala = pd.DataFrame(
            scaler_final.transform(valores[finalistas]),
            columns=finalistas, index=df_cox.index,
        )
        dados = pd.concat([df_cox[["player_id", "start", "stop", "event"]], escala], axis=1)
        modelo = ajustar_cox(dados, finalistas)
        resumo = modelo.summary
        for c in finalistas:
            coeficientes[c].append(float(resumo.loc[c, "coef"]))
            erros[c].append(float(resumo.loc[c, "se(coef)"]))
            p_por_imputacao[c].append(float(resumo.loc[c, "p"]))

    # Ajuste unico (imputacao por mediana/grupo do pipeline) para validar contra
    # cox_hr_{POS}.csv: e o mesmo modelo final gravado pelos passos 11/12/13.
    escala_unica = pd.DataFrame(
        scaler_final.transform(df_imputado[finalistas]),
        columns=finalistas, index=df_cox.index,
    )
    dados_unicos = pd.concat(
        [df_cox[["player_id", "start", "stop", "event"]], escala_unica], axis=1
    )
    modelo_unico = ajustar_cox(dados_unicos, finalistas)
    resumo_unico = modelo_unico.summary

    linhas = []
    for c in finalistas:
        agrupado = rubin(coeficientes[c], erros[c])
        coef_unico = float(resumo_unico.loc[c, "coef"])
        se_unica = float(resumo_unico.loc[c, "se(coef)"])
        # Referencia like-for-like: mesmas covariaveis do modelo imputado (sem a flag
        # _missing), ajustadas no painel com a imputacao por mediana do pipeline.
        p_unica_sem_flag = float(resumo_unico.loc[c, "p"])
        hr_unica_sem_flag = float(np.exp(coef_unico))
        # Original, para referencia: quando cox_hr_{POS}.csv existe, HR/p vem dele
        # (o modelo publicado, que pode conter a flag). Em QB difere do sem flag.
        hr_unica_original = hr_unica_sem_flag
        p_unica_original = p_unica_sem_flag
        fonte_hr = "reconstruido"
        if tabela_hr is not None and c in tabela_hr.index:
            hr_unica_original = float(tabela_hr.loc[c, "HR"])
            p_unica_original = float(tabela_hr.loc[c, "p_value"])
            fonte_hr = "cox_hr_csv"
        # Isola o efeito da inflacao do erro-padrao: mesmo coeficiente do ajuste
        # unico com o erro-padrao agrupado. E o que a tese afirma na linha ~470.
        p_inflacao_apenas = p_valor_wald(coef_unico, agrupado["se_pooled"], agrupado["graus_liberdade"])
        linhas.append({
            "posicao": posicao,
            "covariavel": c,
            "hr_unica_original": hr_unica_original,
            "p_unica_original": p_unica_original,
            "fonte_hr_unica_original": fonte_hr,
            "hr_unica_sem_flag": hr_unica_sem_flag,
            "p_unica_sem_flag": p_unica_sem_flag,
            "hr_pooled": float(np.exp(agrupado["coef_pooled"])),
            "p_pooled": agrupado["p_pooled"],
            "significancia_inalterada": bool((p_unica_sem_flag < ALFA) == (agrupado["p_pooled"] < ALFA)),
            "coef_unico": coef_unico,
            "coef_pooled": agrupado["coef_pooled"],
            "se_unica": se_unica,
            "se_pooled": agrupado["se_pooled"],
            "inflacao_se_pct": round(100 * (agrupado["se_pooled"] / se_unica - 1), 4),
            "p_inflacao_apenas": p_inflacao_apenas,
            "veredito_muda_apenas_pela_inflacao_do_erro_padrao": bool(
                (p_unica_sem_flag < ALFA) != (p_inflacao_apenas < ALFA)
            ),
            "n_imputacoes_p_menor_alfa": int(sum(1 for p in p_por_imputacao[c] if p < ALFA)),
            "variancia_entre": agrupado["variancia_entre"],
            "variancia_intra": agrupado["variancia_intra"],
            "graus_liberdade": agrupado["graus_liberdade"],
            "p_pooled_normal": agrupado["p_pooled_normal"],
        })

    diagnostico = {}
    for c in colunas_ausentes:
        observado = bruto[c].dropna()
        empilhado = np.concatenate(desenhos[c]) if desenhos[c] else np.array([np.nan])
        diagnostico[c] = {
            "n_ausentes": int(mascara_ausente[c].sum()),
            "pct_ausentes": round(100 * float(mascara_ausente[c].mean()), 3),
            "observado": {
                "media": float(observado.mean()),
                "sd": float(observado.std()),
                "min": float(observado.min()),
                "max": float(observado.max()),
            },
            "imputado": {
                "media": float(np.mean(empilhado)),
                "sd": float(np.std(empilhado)),
                "min": float(np.min(empilhado)),
                "max": float(np.max(empilhado)),
            },
            "pct_fora_do_intervalo_observado": round(
                100 * float(np.mean((empilhado < observado.min()) | (empilhado > observado.max()))), 3
            ),
        }

    return {
        "linhas": linhas,
        "colunas_imputadas": colunas_ausentes,
        "coef_unico": {c: float(resumo_unico.loc[c, "coef"]) for c in finalistas},
        "se_unico": {c: float(resumo_unico.loc[c, "se(coef)"]) for c in finalistas},
        "diagnostico": diagnostico,
        "n_imputacoes": M_IMPUTACOES,
    }


def ler_cox_hr(posicao: str):
    """Le ``cox_hr_{POS}.csv`` do pipeline (ajuste unico gravado pelos passos 11/12/13)."""
    caminho = COX_DIR / f"cox_hr_{posicao}.csv"
    if not caminho.exists():
        return None
    return pd.read_csv(caminho).set_index("covariate")


def comparar_com_cox_hr(posicao: str, ajuste: dict, tabela) -> dict:
    """Confere o ajuste unico reconstruido contra o ``cox_hr_{POS}.csv`` do pipeline."""
    caminho = COX_DIR / f"cox_hr_{posicao}.csv"
    if tabela is None:
        return {"arquivo": str(caminho.relative_to(ROOT)), "disponivel": False}
    diferencas = {}
    for c, coef in ajuste["coef_unico"].items():
        if c in tabela.index:
            diferencas[c] = abs(coef - float(tabela.loc[c, "coef"]))
    maximo = max(diferencas.values()) if diferencas else float("nan")
    return {
        "arquivo": str(caminho.relative_to(ROOT)),
        "disponivel": True,
        "covariaveis_comparadas": sorted(diferencas),
        "max_abs_diff_coef": float(maximo),
        "identico_tolerancia_1e-8": bool(maximo < 1e-8) if diferencas else False,
    }


# Script R usado pela sensibilidade de EP robusto e de empates. lifelines nao tem
# cluster no CoxTimeVaryingFitter (robust=True levanta NotImplementedError) nem
# Breslow, entao o modelo final e reajustado com survival::coxph, o mesmo padrao
# de Inferência/shared/run_schoenfeld_diagnostics.R.
SCRIPT_R_COX_FINAL = r'''
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages(library(survival))
df <- read.csv(args[[1]], stringsAsFactors = FALSE)
covs <- strsplit(args[[2]], ",", fixed = TRUE)[[1]]
rhs <- paste(covs, collapse = " + ")
f_base <- as.formula(paste0("Surv(start, stop, event) ~ ", rhs))
f_cluster <- as.formula(paste0("Surv(start, stop, event) ~ ", rhs, " + cluster(player_id)"))
fit_efron <- coxph(f_base, data = df, ties = "efron")
fit_cluster <- coxph(f_cluster, data = df, ties = "efron")
fit_breslow <- coxph(f_base, data = df, ties = "breslow")
# fit_cluster$var nao tem nomes de linha: indexa-se por posicao, com a ordem dos coeficientes.
idx_cluster <- match(covs, names(coef(fit_cluster)))
idx_efron <- match(covs, names(coef(fit_efron)))
idx_breslow <- match(covs, names(coef(fit_breslow)))
out <- data.frame(
  covariavel = covs,
  coef_efron = unname(coef(fit_efron)[idx_efron]),
  ep_modelo = unname(sqrt(diag(vcov(fit_efron)))[idx_efron]),
  coef_efron_cluster = unname(coef(fit_cluster)[idx_cluster]),
  ep_robusto = unname(sqrt(diag(fit_cluster$var))[idx_cluster]),
  coef_breslow = unname(coef(fit_breslow)[idx_breslow]),
  ep_breslow = unname(sqrt(diag(vcov(fit_breslow)))[idx_breslow])
)
if (any(is.na(out$ep_robusto)) || any(is.na(out$ep_modelo))) stop("EP ausente no ajuste R")
write.csv(out, args[[3]], row.names = FALSE)
'''


# Modelo de tempo discreto (Prentice-Gloeckler): GLM binomial com ligacao cloglog
# sobre os intervalos pessoa-periodo, com efeito fixo por grupo de tempo. Os
# coeficientes sao log-HR do modelo de riscos proporcionais agrupado.
SCRIPT_R_TEMPO_DISCRETO = r'''
args <- commandArgs(trailingOnly = TRUE)
df <- read.csv(args[[1]], stringsAsFactors = FALSE)
covs <- strsplit(args[[2]], ",", fixed = TRUE)[[1]]
df$grupo <- factor(df$grupo)
f <- as.formula(paste0("event ~ ", paste(c(covs, "grupo"), collapse = " + ")))
fit <- glm(f, data = df, family = binomial(link = "cloglog"))
if (!fit$converged) stop("glm cloglog nao convergiu")
s <- summary(fit)$coefficients
if (any(!is.finite(s[, "Std. Error"])) || max(s[, "Std. Error"]) > 50) stop("separacao: EP explosivo")
out <- data.frame(
  covariavel = covs,
  coef_cloglog = unname(s[covs, "Estimate"]),
  ep_cloglog = unname(s[covs, "Std. Error"])
)
if (any(is.na(out$coef_cloglog)) || any(is.na(out$ep_cloglog))) stop("coeficiente ausente no glm")
write.csv(out, args[[3]], row.names = FALSE)
'''

MIN_EVENTOS_POR_TEMPO = 3


def agrupa_tempos(dados: pd.DataFrame, minimo: int = MIN_EVENTOS_POR_TEMPO):
    """Agrupa tempos inteiros consecutivos ate cada grupo ter ``minimo`` eventos.

    Tempos com menos de ``minimo`` eventos sao somados ao tempo seguinte. O resto
    final que ainda nao chegou ao minimo e anexado ao ultimo grupo fechado. Assim,
    nenhum grupo fica sem eventos (o que levaria a separacao no efeito de tempo).
    """
    eventos = dados.loc[dados["event"] == 1].groupby("stop").size()
    grupos, atual, n = [], [], 0
    for t in sorted(dados["stop"].unique()):
        atual.append(t)
        n += int(eventos.get(t, 0))
        if n >= minimo:
            grupos.append(atual)
            atual, n = [], 0
    if atual:
        if grupos:
            grupos[-1].extend(atual)
        else:
            grupos.append(atual)
    mapa = {t: i + 1 for i, g in enumerate(grupos) for t in g}
    resumo = [
        {"tempos": f"{min(g):g}-{max(g):g}", "eventos": int(sum(eventos.get(t, 0) for t in g))}
        for g in grupos
    ]
    return mapa, resumo


def ajusta_cloglog_r(dados: pd.DataFrame, covariadas: list[str], grupo: pd.Series) -> pd.DataFrame:
    """GLM cloglog no R com efeito fixo por ``grupo``; devolve coef e EP das covariaveis."""
    pessoa_periodo = dados[["player_id", "start", "stop", "event"] + covariadas].copy()
    pessoa_periodo["grupo"] = grupo.to_numpy()
    with tempfile.TemporaryDirectory() as pasta:
        caminho_csv = Path(pasta) / "pessoa_periodo.csv"
        caminho_r = Path(pasta) / "tempo_discreto.R"
        caminho_saida = Path(pasta) / "saida.csv"
        pessoa_periodo.to_csv(caminho_csv, index=False)
        caminho_r.write_text(SCRIPT_R_TEMPO_DISCRETO, encoding="utf-8")
        subprocess.run(
            [rscript(), str(caminho_r), str(caminho_csv), ",".join(covariadas), str(caminho_saida)],
            check=True, capture_output=True, text=True,
        )
        return pd.read_csv(caminho_saida).set_index("covariavel")


def sensibilidade_tempo_discreto(dados: pd.DataFrame, covariadas: list[str], posicao: str,
                                 empates: pd.DataFrame) -> dict:
    """Cloglog de tempo discreto no modelo final, lado a lado com Efron e Breslow.

    Ajuste principal: efeito fixo por ano (stop inteiro, sem agrupamento). Se o GLM
    por ano nao convergir nesta posicao, o principal passa a ser o agrupado (tempos
    com menos de ``MIN_EVENTOS_POR_TEMPO`` eventos somados ao vizinho), e o motivo
    fica em ``modelo_principal``. O agrupado e sempre reportado como coluna extra.
    """
    mapa, grupos = agrupa_tempos(dados)
    grupo_agrupado = dados["stop"].map(mapa).astype(int)
    r_agrupado = ajusta_cloglog_r(dados, covariadas, grupo_agrupado)
    try:
        r_principal = ajusta_cloglog_r(dados, covariadas, dados["stop"].astype(int))
        modelo_principal = "efeito por ano (stop inteiro)"
        n_niveis_principal = int(dados["stop"].nunique())
    except subprocess.CalledProcessError as erro:
        linhas_erro = (erro.stderr or "").strip().splitlines() or ["erro desconhecido"]
        motivo = next((l for l in reversed(linhas_erro) if l.startswith("Erro")), linhas_erro[-1])
        print(f"  {posicao}: cloglog por ano nao convergiu ({motivo}); principal = agrupado", flush=True)
        r_principal = r_agrupado
        modelo_principal = f"agrupado (fallback: efeito por ano nao convergiu: {motivo})"
        n_niveis_principal = len(grupos)
    emp = empates.set_index("covariavel")
    linhas = []
    for c in covariadas:
        coef = float(r_principal.loc[c, "coef_cloglog"])
        coef_ag = float(r_agrupado.loc[c, "coef_cloglog"])
        linhas.append({
            "covariavel": c,
            "HR_efron": float(emp.loc[c, "HR_efron"]),
            "p_efron": float(emp.loc[c, "p_efron"]),
            "HR_cloglog": round(float(np.exp(coef)), 6),
            "p_cloglog": p_valor_wald(coef, float(r_principal.loc[c, "ep_cloglog"]), float("inf")),
            "HR_breslow": float(emp.loc[c, "HR_breslow"]),
            "p_breslow": float(emp.loc[c, "p_breslow"]),
            "hr_cloglog_agrupado": round(float(np.exp(coef_ag)), 6),
            "p_cloglog_agrupado": p_valor_wald(coef_ag, float(r_agrupado.loc[c, "ep_cloglog"]), float("inf")),
        })
    tabela = pd.DataFrame(linhas)
    muda = tabela.loc[(tabela["p_efron"] < ALFA) != (tabela["p_cloglog"] < ALFA), "covariavel"].tolist()
    muda_ag = tabela.loc[(tabela["p_efron"] < ALFA) != (tabela["p_cloglog_agrupado"] < ALFA), "covariavel"].tolist()
    return {
        "tabela": tabela,
        "modelo_principal": modelo_principal,
        "n_niveis_tempo_principal": n_niveis_principal,
        "grupos_tempo": grupos,
        "n_linhas_pessoa_periodo": int(len(dados)),
        "covariaveis_mudam_de_lado": muda,
        "covariaveis_mudam_de_lado_agrupado": muda_ag,
    }


def rscript() -> str:
    return shutil.which("Rscript") or "/usr/bin/Rscript"


def escalar_final(df_imputado: pd.DataFrame, covariadas: list[str]) -> pd.DataFrame:
    """Colunas de ID/tempo mais as covariaveis finais escalonadas (como no passo 11)."""
    escala = pd.DataFrame(
        StandardScaler().fit_transform(df_imputado[covariadas]),
        columns=covariadas, index=df_imputado.index,
    )
    return pd.concat([df_imputado[["player_id", "start", "stop", "event"]], escala], axis=1)


def ajustar_final(painel: pd.DataFrame, covariadas: list[str]):
    """Modelo final sobre um painel bruto, com a imputacao por mediana do pipeline."""
    cov_cols = [c for c in painel.columns if c not in ID_COLS + TIME_COLS and c != "draft_round"]
    estatisticas = fit_imputation_stats(painel, cov_cols)
    imputado = apply_imputation_stats(painel, cov_cols, estatisticas)
    return ajustar_cox(escalar_final(imputado, covariadas), covariadas)


def sensibilidade_cox_final(df_imputado: pd.DataFrame, covariadas: list[str], posicao: str) -> dict:
    """EP robusto por jogador (cluster) e empates Breslow x Efron no modelo final.

    Os coeficientes do Efron em R sao conferidos contra o lifelines do passo 51.
    """
    faltando = [c for c in covariadas if c not in df_imputado.columns]
    if faltando:
        raise KeyError(f"{posicao}: covariaveis finais ausentes do painel: {faltando}")
    dados = escalar_final(df_imputado, covariadas)
    coef_lifelines = ajustar_cox(dados, covariadas).summary["coef"]

    entrada = dados[["player_id", "start", "stop", "event"] + covariadas]
    with tempfile.TemporaryDirectory() as pasta:
        caminho_csv = Path(pasta) / "dados_cox.csv"
        caminho_r = Path(pasta) / "cox_final.R"
        caminho_saida = Path(pasta) / "saida.csv"
        entrada.to_csv(caminho_csv, index=False)
        caminho_r.write_text(SCRIPT_R_COX_FINAL, encoding="utf-8")
        subprocess.run(
            [rscript(), str(caminho_r), str(caminho_csv), ",".join(covariadas), str(caminho_saida)],
            check=True, capture_output=True, text=True,
        )
        r = pd.read_csv(caminho_saida).set_index("covariavel")

    diferenca_lifelines = float(np.max(np.abs(r.loc[covariadas, "coef_efron"].to_numpy()
                                              - coef_lifelines.loc[covariadas].to_numpy())))
    if r.loc[covariadas, ["ep_modelo", "ep_robusto", "ep_breslow"]].isna().any().any():
        raise ValueError(f"{posicao}: erro-padrao ausente no ajuste do R")

    linhas_ep, linhas_empates = [], []
    for c in covariadas:
        coef = float(r.loc[c, "coef_efron"])
        ep_mod = float(r.loc[c, "ep_modelo"])
        ep_rob = float(r.loc[c, "ep_robusto"])
        p_mod = p_valor_wald(coef, ep_mod, float("inf"))
        p_rob = p_valor_wald(coef, ep_rob, float("inf"))
        linhas_ep.append({
            "covariavel": c,
            "HR": round(float(np.exp(coef)), 6),
            "EP_modelo": ep_mod,
            "EP_robusto": ep_rob,
            "razao_EP_robusto_sobre_modelo": round(ep_rob / ep_mod, 4),
            "p_modelo": p_mod,
            "p_robusto": p_rob,
            "veredito_muda": bool((p_mod < ALFA) != (p_rob < ALFA)),
        })
        coef_b = float(r.loc[c, "coef_breslow"])
        p_efron = p_mod
        p_breslow = p_valor_wald(coef_b, float(r.loc[c, "ep_breslow"]), float("inf"))
        linhas_empates.append({
            "covariavel": c,
            "HR_efron": round(float(np.exp(coef)), 6),
            "HR_breslow": round(float(np.exp(coef_b)), 6),
            "coef_efron": coef,
            "coef_breslow": coef_b,
            "p_efron": p_efron,
            "p_breslow": p_breslow,
            "veredito_muda": bool((p_efron < ALFA) != (p_breslow < ALFA)),
        })

    # Eventos em tempo compartilhado com outro evento (tempos sao anos inteiros).
    eventos_tempo = dados.loc[dados["event"] == 1, "stop"]
    contagem = eventos_tempo.map(eventos_tempo.value_counts())
    n_eventos = int(len(eventos_tempo))
    n_empatados = int((contagem > 1).sum())

    empates = pd.DataFrame(linhas_empates)
    tempo_discreto = sensibilidade_tempo_discreto(dados, covariadas, posicao, empates)

    return {
        "ep_robusto": pd.DataFrame(linhas_ep),
        "empates": empates,
        "tempo_discreto": tempo_discreto,
        "n_linhas": int(len(dados)),
        "n_eventos": n_eventos,
        "eventos_empatados": n_empatados,
        "pct_eventos_empatados": round(100 * n_empatados / n_eventos, 2) if n_eventos else float("nan"),
        "validacao_efron_r_vs_lifelines_max_abs_diff": diferenca_lifelines,
    }


# ------------------------------------------- covariaveis correntes x defasadas (t-1)
# Covariaveis fixas ou defasadas por construcao: sao medidas antes da temporada t
# (historico, draft, altura, lag de carreira ate t-1) ou nao variam no tempo. As
# demais medem a propria temporada t, o que pode ser efeito de estar sendo cortado.
FIXAS_DEFASADAS = {
    "height": "medida pre-carreira",
    "weight": "medida pre-carreira",
    "draft_ovr": "draft (fixo)",
    "draft_round": "draft (fixo)",
    "undrafted_flag": "draft (fixo)",
    "unobs_career_years": "anos de carreira sem estatistica antes da 1a linha",
    "pre_window_veteran": "carreira iniciada antes de 2012",
    "unobserved_history_flag": "historico anterior a janela observada",
    "first_observed_season": "hiato/historico",
    "within_window_gap_years_before_t": "hiato acumulado ate t-1",
    "had_prior_gap_flag": "hiato ate t-1",
}


def classifica_covariavel(cov: str) -> tuple[str, str]:
    """Tipo ('temporada corrente' ou 'fixa/defasada') e motivo de cada covariavel final."""
    if cov.endswith("_missing"):
        return "fixa/defasada", "flag de missing (mecanica, fixa no tempo)"
    if cov.startswith("observed_career_lag_"):
        return "fixa/defasada", "lag de carreira observada (acumulado ate t-1)"
    if cov in FIXAS_DEFASADAS:
        return "fixa/defasada", FIXAS_DEFASADAS[cov]
    return "temporada corrente", "medida na propria temporada t"


def painel_com_t_menos_1(painel: pd.DataFrame, correntes: list[str]) -> pd.DataFrame:
    """Linhas com linha do mesmo jogador em t-1 (season - 1), com as correntes de t-1.

    O t-1 e por temporada: a linha anterior precisa existir no painel com season - 1.
    As colunas de t-1 saem com o sufixo ``__t1``.
    """
    anterior = painel[["player_id", "season"] + correntes].copy()
    anterior["season"] = anterior["season"] + 1
    anterior = anterior.rename(columns={c: f"{c}__t1" for c in correntes})
    anterior["__tem_t1"] = 1
    junta = painel.merge(anterior, on=["player_id", "season"], how="left")
    return junta[junta["__tem_t1"] == 1].drop(columns="__tem_t1").reset_index(drop=True)


def sensibilidade_covariaveis_defasadas(painel: pd.DataFrame, posicao: str) -> dict:
    """Cox final com covariaveis correntes (t) e com as correntes trocadas por t-1.

    Amostra comum: linhas com t-1 disponivel. Imputacao por mediana do pipeline (painel
    inteiro) e escalonamento como no passo 11; a versao defasada usa a escala da corrente
    (mesma unidade), de modo que o HR de cada covariavel e comparavel. Fixas/defasadas
    ficam iguais nos dois ajustes. A referencia oficial e cox_hr_{POS}.csv (amostra completa).
    """
    finais = json.loads((COX_DIR / f"final_covs_{posicao}.json").read_text(encoding="utf-8"))
    faltando = [c for c in finais if c not in painel.columns]
    if faltando:
        raise KeyError(f"{posicao}: covariaveis finais ausentes do painel: {faltando}")
    cov_cols = [c for c in painel.columns if c not in ID_COLS + TIME_COLS and c != "draft_round"]
    estatisticas = fit_imputation_stats(painel, cov_cols)

    correntes = [c for c in finais if classifica_covariavel(c)[0] == "temporada corrente"]
    comum = painel_com_t_menos_1(painel, correntes)
    atual = apply_imputation_stats(comum, cov_cols, estatisticas)
    defasado_bruto = comum.copy()
    for c in correntes:
        defasado_bruto[c] = comum[f"{c}__t1"]
    defasado = apply_imputation_stats(defasado_bruto, cov_cols, estatisticas)

    escalador = StandardScaler().fit(atual[finais])
    cabecalho = ["player_id", "start", "stop", "event"]
    dados_atual = pd.concat([
        comum[cabecalho],
        pd.DataFrame(escalador.transform(atual[finais]), columns=finais, index=comum.index),
    ], axis=1)
    dados_defasado = pd.concat([
        comum[cabecalho],
        pd.DataFrame(escalador.transform(defasado[finais]), columns=finais, index=comum.index),
    ], axis=1)
    ajuste_atual = ajustar_cox(dados_atual, finais).summary
    ajuste_defasado = ajustar_cox(dados_defasado, finais).summary

    # Modelo oficial na amostra completa, pela mesma cadeia: serve de validacao contra cox_hr.
    ajuste_completo = ajustar_final(painel, finais).summary
    tabela_hr = ler_cox_hr(posicao)
    validacao = float("nan")
    if tabela_hr is not None:
        validacao = float(np.max(np.abs(ajuste_completo.loc[finais, "coef"].to_numpy()
                                        - tabela_hr.loc[finais, "coef"].to_numpy())))

    linhas = []
    for c in finais:
        tipo, motivo = classifica_covariavel(c)
        a, d = ajuste_atual.loc[c], ajuste_defasado.loc[c]
        o = ajuste_completo.loc[c]
        linha = {
            "posicao": posicao,
            "covariavel": c,
            "tipo": tipo,
            "motivo_classificacao": motivo,
            "n_linhas_amostra_comum": int(len(comum)),
            "n_eventos_amostra_comum": int(comum["event"].sum()),
            "HR_corrente": round(float(np.exp(a["coef"])), 6),
            "IC95_inf_corrente": round(float(a["exp(coef) lower 95%"]), 6),
            "IC95_sup_corrente": round(float(a["exp(coef) upper 95%"]), 6),
            "p_corrente": float(a["p"]),
            "HR_defasada_t1": round(float(np.exp(d["coef"])), 6),
            "IC95_inf_defasada_t1": round(float(d["exp(coef) lower 95%"]), 6),
            "IC95_sup_defasada_t1": round(float(d["exp(coef) upper 95%"]), 6),
            "p_defasada_t1": float(d["p"]),
            "veredito_muda_corrente_vs_defasada": bool((a["p"] < ALFA) != (d["p"] < ALFA)),
            "n_linhas_amostra_completa": int(len(painel)),
            "n_eventos_amostra_completa": int(painel["event"].sum()),
            "HR_oficial_completa": round(float(np.exp(o["coef"])), 6),
            "p_oficial_completa": float(o["p"]),
        }
        if tabela_hr is not None and c in tabela_hr.index:
            linha["HR_oficial_cox_hr"] = round(float(tabela_hr.loc[c, "HR"]), 6)
            # cox_hr_{POS}.csv guarda os limites do IC na escala do coeficiente (log HR).
            linha["IC95_inf_oficial_cox_hr"] = round(float(np.exp(tabela_hr.loc[c, "CI_lower"])), 6)
            linha["IC95_sup_oficial_cox_hr"] = round(float(np.exp(tabela_hr.loc[c, "CI_upper"])), 6)
            linha["p_oficial_cox_hr"] = float(tabela_hr.loc[c, "p_value"])
        linhas.append(linha)

    tabela = pd.DataFrame(linhas)
    return {
        "tabela": tabela,
        "correntes": correntes,
        "fixas_defasadas": [c for c in finais if c not in correntes],
        "n_linhas_completa": int(len(painel)),
        "n_eventos_completa": int(painel["event"].sum()),
        "n_linhas_amostra_comum": int(len(comum)),
        "n_eventos_amostra_comum": int(comum["event"].sum()),
        "n_linhas_sem_t1": int(len(painel) - len(comum)),
        "validacao_completa_vs_cox_hr_max_abs_diff": validacao,
        "covariaveis_mudam_veredito": tabela.loc[tabela["veredito_muda_corrente_vs_defasada"], "covariavel"].tolist(),
    }


def sensibilidade_passer_rating_zero(base: pd.DataFrame) -> dict:
    """QB: HR do passer_rating com as 26 linhas sem tentativas tratadas de 3 formas."""
    painel = construir_painel(base, "QB", LIMIAR_PADRAO, "A")
    covariadas = json.loads((COX_DIR / "final_covs_QB.json").read_text(encoding="utf-8"))
    ausente = painel["passer_rating"].isna()
    cenarios = {
        "a_mediana": ("mediana global do pipeline (atual)", painel),
        "b_zero": ("valor 0 nas linhas sem tentativas", painel.assign(
            passer_rating=painel["passer_rating"].where(~ausente, 0.0))),
        "c_excluir": ("exclui as linhas sem tentativas", painel[~ausente]),
    }
    linhas = []
    ajuste_a = None
    for chave, (descricao, df) in cenarios.items():
        modelo = ajustar_final(df, covariadas)
        if chave == "a_mediana":
            ajuste_a = modelo.summary
        r = modelo.summary.loc["passer_rating"]
        linhas.append({
            "cenario": chave,
            "descricao": descricao,
            "n_linhas": int(len(df)),
            "n_eventos": int(df["event"].sum()),
            "n_linhas_sem_tentativas_no_modelo": int(ausente.sum()) if chave != "c_excluir" else 0,
            "HR_passer_rating": round(float(np.exp(r["coef"])), 6),
            "coef_passer_rating": float(r["coef"]),
            "EP_passer_rating": float(r["se(coef)"]),
            "p_passer_rating": float(r["p"]),
        })

    # O cenario (a) deve reproduzir o cox_hr_QB.csv publicado (que tem a flag).
    tabela_hr = ler_cox_hr("QB")
    maximo = float("nan")
    if tabela_hr is not None:
        comuns = [c for c in covariadas if c in tabela_hr.index]
        maximo = float(np.max(np.abs(ajuste_a.loc[comuns, "coef"].to_numpy()
                                     - tabela_hr.loc[comuns, "coef"].to_numpy())))
    return {
        "tabela": pd.DataFrame(linhas),
        "linhas_sem_tentativas": int(ausente.sum()),
        "validacao_cenario_a_vs_cox_hr_max_abs_diff": maximo,
    }


def analisar_posicao(posicao: str, backward) -> dict:
    """Roda as tres sensibilidades para uma posicao."""
    tempos = {}
    t0 = time.time()
    base = carregar_base(posicao)
    tempos["base_s"] = round(time.time() - t0, 2)
    print(f"  base limpa: {base.shape[0]} linhas x {base.shape[1]} colunas "
          f"({tempos['base_s']:.1f}s)", flush=True)

    # ---------- 1. limiar do denominador das taxas ----------
    resultados_limiar = {}
    for limiar in LIMIARES_TAXA:
        t = time.time()
        painel = construir_painel(base, posicao, limiar, "A")
        selecao = rodar_selecao(painel, backward)
        resultados_limiar[limiar] = selecao
        print(f"  limiar {limiar:>3}: {selecao['n_linhas']} linhas, "
              f"{selecao['n_eventos']} eventos, {len(selecao['substantivas'])} covariaveis "
              f"({time.time() - t:.1f}s)", flush=True)
    tempos["limiares_s"] = round(time.time() - t0 - tempos["base_s"], 2)

    # O limiar padrao e o proprio pipeline: reutilizado no teste de aceitacao,
    # no Jaccard de referencia e no cenario A de codificacao de draft.
    padrao = resultados_limiar[LIMIAR_PADRAO]
    esperado = json.loads((COX_DIR / f"final_covs_{posicao}.json").read_text(encoding="utf-8"))
    esperado_subst = substantivas(esperado)
    reproduz = padrao["substantivas"] == esperado_subst

    # ---------- 2. codificacoes alternativas para nao draftados ----------
    t = time.time()
    resultados_cenario = {"A": padrao}
    for cenario in ("B", "C", "D"):
        painel = construir_painel(base, posicao, LIMIAR_PADRAO, cenario)
        resultados_cenario[cenario] = rodar_selecao(painel, backward)
        print(f"  cenario {cenario}: {len(resultados_cenario[cenario]['substantivas'])} covariaveis",
              flush=True)
    tempos["cenarios_s"] = round(time.time() - t, 2)

    # ---------- 3. imputacao multipla no modelo final ----------
    t = time.time()
    painel_padrao = construir_painel(base, posicao, LIMIAR_PADRAO, "A")
    resultado_mi = imputacao_multipla(painel_padrao, padrao, posicao)
    validacao = comparar_com_cox_hr(posicao, resultado_mi, ler_cox_hr(posicao))
    tempos["imputacao_s"] = round(time.time() - t, 2)

    # ---------- 4. covariaveis de temporada corrente x defasadas (t-1) ----------
    t = time.time()
    defasadas = sensibilidade_covariaveis_defasadas(painel_padrao, posicao)
    tempos["defasadas_s"] = round(time.time() - t, 2)
    tempos["total_s"] = round(time.time() - t0, 2)

    return {
        "limiares": resultados_limiar,
        "cenarios": resultados_cenario,
        "padrao": padrao,
        "esperado_subst": esperado_subst,
        "esperado_json": esperado,
        "reproduz": reproduz,
        "imputacao": resultado_mi,
        "validacao": validacao,
        "defasadas": defasadas,
        "tempos": tempos,
        "n_candidatas": padrao["n_candidatas"],
        "limiar_p": padrao["limiar_p"],
        "max_vars": padrao["max_vars"],
    }


def main() -> int:
    inicio = time.time()
    np.random.seed(SEMENTE_BASE)
    SAIDA.mkdir(parents=True, exist_ok=True)
    faltando = [p for p in POSICOES if not (COX_DIR / f"final_covs_{p}.json").exists()]
    if faltando:
        raise FileNotFoundError(
            "final_covs_{POS}.json ausente para " + ", ".join(faltando)
            + ": rode os passos 11/12/13 antes deste."
        )

    print("Sensibilidade inferencial do Cox (limiar da taxa, nao draftados, imputacao multipla)")
    backward = carregar_backward_cox_tdv()
    print(f"backward_cox_tdv carregada do passo 11: {PASSO_COX.relative_to(ROOT)}", flush=True)

    analises = {}
    for posicao in POSICOES:
        print(f"\n=== {posicao} ===", flush=True)
        analises[posicao] = analisar_posicao(posicao, backward)

    print("\n=== Cox final: EP robusto por jogador e empates (Breslow x Efron) ===", flush=True)
    sens_cox = {}
    for posicao in POSICOES:
        sens_cox[posicao] = sensibilidade_cox_final(
            analises[posicao]["padrao"]["df_imputado"], analises[posicao]["esperado_json"], posicao
        )
        print(f"  {posicao}: eventos empatados {sens_cox[posicao]['eventos_empatados']}/"
              f"{sens_cox[posicao]['n_eventos']} | validacao Efron R vs lifelines = "
              f"{sens_cox[posicao]['validacao_efron_r_vs_lifelines_max_abs_diff']:.2e}", flush=True)
        mudam = sens_cox[posicao]["tempo_discreto"]["covariaveis_mudam_de_lado"]
        mudam_ag = sens_cox[posicao]["tempo_discreto"]["covariaveis_mudam_de_lado_agrupado"]
        print(f"  {posicao}: cloglog por ano ({sens_cox[posicao]['tempo_discreto']['n_niveis_tempo_principal']} "
              f"niveis) mudam de lado em 0,05 vs Efron: {len(mudam)} {mudam or ''} | agrupado "
              f"({len(sens_cox[posicao]['tempo_discreto']['grupos_tempo'])} grupos): {len(mudam_ag)} "
              f"{mudam_ag or ''}", flush=True)

    print("\n=== QB: passer_rating com 0 tentativas ===", flush=True)
    sens_passer = sensibilidade_passer_rating_zero(carregar_base("QB"))
    print(sens_passer["tabela"].to_string(index=False), flush=True)

    # ---------------------------------------------------------------- artefatos
    linhas_limiares = []
    for posicao in POSICOES:
        analise = analises[posicao]
        referencia = analise["limiares"][LIMIAR_PADRAO]["substantivas"]
        for limiar in LIMIARES_TAXA:
            selecao = analise["limiares"][limiar]
            linhas_limiares.append({
                "limiar": limiar,
                "posicao": posicao,
                "n_linhas": selecao["n_linhas"],
                "n_eventos": selecao["n_eventos"],
                "n_covariaveis": len(selecao["substantivas"]),
                "covariaveis": ";".join(selecao["substantivas"]),
                "jaccard_vs_50": round(jaccard(selecao["substantivas"], referencia), 6),
            })
    tabela_limiares = pd.DataFrame(linhas_limiares, columns=[
        "limiar", "posicao", "n_linhas", "n_eventos", "n_covariaveis",
        "covariaveis", "jaccard_vs_50",
    ])
    tabela_limiares.to_csv(SAIDA / "limiares_taxa.csv", index=False)

    linhas_cenarios = []
    for posicao in POSICOES:
        analise = analises[posicao]
        referencia = analise["cenarios"]["A"]["substantivas"]
        for cenario in ("A", "B", "C", "D"):
            selecao = analise["cenarios"][cenario]
            linhas_cenarios.append({
                "cenario": cenario,
                "posicao": posicao,
                "n_covariaveis": len(selecao["substantivas"]),
                "covariaveis": ";".join(selecao["substantivas"]),
                "jaccard_vs_padrao": round(jaccard(selecao["substantivas"], referencia), 6),
            })
    tabela_cenarios = pd.DataFrame(linhas_cenarios, columns=[
        "cenario", "posicao", "n_covariaveis", "covariaveis", "jaccard_vs_padrao",
    ])
    tabela_cenarios.to_csv(SAIDA / "undrafted_cenarios.csv", index=False)

    linhas_mi = []
    for posicao in POSICOES:
        for linha in analises[posicao]["imputacao"]["linhas"]:
            linha_padrao = {
                "posicao": linha["posicao"],
                "covariavel": linha["covariavel"],
                "hr_unica_original": round(linha["hr_unica_original"], 6),
                "p_unica_original": linha["p_unica_original"],
                "hr_unica_sem_flag": round(linha["hr_unica_sem_flag"], 6),
                "p_unica_sem_flag": linha["p_unica_sem_flag"],
                "hr_pooled": round(linha["hr_pooled"], 6),
                "p_pooled": linha["p_pooled"],
                "significancia_inalterada": linha["significancia_inalterada"],
            }
            linhas_mi.append(linha_padrao)
    tabela_mi = pd.DataFrame(linhas_mi, columns=[
        "posicao", "covariavel", "hr_unica_original", "p_unica_original",
        "hr_unica_sem_flag", "p_unica_sem_flag", "hr_pooled", "p_pooled",
        "significancia_inalterada",
    ])
    tabela_mi.to_csv(SAIDA / "imputacao_multipla.csv", index=False)

    for posicao in POSICOES:
        sens_cox[posicao]["ep_robusto"].to_csv(SAIDA / f"sens_ep_robusto_{posicao}.csv", index=False)
        sens_cox[posicao]["empates"].to_csv(SAIDA / f"sens_empates_{posicao}.csv", index=False)
        sens_cox[posicao]["tempo_discreto"]["tabela"].to_csv(
            SAIDA / f"sens_tempo_discreto_{posicao}.csv", index=False)
    sens_passer["tabela"].to_csv(SAIDA / "sens_passer_rating_zero_QB.csv", index=False)
    for posicao in POSICOES:
        analises[posicao]["defasadas"]["tabela"].to_csv(
            SAIDA / f"sens_covariaveis_defasadas_{posicao}.csv", index=False)

    # ------------------------------------------------------------------ resumo
    aceitacao = {}
    for posicao in POSICOES:
        analise = analises[posicao]
        aceitacao[posicao] = {
            "reproduz_selecao_do_pipeline": bool(analise["reproduz"]),
            "selecao_reconstruida_limiar_50": analise["padrao"]["substantivas"],
            "final_covs_json": analise["esperado_json"],
            "final_covs_json_sem_missing": analise["esperado_subst"],
            "apenas_no_json": sorted(set(analise["esperado_subst"]) - set(analise["padrao"]["substantivas"])),
            "apenas_na_reconstrucao": sorted(set(analise["padrao"]["substantivas"]) - set(analise["esperado_subst"])),
        }

    jaccard_limiares = {}
    adjacentes_estaveis = {}
    for posicao in POSICOES:
        analise = analises[posicao]
        jaccard_limiares[posicao] = {
            str(limiar): round(
                jaccard(analise["limiares"][limiar]["substantivas"],
                        analise["limiares"][LIMIAR_PADRAO]["substantivas"]), 6)
            for limiar in LIMIARES_TAXA
        }
        pares = list(zip(LIMIARES_TAXA, LIMIARES_TAXA[1:]))
        valores = [
            jaccard(analise["limiares"][a]["substantivas"], analise["limiares"][b]["substantivas"])
            for a, b in pares
        ]
        adjacentes_estaveis[posicao] = {
            "jaccard_minimo_entre_limiares_adjacentes": round(min(valores), 6),
            "todos_os_pares_adjacentes_identicos": bool(all(v == 1.0 for v in valores)),
        }

    jaccard_cenarios = {}
    for posicao in POSICOES:
        analise = analises[posicao]
        referencia = analise["cenarios"]["A"]["substantivas"]
        jaccard_cenarios[posicao] = {
            cenario: {
                "jaccard_vs_A": round(jaccard(analise["cenarios"][cenario]["substantivas"], referencia), 6),
                "covariaveis": analise["cenarios"][cenario]["substantivas"],
            }
            for cenario in ("A", "B", "C", "D")
        }

    mi_resumo = {}
    for posicao in POSICOES:
        analise = analises[posicao]
        linhas = analise["imputacao"]["linhas"]
        mi_resumo[posicao] = {
            "colunas_imputadas": analise["imputacao"]["colunas_imputadas"],
            "n_imputacoes": analise["imputacao"]["n_imputacoes"],
            "n_covariaveis": len(linhas),
            "n_verditos_alterados": int(sum(1 for linha in linhas if not linha["significancia_inalterada"])),
            "verditos_alterados": [
                linha["covariavel"] for linha in linhas if not linha["significancia_inalterada"]
            ],
            "inflacao_do_erro_padrao_pct": {
                linha["covariavel"]: linha["inflacao_se_pct"] for linha in linhas
            },
            "veredito_mudaria_apenas_pela_inflacao_do_erro_padrao": bool(
                any(linha["veredito_muda_apenas_pela_inflacao_do_erro_padrao"] for linha in linhas)
            ),
            "diagnostico_imputacao": analise["imputacao"]["diagnostico"],
            "covariaveis": [
                {
                    "covariavel": linha["covariavel"],
                    "hr_unica_original": round(linha["hr_unica_original"], 6),
                    "p_unica_original": linha["p_unica_original"],
                    "hr_unica_sem_flag": round(linha["hr_unica_sem_flag"], 6),
                    "p_unica_sem_flag": linha["p_unica_sem_flag"],
                    "hr_pooled": round(linha["hr_pooled"], 6),
                    "p_pooled": linha["p_pooled"],
                    "p_pooled_normal": linha["p_pooled_normal"],
                    "p_inflacao_apenas": linha["p_inflacao_apenas"],
                    "coef_unico": linha["coef_unico"],
                    "coef_pooled": linha["coef_pooled"],
                    "se_unica": linha["se_unica"],
                    "se_pooled": linha["se_pooled"],
                    "inflacao_se_pct": linha["inflacao_se_pct"],
                    "n_imputacoes_p_menor_alfa": linha["n_imputacoes_p_menor_alfa"],
                    "significancia_inalterada": linha["significancia_inalterada"],
                }
                for linha in linhas
            ],
        }

    tempo_total = round(time.time() - inicio, 2)
    algum_verdito_alterado = any(
        mi_resumo[p]["n_verditos_alterados"] > 0 for p in POSICOES
    )

    # Ligacao explicita entre cada afirmacao da escrita e a evidencia gerada.
    afirmacoes = {
        "1_limiar_das_taxas_entre_25_e_100": {
            "afirmacao": (
                "cortes entre 25 e 100 mantiveram estavel a selecao de covariaveis do "
                "Cox entre limiares adjacentes"
            ),
            "teste": (
                f"{len(LIMIARES_TAXA)} limiares x 3 posicoes, com reconstrucao integral do "
                "painel e da cadeia de selecao; Jaccard entre limiares adjacentes e contra o limiar 50"
            ),
            "por_posicao": {
                p: {
                    "todos_os_pares_adjacentes_identicos": adjacentes_estaveis[p][
                        "todos_os_pares_adjacentes_identicos"
                    ],
                    "jaccard_minimo_entre_limiares_adjacentes": adjacentes_estaveis[p][
                        "jaccard_minimo_entre_limiares_adjacentes"
                    ],
                    "limiares_com_selecao_diferente_do_padrao_50": [
                        limiar for limiar in LIMIARES_TAXA if jaccard_limiares[p][str(limiar)] < 1.0
                    ],
                }
                for p in POSICOES
            },
            "sustentada_em_todas_as_posicoes": bool(
                all(adjacentes_estaveis[p]["todos_os_pares_adjacentes_identicos"] for p in POSICOES)
            ),
        },
        "2_codificacao_de_nao_draftados": {
            "afirmacao": (
                "codificacoes alternativas para nao draftados nao alteraram a selecao "
                "final em QB, RB ou WR"
            ),
            "teste": "4 cenarios (A a D) x 3 posicoes no limiar de taxa 50; Jaccard contra o cenario A",
            "por_posicao": {
                p: {
                    "jaccard_vs_A": {c: jaccard_cenarios[p][c]["jaccard_vs_A"] for c in ("A", "B", "C", "D")},
                    "cenarios_com_selecao_diferente_de_A": [
                        c for c in ("B", "C", "D") if jaccard_cenarios[p][c]["jaccard_vs_A"] < 1.0
                    ],
                }
                for p in POSICOES
            },
            "sustentada_em_todas_as_posicoes": bool(
                all(
                    all(jaccard_cenarios[p][c]["jaccard_vs_A"] == 1.0 for c in ("B", "C", "D"))
                    for p in POSICOES
                )
            ),
        },
        "3_imputacao_multipla_e_alfa_005": {
            "afirmacao": (
                "analise exploratoria condicional da imputacao multipla, nao uma demonstracao "
                "formal de compatibilidade com White–Royston ou de robustez inferencial"
            ),
            "teste": (
                "m = 10 imputacoes por IterativeImputer nas covariaveis finais com ausentes, "
                "ajuste Cox por imputacao e pooling de Rubin; compara descritivamente p < 0,05 "
                "do ajuste unico like-for-like (sem flags _missing) com p < 0,05 agrupado. "
                "A comparacao e condicional ao conjunto de covariaveis selecionado no modelo "
                "original; n_imputacoes_p_menor_alfa conta, entre m, quantos ajustes individuais "
                "tiveram p < 0,05, sem constituir criterio formal de robustez"
            ),
            "por_posicao": {
                p: {
                    "testavel": bool(mi_resumo[p]["colunas_imputadas"]),
                    "colunas_imputadas": mi_resumo[p]["colunas_imputadas"],
                    "verditos_alterados": mi_resumo[p]["verditos_alterados"],
                    "erro_padrao_inflado": any(
                        linha["se_pooled"] > linha["se_unica"]
                        for linha in analises[p]["imputacao"]["linhas"]
                    ),
                    "veredito_mudaria_apenas_pela_inflacao_do_erro_padrao": mi_resumo[p][
                        "veredito_mudaria_apenas_pela_inflacao_do_erro_padrao"
                    ],
                }
                for p in POSICOES
            },
            "sustentada_em_todas_as_posicoes": bool(not algum_verdito_alterado),
        },
    }
    limitacoes = [
        (
            "A amostra comum das covariaveis defasadas exige linha em t-1: jogadores com "
            "um unico episodio ou com hiato antes do evento saem dela, e a comparacao com "
            "o HR oficial (amostra completa) mede tambem esse efeito de selecao. A troca "
            "por t-1 nao corrige a endogeneidade por completo: o valor de t-1 pode "
            "tambem refletir o desempenho que levou ao corte."
        ),
        (
            "Em WR nenhuma covariavel do modelo final tem valor ausente no painel, "
            "entao a imputacao multipla nao tem o que imputar: as m = 10 copias sao "
            "identicas ao ajuste unico e o criterio se mantem por construcao, nao por "
            "evidencia. A comparacao exploratoria e testavel apenas em QB e RB e permanece "
            "condicional a selecao original; nao e prova formal White–Royston."
        ),
        (
            "O cenario C muda apenas o codigo sentinela de draft_round (8 -> 9); "
            "undrafted_flag mantem os valores do cenario A para isolar uma unica "
            "mudanca de codificacao."
        ),
        (
            "As covariaveis imputadas sao as covariaveis finais com algum valor "
            "ausente no painel (nao apenas as que casam com as palavras-chave de "
            "taxa), porque sao as unicas que a imputacao multipla pode afetar."
        ),
        (
            "O modelo de imputacao (IterativeImputer) inclui o desfecho: o indicador de "
            "evento do intervalo e o risco acumulado de Nelson-Aalen em stop (entrada tardia), "
            "como preditores extras que nao entram no Cox, em inspiracao a White & Royston (2009), "
            "nao como implementacao formal validada para este painel temporal. O conjunto final "
            "de covariaveis fica fixo na selecao original e as flags _missing nao entram na MI; "
            "as comparacoes sao exploratorias e condicionais. A comparacao com a imputacao "
            "deterministica por mediana tambem troca o modelo de imputacao."
        ),
    ]

    resumo = {
        "gerado_por": "pipeline/51_sensibilidade_inferencial.py",
        "metodo": {
            "teste_aceitacao": {
                "regra": (
                    "a selecao substantiva reconstruida com o limiar 50 tem de ser igual "
                    "ao conteudo de Inferência/Modelagem de COX/resultados/final_covs_{POS}.json "
                    "ignorando as entradas terminadas em _missing"
                ),
                "resultado": {p: aceitacao[p]["reproduz_selecao_do_pipeline"] for p in POSICOES},
                "aprovado": bool(all(aceitacao[p]["reproduz_selecao_do_pipeline"] for p in POSICOES)),
            },
            "limiar_taxa": {
                "descricao": (
                    "o corte do denominador das taxas de carreira (den_lag > limiar) e "
                    "varrido de 25 a 100; para cada limiar o painel Cox e reconstruido do "
                    "zero e a cadeia de selecao dos passos 11/12/13 e repetida integralmente"
                ),
                "limiar_padrao": LIMIAR_PADRAO,
                "limiares_testados": list(LIMIARES_TAXA),
                "similaridade": (
                    "Jaccard entre os conjuntos de covariaveis substantivas selecionadas "
                    "(sem as flags _missing), contra a selecao do limiar 50"
                ),
                "cadeia_reproduzida": (
                    "load_clean_position_data -> codificacao de draft -> filtro de inclusao "
                    "minima -> add_gap_aware_survival_structure -> lags de carreira "
                    "(clip em 0) -> taxas com den_lag > limiar -> flags _missing -> "
                    "fit_imputation_stats/apply_imputation_stats -> StandardScaler -> "
                    "filter_collinear_covariates(0.85) -> force_missing_flags -> "
                    "covariaveis_identificaveis_cox -> backward_cox_tdv(sem penalizacao, p < 0.05)"
                ),
                "fonte_da_cadeia": (
                    "logica de painel dos passos 04/05/06 e selecao dos passos 11/12/13; "
                    "backward_cox_tdv e executada a partir do codigo do passo 11, sem reimplementacao"
                ),
            },
            "undrafted": {
                "descricao": (
                    "quatro codificacoes alternativas para nao draftados, comparadas por "
                    "Jaccard contra o cenario A (o do pipeline), todas no limiar de taxa 50"
                ),
                "cenarios": DESCRICAO_CENARIOS,
                "referencia": "A",
            },
            "imputacao_multipla": {
                "descricao": (
                    "analise exploratoria condicional: modelo final de cada posicao, com "
                    "covariaveis de final_covs_{POS}.json fixadas pela selecao original e sem "
                    "flags _missing, reajustado sobre m = 10 paineis imputados; nao e teste "
                    "formal de White–Royston nem prova de robustez inferencial"
                ),
                "m": M_IMPUTACOES,
                "imputador": (
                    "sklearn.impute.IterativeImputer(sample_posterior=True, "
                    "initial_strategy='median', max_iter=10) com random_state = 42 + m; "
                    "preditores extras so da imputacao: aux_evento (evento do intervalo) e "
                    "aux_nelson_aalen_stop (Nelson-Aalen marginal com entrada tardia, em stop; n(s) = linhas com start < s <= stop); inspiracao em White–Royston, sem alegar implementacao formal validada para este painel"
                ),
                "linhas": (
                    "todas as linhas do painel usado no ajuste final do pipeline "
                    "(mesmas linhas de df_fit)"
                ),
                "colunas_imputadas": (
                    "covariaveis finais com ao menos um valor ausente no painel: "
                    + "; ".join(
                        f"{p}: {', '.join(analises[p]['imputacao']['colunas_imputadas']) or 'nenhuma'}"
                        for p in POSICOES
                    )
                ),
                "escala": (
                    "StandardScaler restrito as covariaveis finais, equivalente ao "
                    "escalonamento coluna a coluna do passo 11"
                ),
                "ajuste": "CoxTimeVaryingFitter sem penalizacao, por imputacao",
                "ajuste_unico": (
                    "HR e p de ajuste unico vem de cox_hr_{POS}.csv (passos 11/12/13) "
                    "quando o arquivo existe; o ajuste unico reconstruido aqui e conferido "
                    "contra ele em validacao_ajuste_unico_vs_cox_hr"
                ),
                "pooling": (
                    "regras de Rubin: Qbar = media dos coeficientes; Ubar = media das "
                    "variancias intra; B = variancia entre as m estimativas; "
                    "T = Ubar + (1 + 1/m) B; p pela t de Student com graus de liberdade "
                    "classicos de Rubin, nu = (m - 1)(1 + Ubar/((1 + 1/m)B))^2 (normal quando B = 0)"
                ),
                "alfa": ALFA,
                "veredito_inalterado": (
                    "indicador descritivo e condicional: compara p < 0,05 no ajuste unico "
                    "like-for-like (imputacao por mediana, sem flags _missing) com p < 0,05 "
                    "agrupado por Rubin, mantendo fixo o conjunto selecionado originalmente; "
                    "nao demonstra robustez formal nem compatibilidade White–Royston"
                ),
                "separacao_dos_mecanismos": (
                    "inflacao_se_pct mede a variacao do erro-padrao agrupado contra o do ajuste "
                    "unico; p_inflacao_apenas reavalia o veredito mantendo o coeficiente do "
                    "ajuste unico e trocando apenas o erro-padrao pelo agrupado; esta decomposicao "
                    "descritiva e condicional a selecao original e as flags _missing omitidas; "
                    "n_imputacoes_p_menor_alfa conta em quantas das m imputacoes a covariavel "
                    "tem p < 0,05, sem constituir criterio formal de robustez"
                ),
                "diagnostico_das_imputacoes": (
                    "media, desvio, minimo e maximo dos valores imputados (empilhados nas m "
                    "imputacoes) contra os observados, e percentual de valores imputados fora "
                    "do intervalo observado"
                ),
            },
        },
        "destaques": {
            "limiar_50_reproduz_selecao_do_pipeline": bool(
                all(aceitacao[p]["reproduz_selecao_do_pipeline"] for p in POSICOES)
            ),
            "limiares_adjacentes_estaveis": adjacentes_estaveis,
            "undrafted_nao_altera_selecao_final": {
                p: bool(all(jaccard_cenarios[p][c]["jaccard_vs_A"] == 1.0 for c in ("B", "C", "D")))
                for p in POSICOES
            },
            "imputacao_multipla_altera_algum_veredito_alfa_005": bool(algum_verdito_alterado),
            # So as covariaveis efetivamente imputadas entram nesta conta: nas
            # demais, a variacao do erro-padrao e ruido de Monte Carlo (0,2% a 2%).
            "inflacao_conservadora_do_erro_padrao_sob_imputacao_multipla": {
                p: bool(any(
                    linha["se_pooled"] > linha["se_unica"]
                    for linha in analises[p]["imputacao"]["linhas"]
                    if linha["covariavel"] in (mi_resumo[p]["colunas_imputadas"] or [])
                ))
                for p in POSICOES
            },
            "veredito_mudaria_apenas_pela_inflacao_do_erro_padrao": {
                p: mi_resumo[p]["veredito_mudaria_apenas_pela_inflacao_do_erro_padrao"]
                for p in POSICOES
            },
        },
        "aceitacao_limiar_50": aceitacao,
        "afirmacoes_da_escrita": afirmacoes,
        "limiares_taxa": {
            "jaccard_vs_50": jaccard_limiares,
            "adjacentes": adjacentes_estaveis,
            "selecoes": {
                p: {str(limiar): analises[p]["limiares"][limiar]["substantivas"] for limiar in LIMIARES_TAXA}
                for p in POSICOES
            },
        },
        "undrafted": jaccard_cenarios,
        "imputacao_multipla": mi_resumo,
        "validacao_ajuste_unico_vs_cox_hr": {p: analises[p]["validacao"] for p in POSICOES},
        "sensibilidade_cox_final": {
            p: {
                "n_linhas": sens_cox[p]["n_linhas"],
                "n_eventos": sens_cox[p]["n_eventos"],
                "eventos_empatados": sens_cox[p]["eventos_empatados"],
                "pct_eventos_empatados": sens_cox[p]["pct_eventos_empatados"],
                "validacao_efron_r_vs_lifelines_max_abs_diff": sens_cox[p][
                    "validacao_efron_r_vs_lifelines_max_abs_diff"],
                "ep_robusto_veredito_muda": sens_cox[p]["ep_robusto"].loc[
                    sens_cox[p]["ep_robusto"]["veredito_muda"], "covariavel"].tolist(),
                "breslow_veredito_muda": sens_cox[p]["empates"].loc[
                    sens_cox[p]["empates"]["veredito_muda"], "covariavel"].tolist(),
            }
            for p in POSICOES
        },
        "sensibilidade_tempo_discreto": {
            p: {
                "modelo": "GLM binomial cloglog sobre intervalos pessoa-periodo, com efeito fixo por grupo de tempo",
                "modelo_principal": sens_cox[p]["tempo_discreto"]["modelo_principal"],
                "n_niveis_tempo_principal": sens_cox[p]["tempo_discreto"]["n_niveis_tempo_principal"],
                "n_linhas_pessoa_periodo": sens_cox[p]["tempo_discreto"]["n_linhas_pessoa_periodo"],
                "grupos_de_tempo_agrupados": sens_cox[p]["tempo_discreto"]["grupos_tempo"],
                "min_eventos_por_grupo_agrupado": MIN_EVENTOS_POR_TEMPO,
                "covariaveis_mudam_de_lado_0_05_efron_cloglog_por_ano": sens_cox[p]["tempo_discreto"]["covariaveis_mudam_de_lado"],
                "n_mudam_de_lado_0_05_efron_cloglog_por_ano": len(sens_cox[p]["tempo_discreto"]["covariaveis_mudam_de_lado"]),
                "covariaveis_mudam_de_lado_0_05_efron_cloglog_agrupado": sens_cox[p]["tempo_discreto"]["covariaveis_mudam_de_lado_agrupado"],
                "n_mudam_de_lado_0_05_efron_cloglog_agrupado": len(sens_cox[p]["tempo_discreto"]["covariaveis_mudam_de_lado_agrupado"]),
            }
            for p in POSICOES
        },
        "covariaveis_defasadas": {
            "gerado_por": "sens_covariaveis_defasadas_{POS}.csv",
            "metodo": (
                "Cox final (mesmos finais do passo 11, lifelines, Efron, sem penalizacao) "
                "em duas versoes na amostra comum (linhas com linha do mesmo jogador em "
                "season - 1): (i) covariaveis de temporada corrente medidas em t; (ii) as "
                "mesmas trocadas pelo valor de t-1. Fixas/defasadas (draft, altura, "
                "unobs_career_years, observed_career_lag_*, flags _missing, hiato) ficam iguais. "
                "A escala da versao (ii) e a da corrente. HR oficial: cox_hr_{POS}.csv."
            ),
            "por_posicao": {
                p: {
                    "correntes": analises[p]["defasadas"]["correntes"],
                    "fixas_defasadas": analises[p]["defasadas"]["fixas_defasadas"],
                    "n_linhas_amostra_completa": analises[p]["defasadas"]["n_linhas_completa"],
                    "n_eventos_amostra_completa": analises[p]["defasadas"]["n_eventos_completa"],
                    "n_linhas_amostra_comum": analises[p]["defasadas"]["n_linhas_amostra_comum"],
                    "n_eventos_amostra_comum": analises[p]["defasadas"]["n_eventos_amostra_comum"],
                    "n_linhas_sem_t1": analises[p]["defasadas"]["n_linhas_sem_t1"],
                    "validacao_completa_vs_cox_hr_max_abs_diff": analises[p]["defasadas"][
                        "validacao_completa_vs_cox_hr_max_abs_diff"],
                    "covariaveis_mudam_veredito_0_05": analises[p]["defasadas"]["covariaveis_mudam_veredito"],
                }
                for p in POSICOES
            },
        },
        "sensibilidade_passer_rating_zero_QB": {
            "linhas_sem_tentativas": sens_passer["linhas_sem_tentativas"],
            "validacao_cenario_a_vs_cox_hr_max_abs_diff": sens_passer["validacao_cenario_a_vs_cox_hr_max_abs_diff"],
            "tabela": sens_passer["tabela"].to_dict(orient="records"),
        },
        "detalhe_por_posicao": {
            p: {
                "n_candidatas_apos_colinearidade": analises[p]["n_candidatas"],
                "limiar_p_usado": analises[p]["limiar_p"],
                "max_vars": analises[p]["max_vars"],
                "tempos_s": analises[p]["tempos"],
            }
            for p in POSICOES
        },
        "tempo_s": {
            "total": tempo_total,
            "por_posicao": {p: analises[p]["tempos"]["total_s"] for p in POSICOES},
        },
        "limitacoes": limitacoes,
    }
    (SAIDA / "resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------- relatorio
    print("\n=== Teste de aceitacao (limiar 50 reproduz a selecao do pipeline) ===")
    for posicao in POSICOES:
        estado = "OK" if aceitacao[posicao]["reproduz_selecao_do_pipeline"] else "FALHOU"
        print(f"  {posicao}: {estado} | selecao {aceitacao[posicao]['selecao_reconstruida_limiar_50']}")
        if estado == "FALHOU":
            print(f"      final_covs: {aceitacao[posicao]['final_covs_json_sem_missing']}")
            print(f"      apenas no json: {aceitacao[posicao]['apenas_no_json']}")
            print(f"      apenas na reconstrucao: {aceitacao[posicao]['apenas_na_reconstrucao']}")

    print("\n=== Jaccard dos limiares contra o limiar 50 ===")
    for posicao in POSICOES:
        valores = ", ".join(
            f"{limiar}:{jaccard_limiares[posicao][str(limiar)]:.2f}" for limiar in LIMIARES_TAXA
        )
        adjacente = adjacentes_estaveis[posicao]
        print(f"  {posicao}: {valores}")
        print(f"      adjacentes: jaccard minimo "
              f"{adjacente['jaccard_minimo_entre_limiares_adjacentes']:.2f} | "
              f"todos identicos = {adjacente['todos_os_pares_adjacentes_identicos']}")

    print("\n=== Cenarios de codificacao de nao draftados (Jaccard contra A) ===")
    for posicao in POSICOES:
        valores = ", ".join(
            f"{c}:{jaccard_cenarios[posicao][c]['jaccard_vs_A']:.2f}" for c in ("A", "B", "C", "D")
        )
        print(f"  {posicao}: {valores}")

    print("\n=== Imputacao multipla exploratoria e condicional (m = 10) ===")
    for posicao in POSICOES:
        dados = mi_resumo[posicao]
        print(f"  {posicao}: imputadas = {dados['colunas_imputadas'] or 'nenhuma'} | "
              f"verditos alterados = {dados['n_verditos_alterados']}")
        for linha in dados["covariaveis"]:
            marca = "=" if linha["significancia_inalterada"] else "MUDOU"
            print(f"      {linha['covariavel']:<45} HR {linha['hr_unica_sem_flag']:.3f} -> "
                  f"{linha['hr_pooled']:.3f} | p {linha['p_unica_sem_flag']:.4f} -> "
                  f"{linha['p_pooled']:.4f} [{marca}] | se {linha['inflacao_se_pct']:+.1f}% | "
              f"p<0,05 em {linha['n_imputacoes_p_menor_alfa']}/{dados['n_imputacoes']} imputacoes "
              "(contagem descritiva, condicional ao modelo selecionado)")

    print("\n=== Afirmacoes e leitura exploratoria ===")
    rotulos = {
        "1_limiar_das_taxas_entre_25_e_100": "1. limiares de taxa adjacentes estaveis",
        "2_codificacao_de_nao_draftados": "2. codificacoes alternativas de nao draftados",
        "3_imputacao_multipla_e_alfa_005": "3. MI exploratoria condicional (nao robustez formal White–Royston)",
    }
    for chave, rotulo in rotulos.items():
        bloco = afirmacoes[chave]
        if chave == "3_imputacao_multipla_e_alfa_005":
            estado = "LEITURA EXPLORATORIA CONDICIONAL"
        else:
            estado = "SUSTENTADA" if bloco["sustentada_em_todas_as_posicoes"] else "NAO SUSTENTADA em todas"
        print(f"  {rotulo}: {estado}")
        for posicao in POSICOES:
            info = bloco["por_posicao"][posicao]
            if chave == "1_limiar_das_taxas_entre_25_e_100":
                print(f"      {posicao}: adjacentes identicos = "
                      f"{info['todos_os_pares_adjacentes_identicos']} | "
                      f"limiares diferentes do 50 = {info['limiares_com_selecao_diferente_do_padrao_50']}")
            elif chave == "2_codificacao_de_nao_draftados":
                print(f"      {posicao}: Jaccard {info['jaccard_vs_A']} | "
                      f"diferentes de A = {info['cenarios_com_selecao_diferente_de_A']}")
            else:
                print(f"      {posicao}: testavel = {info['testavel']} | imputadas = "
                      f"{info['colunas_imputadas']} | verditos alterados = {info['verditos_alterados']} | "
                      f"veredito mudaria so pela inflacao do se = "
                      f"{info['veredito_mudaria_apenas_pela_inflacao_do_erro_padrao']}")

    print("\n=== Validacao do ajuste unico contra cox_hr_{POS}.csv ===")
    for posicao in POSICOES:
        validacao = analises[posicao]["validacao"]
        if validacao.get("disponivel"):
            print(f"  {posicao}: max |dif coef| = {validacao['max_abs_diff_coef']:.2e} | "
                  f"identico = {validacao['identico_tolerancia_1e-8']}")
        else:
            print(f"  {posicao}: cox_hr_{posicao}.csv indisponivel")

    print(f"\nArtefatos em {SAIDA.relative_to(ROOT)}:")
    print(f"  limiares_taxa.csv ({len(tabela_limiares)} linhas)")
    print(f"  undrafted_cenarios.csv ({len(tabela_cenarios)} linhas)")
    print(f"  imputacao_multipla.csv ({len(tabela_mi)} linhas)")
    print("  sens_ep_robusto_{QB,RB,WR}.csv, sens_empates_{QB,RB,WR}.csv, sens_tempo_discreto_{QB,RB,WR}.csv")
    print("  sens_passer_rating_zero_QB.csv")
    print("  sens_covariaveis_defasadas_{QB,RB,WR}.csv")
    print("  resumo.json")
    print(f"Tempo total: {tempo_total:.1f}s")

    if not resumo["metodo"]["teste_aceitacao"]["aprovado"]:
        raise AssertionError(
            "Teste de aceitacao falhou: a reconstrucao com limiar 50 nao reproduz "
            "final_covs_{POS}.json para " + ", ".join(
                p for p in POSICOES if not aceitacao[p]["reproduz_selecao_do_pipeline"]
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
