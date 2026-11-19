"""Protocolo preditivo oficial: um X, walk-forward, landmark van Houwelingen, métricas."""
from __future__ import annotations

import json
import sys
import warnings
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parent.parent.parent
INFER_SHARED = ROOT / "Inferência" / "shared"
sys.path.insert(0, str(INFER_SHARED))
from survival_utils import (  # noqa: E402
    apply_imputation_stats,
    apply_scaler,
    build_residual_snapshot_panel,
    fit_imputation_stats,
    fit_scaler,
    imputa_years_exp,
)

PRED_SHARED = ROOT / "Predição" / "shared"
HORIZONS = np.array([1.0, 2.0, 3.0, 5.0], dtype=float)
TEST_SEASONS = list(range(2018, 2024))
LANDMARKS = list(range(2020, 2024))
LANDMARK_WS = (1, 2, 3)
# Ultima temporada da base. A saida ao fim da temporada s so se confirma pela ausencia em s + 1.
FIM_BASE = 2024


def landmark_completo(t_l: int, w: int, fim: int = FIM_BASE) -> bool:
    """A origem t_L tem acompanhamento completo para o horizonte w.

    "Sair em ate w anos" e ter a ultima temporada ate t_L + w, e a ultima temporada so se
    confirma pela ausencia na seguinte. A origem entra nas medias e nos quintis do
    landmarking so se a base cobre t_L + w + 1. Na origem t_L = fim - w, quem jogou a
    temporada fim fica com o rotulo de "nao saiu" sem que isso possa ser verificado.
    """
    return int(t_l) + int(w) + 1 <= int(fim)
META_COLS = [
    "age",
    "games_missed",
    "offense_pct",
    "depth_team",
    "years_exp",
    "unobs_career_years",
    "pre_window_veteran",
    "unobserved_history_flag",
    "start",
    "player_name",
]
# start := years_exp na estrutura counting process; por isso fica fora do Aalen.
AALEN_FORBIDDEN_COVS = ("years_exp", "first_observed_season")

# Ranking oficial: média dos folds 2018 a 2021. 2022-2023 existem e não entram na média.
OFFICIAL_SEASONS = (2018, 2019, 2020, 2021)

# Hiperparametros do protocolo principal: escolhidos pela busca aninhada em CADA fold
# (estudo_tune_preditivo/resultados/hps_por_fold.csv). Cada fold usa so a sua escolha,
# feita com o treino do proprio fold (validacao interna = 2 ultimos anos do treino).
# Nao ha moda, congelamento nem fallback silencioso: chave ausente levanta erro.
HPS_ANINHADOS_CSV = (
    ROOT / "documentos" / "estudo_tune_preditivo" / "resultados" / "hps_por_fold.csv"
)

# Valores padrao (scikit-survival e lifelines). Usados APENAS na sensibilidade
# `--hp default` do pacote oficial, para comparacao com o protocolo principal.
DEFAULT_COX_PENALIZER = 0.0
DEFAULT_GBSA_HP = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 1.0,
}
DEFAULT_RSF_HP = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 6,
    "min_samples_leaf": 3,
}

# Landmark: cada origem t_L (2020 a 2023) reusa o HP do fold walk-forward de mesma origem
# (mesmo corte de informacao; nao ha busca por horizonte w). Decisao registrada.
LANDMARK_HP_ORIGEM = "fold walk-forward de mesma temporada t_L"

# RSF_MIN_EVENTOS registra a ordem de grandeza de eventos do QB (108) como diagnóstico.
# O RSF roda em todas as posicoes; a regra (rsf_permitido) e constante e a contagem
# de eventos nao decide nada.
RSF_MIN_EVENTOS = 200
# Tempo minimo usado no lugar de duracao nula (evento na propria temporada do retrato).
MODEL_EPSILON_PROTOCOLO = 1e-3


def common_training_horizon(
    train_followup_by_position: Mapping[str, object],
    tau_requested: float,
    required_positions: tuple[str, ...] = ("QB", "RB", "WR"),
) -> tuple[float, dict[str, float]]:
    requested = float(tau_requested)
    if not np.isfinite(requested) or requested <= 0:
        raise ValueError("tau solicitado deve ser finito e positivo")
    if not isinstance(train_followup_by_position, Mapping) or not train_followup_by_position:
        raise ValueError("são necessários seguimentos de treino para todas as posições")
    expected_positions = set(required_positions)
    provided_positions = {str(position) for position in train_followup_by_position}
    if not expected_positions or provided_positions != expected_positions:
        raise ValueError(
            f"posições de treino incompletas: esperado={sorted(expected_positions)}, "
            f"recebido={sorted(provided_positions)}"
        )
    max_train_by_position = {}
    for position in required_positions:
        followup = train_followup_by_position[position]
        values = np.asarray(pd.to_numeric(pd.Series(followup), errors="coerce"), dtype=float).reshape(-1)
        if not values.size or not np.isfinite(values).all():
            raise ValueError(f"{position}: seguimentos de treino vazios ou não finitos")
        max_train = float(values.max())
        if max_train <= 0:
            raise ValueError(f"{position}: suporte máximo de treino deve ser positivo")
        max_train_by_position[str(position)] = max_train
    return min(requested, min(max_train_by_position.values())), max_train_by_position


def restricted_time_identifiable(event, followup, tau) -> np.ndarray:
    event_values, followup_values, horizons = np.broadcast_arrays(
        np.asarray(event, dtype=float),
        np.asarray(followup, dtype=float),
        np.asarray(tau, dtype=float),
    )
    valid = np.isfinite(event_values) & np.isfinite(followup_values) & np.isfinite(horizons)
    return valid & ((event_values == 1) | (followup_values >= horizons))


def restricted_time_target(followup, tau, identifiable) -> np.ndarray:
    followup_values, horizons, mask = np.broadcast_arrays(
        np.asarray(followup, dtype=float),
        np.asarray(tau, dtype=float),
        np.asarray(identifiable, dtype=bool),
    )
    target = np.minimum(followup_values, horizons)
    return np.where(mask & np.isfinite(target), target, np.nan)
RSF_POSICOES_COM_RSF = ("QB", "RB", "WR")


def rsf_permitido(pos: str, n_eventos: int | None = None) -> bool:
    """RSF entra no pacote em todas as posições de RSF_POSICOES_COM_RSF (QB, RB, WR).

    `n_eventos` é aceito para diagnóstico e não altera a decisão. O HP do RSF vem da
    busca aninhada do próprio fold, e a instabilidade é tratada pela escolha, não pela
    exclusão.
    """
    del n_eventos
    return pos in RSF_POSICOES_COM_RSF


def carrega_hps_aninhados(caminho: Path = HPS_ANINHADOS_CSV) -> pd.DataFrame:
    """Tabela (posicao, modelo, teste_em_temporada) -> hp_escolhido, gravada pelo tune."""
    df = pd.read_csv(caminho)
    if "inner_fallback" in df.columns and df["inner_fallback"].notna().any():
        raise ValueError(f"{caminho}: há folds com fallback para o padrão; a busca precisa ser refeita")
    chave = ["posicao", "modelo", "teste_em_temporada"]
    if df.duplicated(chave).any():
        raise ValueError(f"{caminho}: chaves duplicadas (posicao, modelo, temporada)")
    return df


def hp_aninhado(hps: pd.DataFrame, pos: str, modelo: str, origem: int) -> dict:
    """HP escolhido pela busca aninhada no fold de origem `origem` (erro explícito se faltar)."""
    sel = hps[(hps["posicao"] == pos) & (hps["modelo"] == modelo) & (hps["teste_em_temporada"] == int(origem))]
    if len(sel) != 1:
        raise KeyError(f"HP aninhado ausente para ({pos}, {modelo}, {origem}): {len(sel)} linhas")
    return json.loads(sel["hp_escolhido"].iloc[0])


def hp_padrao(modelo: str) -> dict:
    """Valor padrão para a sensibilidade `--hp default`."""
    if modelo == "cox":
        return {"penalizer": float(DEFAULT_COX_PENALIZER)}
    if modelo == "gbsa":
        return dict(DEFAULT_GBSA_HP)
    if modelo == "rsf":
        return dict(DEFAULT_RSF_HP)
    raise ValueError(modelo)


def cox_penalizer(pos: str) -> float:
    """Penalizacao padrao do Cox (sensibilidade `--hp default`). O protocolo principal usa
    o penalizer escolhido pela busca aninhada em cada fold (hp_aninhado)."""
    del pos
    return float(DEFAULT_COX_PENALIZER)


def load_covs(pos: str) -> list[str]:
    payload = json.loads((PRED_SHARED / f"covs_preditivas_{pos}.json").read_text())
    return list(payload["covariates"])


def load_panel(pos: str) -> tuple[pd.DataFrame, list[str]]:
    covs = load_covs(pos)
    df = pd.read_pickle(PRED_SHARED / f"df_cox_{pos}_features.pkl")
    missing = [c for c in covs if c not in df.columns]
    if missing:
        raise ValueError(f"{pos}: covariáveis ausentes no painel: {missing}")
    panel = build_residual_snapshot_panel(df, covs, META_COLS)
    return panel, covs


def aalen_covs(covs: list[str]) -> list[str]:
    """Aalen usa Surv(start, stop) com start=years_exp; não recolocá-lo em X."""
    return [c for c in covs if c not in AALEN_FORBIDDEN_COVS]


def usable_covs(train_proc: pd.DataFrame, covs: list[str]) -> list[str]:
    keep = []
    for c in covs:
        s = pd.to_numeric(train_proc[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        if float(s.var(skipna=True) or 0) <= 1e-12:
            continue
        keep.append(c)
    return keep


def preprocess(train_raw: pd.DataFrame, test_raw: pd.DataFrame, covs: list[str]):
    stats = fit_imputation_stats(train_raw, covs)
    tr = apply_imputation_stats(train_raw, covs, stats)
    te = apply_imputation_stats(test_raw, covs, stats)
    use = usable_covs(tr, covs)
    scaler = fit_scaler(tr, use)
    tr = apply_scaler(tr, scaler, use)
    te = apply_scaler(te, scaler, use)
    return tr, te, use, stats, scaler


def mask_test_support(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Desligada: devolve o teste inteiro.

    A regra descartaria do teste as linhas com tempo maior ou igual ao maximo do treino.
    Com os rotulos de treino em tempo real (abaixo), ela apagaria justamente os censurados
    de seguimento mais longo, isto e, filtraria o teste pelo desfecho; por isso fica
    desligada, e os chamadores registram zero remocoes.
    """
    del train
    return test.copy()


@lru_cache(maxsize=1)
def _liga_anos_reg() -> pd.DataFrame:
    """Presenca na liga (qualquer posicao) na temporada regular, com years_exp imputado.

    Mesma base de `load_clean_position_data` (Inferencia/shared): a inferencia define a saida
    pela ultima temporada do jogador na liga, nao na posicao.
    """
    raw = pd.read_csv(ROOT / "DataBase" / "yearly_player_stats_offense.csv")
    reg = imputa_years_exp(raw[raw["season_type"] == "REG"])
    return reg[["player_id", "season", "years_exp"]].drop_duplicates(["player_id", "season"])


def rotulo_em_tempo_real(treino: pd.DataFrame, panel: pd.DataFrame, origem: int) -> pd.DataFrame:
    """Tempo restante e evento das linhas de treino com o que se sabia ao fim de `origem`.

    Mesma regra do painel residual (`build_residual_snapshot_panel`), avaliada na origem:
    - evento: a ultima temporada da POSICAO (P) coincide com a ultima temporada na LIGA (L)
      e L e anterior a `origem`. Quem muda de posicao e segue na liga nao tem evento;
    - duracao: stop(P) - snapshot_stop, ou seja, censura ao deixar a posicao (como no painel);
    - duracao do modelo: evento com duracao <= 0 recebe MODEL_EPSILON_PROTOCOLO.
    O conjunto de teste NAO passa por aqui: e avaliado com o desfecho real.
    """
    conh = panel[panel["snapshot_season"] <= origem]
    ultimo_pos = (
        conh.sort_values("snapshot_season")
        .groupby("player_id", observed=True)
        .tail(1)
        .set_index("player_id")[["snapshot_season", "snapshot_stop"]]
        .rename(columns={"snapshot_season": "_P", "snapshot_stop": "_stop_P"})
    )
    liga = _liga_anos_reg()
    ultimo_liga = (
        liga[liga["season"] <= origem]
        .sort_values("season")
        .groupby("player_id", observed=True)
        .tail(1)
        .set_index("player_id")[["season"]]
        .rename(columns={"season": "_L"})
    )
    out = treino.join(ultimo_pos, on="player_id").join(ultimo_liga, on="player_id")
    evento = ((out["_P"] == out["_L"]) & (out["_L"] < origem)).astype(int)
    dur = (out["_stop_P"] - out["snapshot_stop"]).astype(float)
    out["residual_duration"] = dur
    out["residual_event"] = evento
    out["residual_duration_model"] = dur.where(~((evento == 1) & (dur <= 0)), MODEL_EPSILON_PROTOCOLO)
    return out.drop(columns=["_P", "_stop_P", "_L"])


def evento_em_tempo_real_cp(cp: pd.DataFrame, origem: int) -> pd.DataFrame:
    """Linhas de treino (season < origem) do painel counting process, com o evento que se sabia ao fim de `origem`.

    Mesma regra de `rotulo_em_tempo_real`, na liga: o evento e a ultima temporada da posicao
    coincidindo com a ultima temporada na liga (qualquer posicao), e essa temporada precisa
    ser anterior a `origem`. Quem mudou de posicao e segue na liga nao tem evento no treino.
    `cp` precisa conter a origem.
    """
    ultima = cp[cp["season"] <= origem].groupby("player_id", observed=True)["season"].max()
    ultima_liga = (
        _liga_anos_reg()[lambda d: d["season"] <= origem].groupby("player_id", observed=True)["season"].max()
    )
    out = cp[cp["season"] < origem].copy()
    ult = out["player_id"].map(ultima)
    ul = out["player_id"].map(ultima_liga)
    out["event"] = ((out["season"] == ult) & (ult == ul) & (ul < origem)).astype(int)
    return out


def media_restrita_sksurv(
    survival_functions, tau: float, exige_dominio: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Integra StepFunctions nos nós nativos; em modo não estrito avisa o tau efetivo."""
    tau = float(tau)
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau deve ser finito e não negativo")
    functions = list(survival_functions)
    if not functions:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    supports = []
    curves = []
    for fn in functions:
        x = np.asarray(fn.x, dtype=float).reshape(-1)
        y = np.asarray(fn.y, dtype=float).reshape(-1)
        if len(x) == 0 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("curva de sobrevivência sem nós/valores finitos alinhados")
        if np.any(x < 0) or np.any(np.diff(x) <= 0):
            raise ValueError("nós da StepFunction devem ser não negativos e estritamente crescentes")
        try:
            scale = float(fn.a)
            offset = float(fn.b)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("StepFunction deve expor escala/offset finitos a e b") from exc
        if not np.isfinite(scale) or not np.isfinite(offset):
            raise ValueError("escala/offset a e b da StepFunction devem ser finitos")
        y = scale * y + offset
        tolerance = 1e-12
        if np.any(y < -tolerance) or np.any(y > 1.0 + tolerance):
            raise ValueError("sobrevivência efetiva deve permanecer no intervalo [0, 1]")
        if np.any(np.diff(y) > tolerance):
            raise ValueError("sobrevivência efetiva não pode aumentar nos nós")
        y = np.clip(y, 0.0, 1.0)
        y = np.minimum.accumulate(y)
        supports.append(float(x[-1]))
        curves.append((x, y))
    min_support = min(supports)
    if exige_dominio and tau > min_support:
        raise ValueError(f"tau={tau} passa do suporte das curvas (suporte comum máximo={min_support})")
    if not exige_dominio and tau > min_support:
        effective_tau = min(tau, min_support)
        warnings.warn(
            "RMST truncada por suporte: tau solicitado="
            f"{tau}; tau efetivo comum={effective_tau}",
            RuntimeWarning,
            stacklevel=2,
        )
    means = []
    medians = []
    for x, y in curves:
        upper = tau if exige_dominio else min(tau, min_support)
        knots = np.unique(np.r_[0.0, x[(x > 0) & (x < upper)], upper])
        starts = knots[:-1]
        indexes = np.searchsorted(x, starts, side="right") - 1
        step_values = np.ones(len(starts), dtype=float)
        on_or_after_first = indexes >= 0
        step_values[on_or_after_first] = y[indexes[on_or_after_first]]
        means.append(float(np.dot(step_values, np.diff(knots))))
        crossings = np.flatnonzero((x <= upper) & (y <= 0.5))
        medians.append(float(x[crossings[0]]) if len(crossings) else float(upper))
    return np.asarray(means, dtype=float), np.asarray(medians, dtype=float)


def media_restrita_cox(cph, X: pd.DataFrame, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Media restrita e mediana do Cox integrando a escada de Breslow exatamente.

    `predict_survival_function` do lifelines interpola linearmente a hazard de base entre
    os tempos de evento; com tempos inteiros isso subestima a media restrita em ate
    0,2 ano. Aqui S(t) e constante entre tempos de evento, como no estimador de Breslow.
    """
    h0 = cph.baseline_cumulative_hazard_.iloc[:, 0]
    tempos = h0.index.to_numpy(dtype=float)
    dentro = tempos < tau
    pontos = np.r_[0.0, tempos[dentro], tau]
    h_degrau = np.r_[0.0, h0.to_numpy(dtype=float)[dentro]]  # hazard em cada degrau [pontos[k], pontos[k+1])
    ph = cph.predict_partial_hazard(X).to_numpy(dtype=float).reshape(-1, 1)
    s = np.exp(-ph * h_degrau.reshape(1, -1))
    media = (s * np.diff(pontos).reshape(1, -1)).sum(axis=1)
    mediana = []
    for linha in s:
        abaixo = np.where(linha <= 0.5)[0]
        mediana.append(float(pontos[abaixo[0]]) if len(abaixo) else float(tau))
    return media, np.array(mediana, dtype=float)


def walkforward_split(panel: pd.DataFrame, season: int):
    """Treino: retratos anteriores a `season`, rotulados so com o que se sabia em `season`."""
    tr = rotulo_em_tempo_real(panel[panel["snapshot_season"] < season].copy(), panel, season)
    te = panel[panel["snapshot_season"] == season].copy()
    return tr, te


def landmark_outcomes(df: pd.DataFrame, w: float) -> pd.DataFrame:
    out = df.copy()
    dur = pd.to_numeric(out["residual_duration_model"], errors="coerce")
    ev = pd.to_numeric(out["residual_event"], errors="coerce").fillna(0).astype(int)
    out["lm_time"] = np.minimum(dur, w)
    out["lm_event"] = ((ev == 1) & (dur <= w)).astype(int)
    out.loc[out["lm_time"] <= 0, "lm_time"] = 1e-6
    return out


def landmark_split(panel: pd.DataFrame, t_l: int, w: float):
    tr = landmark_outcomes(rotulo_em_tempo_real(panel[panel["snapshot_season"] < t_l].copy(), panel, t_l), w)
    te = landmark_outcomes(panel[panel["snapshot_season"] == t_l].copy(), w)
    return tr, te


def y_surv(df: pd.DataFrame, time_col: str, event_col: str):
    return Surv.from_arrays(
        event=df[event_col].astype(bool).to_numpy(),
        time=df[time_col].astype(float).to_numpy(),
    )


def candidate_times(train_time: np.ndarray, test_time: np.ndarray, horizons=HORIZONS) -> np.ndarray:
    upper = min(float(train_time.max()) - 1e-8, float(test_time.max()) - 1e-8)
    return np.array([t for t in horizons if t < upper], dtype=float)


def valid_horizon_mask(y_test, times: np.ndarray) -> np.ndarray:
    valid = []
    for t in times:
        cases = np.any(y_test["event"] & (y_test["time"] <= t))
        controls = np.any(y_test["time"] > t)
        valid.append(bool(cases and controls))
    return np.array(valid, dtype=bool)


def evaluate_fold(
    y_train,
    y_test,
    risk: np.ndarray,
    surv: np.ndarray | None,
    times: np.ndarray,
    compute_uno: bool = True,
) -> dict:
    row = {
        "c_index": np.nan,
        "uno_c_index": np.nan,
        "mean_time_auc": np.nan,
        "ibs": np.nan,
        "brier_1a": np.nan,
        "brier_2a": np.nan,
        "brier_3a": np.nan,
        "auc_1a": np.nan,
        "auc_2a": np.nan,
        "auc_3a": np.nan,
        "auc_5a": np.nan,
        "uno_degenerado": np.nan,
    }
    if len(y_test) < 5 or int(y_test["event"].sum()) < 1:
        return row
    # Distribuicao de censura para o IPCW: estimada no proprio conjunto avaliado. A
    # censura do teste combina o fim administrativo de 2024 e a saida da posicao (troca
    # de posicao, como no painel); a do treino, com rotulos em tempo
    # real, concentra-se na origem de cada fold e daria pesos errados ao teste
    # (Gerds e Schumacher, 2006: G vem da amostra em que o escore e calculado).
    y_cens = y_test
    try:
        row["c_index"] = float(concordance_index_censored(y_test["event"], y_test["time"], risk)[0])
    except Exception:
        pass
    if compute_uno:
        try:
            uno = float(concordance_index_ipcw(y_cens, y_test, risk)[0])
            harrell = row["c_index"]
            # IPCW 0/1 neste recorte faz Uno=Harrell; não reportar como métrica extra.
            if np.isfinite(uno) and np.isfinite(harrell) and abs(uno - harrell) < 1e-12:
                row["uno_c_index"] = np.nan
                row["uno_degenerado"] = 1
            else:
                row["uno_c_index"] = uno
                row["uno_degenerado"] = 0
        except Exception:
            pass
    if times is None or len(times) == 0 or surv is None or surv.size == 0:
        return row
    mask = valid_horizon_mask(y_test, times)
    tv, sv = times[mask], surv[:, mask] if surv.ndim == 2 else surv
    if len(tv) == 0:
        return row
    try:
        auc_values, mean_auc = cumulative_dynamic_auc(y_cens, y_test, risk, tv)
        row["mean_time_auc"] = float(mean_auc)
        for t, v in zip(tv, auc_values):
            row[f"auc_{int(t)}a"] = float(v)
    except Exception:
        pass
    try:
        _, brier_values = brier_score(y_cens, y_test, sv, tv)
        for t, v in zip(tv, brier_values):
            if int(t) in (1, 2, 3):
                row[f"brier_{int(t)}a"] = float(v)
        if len(tv) >= 2:
            row["ibs"] = float(integrated_brier_score(y_cens, y_test, sv, tv))
    except Exception:
        pass
    return row


def km_survival_matrix(train: pd.DataFrame, n_test: int, times: np.ndarray, time_col: str, event_col: str):
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter()
    kmf.fit(train[time_col], event_observed=train[event_col])
    if len(times) == 0:
        return np.empty((n_test, 0)), np.zeros(n_test)
    s = np.array([float(kmf.predict(t)) for t in times], dtype=float)
    surv = np.tile(s, (n_test, 1))
    risk = np.zeros(n_test)
    return surv, risk



def modelos_destaque(resumo_wf: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Por posicao: (modelo de maior C nos folds de referencia, modelo de contraste).

    O contraste e o Cox; quando o proprio Cox tem o maior C, e a arvore de maior C.
    Lido do artefato para que figuras e diagnosticos sigam o resultado, sem lista fixa.
    """
    out = {}
    d = resumo_wf[resumo_wf["modelo"] != "km"]
    for pos, g in d.groupby("posicao"):
        g = g.sort_values("c_index", ascending=False)
        campeao = str(g.iloc[0]["modelo"])
        if campeao == "cox":
            contraste = str(g[g["modelo"] != "cox"].iloc[0]["modelo"])
        else:
            contraste = "cox"
        out[str(pos)] = (campeao, contraste)
    return out
