#!/usr/bin/env python3
"""Leitura aplicada dos resultados preditivos (passo 62). Nao reajusta modelo nenhum.

Le artefatos existentes (predicoes out-of-fold do walk-forward 2018-2021, lista 2024,
painel inferencial, Cox e Aalen finais, importancia por permutacao, DataBase anual) e
grava em ``documentos/estudo_leitura_aplicada/resultados/``:

(a) casos_auge.csv                  casos no auge por posicao, regra em CASE_RULES
(b) tipologia_erro.csv              erro em anos por tipo de trajetoria (TIPOS)
(c) concordancia_modelos.csv        Spearman entre Cox, GBSA e RSF por posicao e fold
    concordancia_divergentes.csv    jogadores de maior divergencia entre modelos
(d) inferencia_vs_importancia.csv   covariavel do pool x Cox final x Aalen final x ranking
(e) dp_covariaveis.csv              media e DP cruas do painel inferencial por posicao, e
                                    dp_escalonador = DP usado pelo StandardScaler do Cox
                                    inferencial (mesma imputacao e padronizacao dos passos 11-13)
(f) ausencias_retorno.csv           ausencias iniciadas em t (presente em t-1, ausente em t,
                                    t <= 2023) na temporada regular, e fracao com retorno depois
(g) lista_2024_extremos.csv          maiores e menores previsoes da lista 2024 por posicao
(g2) lista_2024_quintis.csv          mediana de covariaveis de 2024 por quintil de risco previsto; figura tgb_lista_quintis.pdf
                                    (modelo principal), com previsoes dos 3 modelos e idade/experiencia
                                    cruas; figura tgb_lista_2024.pdf

Convencoes (iguais as do pacote oficial, ver run_oficial.py e diagnostico de erro):
- ``expected_years`` e a media restrita da sobrevivencia ate ``tau`` (tau = 2024 - origem).
- Alvo restrito: ``min(T, tau)``, identificavel quando ha saida (evento=1) ou
  follow-up >= tau. Erro = previsto - alvo: positivo = superestimacao da carreira.
- HR > 1 no Cox significa maior risco de saida (carreira mais curta).
- ``years_exp`` nas tabelas de predicao e padronizado; valores em anos crus vem do
  painel ``df_cox_{POS}.csv`` (linha da temporada de origem).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "run_all.py").exists() and (candidate / "DataBase").exists():
            return candidate
    raise RuntimeError("Project root not found")


ROOT = find_project_root()
MODELOS = ("cox", "gbsa", "rsf")
POSICOES = ("QB", "RB", "WR")
ANO_FIM = 2024  # tau = ANO_FIM - temporada de origem (mesmo do pacote oficial)

OOF = ROOT / "documentos/estudo_modelos_finais/resultados/pred_oof_walkforward_2018_2021.csv"
LISTA_2024 = ROOT / "documentos/estudo_modelos_finais/resultados/predicao_2024_modelos.csv"
PERM = ROOT / "documentos/estudo_leitura_preditiva/resultados/perm_importance_media.csv"
PAINEL_DIR = ROOT / "Inferência/Preparação Modelagem/resultados"
COX_DIR = ROOT / "Inferência/Modelagem de COX/resultados"
AALEN_DIR = ROOT / "Inferência/Modelagem de Aalen/resultados"
FEAT_DIR = ROOT / "Predição/shared"
ANUAL = ROOT / "DataBase/yearly_player_stats_offense.csv"
OUT = ROOT / "documentos/estudo_leitura_aplicada/resultados"

KEY = ["posicao", "player_id", "snapshot_season"]

# Casos no auge (a). Regras em CASE_RULES. Usam so pares jogador x origem dos folds walk-forward
# com desfecho observavel (origens 2018-2021, predicoes out-of-fold). Nada de 2024.
RESUMO_WF = ROOT / "documentos/estudo_modelos_finais/resultados/resumo_walkforward_2018_2021.csv"
TOP_AUGE = 10  # no auge = entre os 10 melhores da posicao na temporada de origem (estatistica abaixo)
TITULAR_MIN_OFF = 0.5  # titular = participacao ofensiva (offense_pct) >= 50% na temporada de origem
ORIGENS_OBS = (2018, 2021)
ESTAT_CHAVE = {
    "QB": "passing_yards",
    "RB": "rushing_yards + receiving_yards",
    "WR": "receiving_yards",
}
CASE_RULES = {
    "acerta_bem": "no auge, desfecho min(T,tau) >= 3 anos (saida observada ou ativo ate tau); menor erro absoluto do modelo principal",
    "saida_abrupta": "no auge, saiu (evento=1) entre 1 ano e tau, antes do previsto (erro > 0 do modelo principal); entre elas, tipo saida_abrupta_apos_alta primeiro, depois o maior erro",
    "durou_alem": "no auge, ainda ativo ao fim de tau (evento=0); menor erro do modelo principal (previu menos do que o minimo observado)",
    "veterano": "no auge, maior experiencia crua na origem entre os pares ainda nao escolhidos",
}
ORDEM_CASOS = ("acerta_bem", "saida_abrupta", "durou_alem", "veterano")

# Tipos de trajetoria (b). Atribuicao por jogador, com prioridade na ordem abaixo.
#   1. alem_do_previsto: sem saida ate o horizonte (residual_event = 0 e identificavel).
#   2. hiato: ha temporada ausente entre a primeira e a ultima temporada REG (DataBase).
#   3. saida_apos_queda: participacao (offense_pct do painel) caiu em dois anos
#      consecutivos antes da saida: off(L-2) > off(L-1) > off(L), L = ultima temporada REG.
#   4. saida_abrupta_apos_alta: saida com L acima da mediana da posicao-temporada em
#      participacao (offense_pct) OU em estatistica principal (QB: passing_yards;
#      RB: rushing_yards + receiving_yards; WR: receiving_yards).
#   5. saida_outra: saida sem nenhuma das condicoes acima.
#   6. sem_trajetoria: jogador sem registro REG na DataBase ou sem painel na temporada de saida.
TIPOS = (
    "alem_do_previsto",
    "hiato",
    "saida_apos_queda",
    "saida_abrupta_apos_alta",
    "saida_outra",
    "sem_trajetoria",
)

def arredonda(df: pd.DataFrame, casas: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(casas)
    return out


def estat_principal(pos: str, df: pd.DataFrame) -> pd.Series:
    if pos == "QB":
        return df["passing_yards"]
    if pos == "RB":
        return df["rushing_yards"].fillna(0) + df["receiving_yards"].fillna(0)
    return df["receiving_yards"]


# ---------------------------------------------------------------- carregamento
def carregar() -> dict:
    oof = pd.read_csv(OOF, dtype={"player_id": str})
    lista = pd.read_csv(LISTA_2024, dtype={"player_id": str})
    perm = pd.read_csv(PERM)
    anual = pd.read_csv(
        ANUAL,
        usecols=[
            "player_id", "player_name", "position", "season", "season_type", "team",
            "passing_yards", "rushing_yards", "receiving_yards",
        ],
        dtype={"player_id": str},
    )
    anual = anual[anual.season_type == "REG"].copy()
    painel = {}
    for pos in POSICOES:
        df = pd.read_csv(PAINEL_DIR / f"df_cox_{pos}.csv", dtype={"player_id": str})
        if df.duplicated(["player_id", "season"]).any():
            raise SystemExit(f"painel {pos} com linhas duplicadas por jogador-temporada")
        painel[pos] = df
    return {"oof": oof, "lista": lista, "perm": perm, "anual": anual, "painel": painel}


def verificar_erro(oof: pd.DataFrame) -> None:
    """Confere que identificavel e alvo restrito batem com a regra (evento ou follow-up >= tau)."""
    ident = (oof["residual_event"] == 1) | (oof["residual_duration_model"] >= oof["tau"])
    if not np.array_equal(ident.to_numpy(), oof["error_identifiable"].astype(bool).to_numpy()):
        raise SystemExit("error_identifiable diverge da regra evento==1 ou follow-up>=tau")
    alvo = np.where(ident, np.minimum(oof["residual_duration_model"], oof["tau"]), np.nan)
    if not np.allclose(alvo, oof["error_target_years"].astype(float), equal_nan=True):
        raise SystemExit("error_target_years diverge de min(T, tau) nos identificaveis")


def wide_oof(oof: pd.DataFrame) -> pd.DataFrame:
    base_cols = KEY + ["player_name", "residual_event", "residual_duration_model",
                       "error_identifiable", "error_target_years", "tau"]
    base = oof[base_cols].drop_duplicates(KEY).set_index(KEY)
    if (oof.groupby(KEY).size() != len(MODELOS)).any():
        raise SystemExit("OOF: chave posicao/jogador/origem sem os 3 modelos")
    for m in MODELOS:
        sub = oof[oof.modelo == m].set_index(KEY)[["expected_years", "median_years", "risk"]]
        sub.columns = [f"{c}_{m}" for c in sub.columns]
        base = base.join(sub)
    base = base.reset_index()
    for m in MODELOS:
        base[f"erro_{m}"] = np.where(
            base.error_identifiable.astype(bool), base[f"expected_years_{m}"] - base.error_target_years, np.nan
        )
    base["erro_medio"] = base[[f"erro_{m}" for m in MODELOS]].mean(axis=1)
    base["erro_abs_medio"] = base[[f"erro_{m}" for m in MODELOS]].abs().mean(axis=1)
    return base


def wide_lista(lista: pd.DataFrame) -> pd.DataFrame:
    chave = ["posicao", "player_id"]
    base = lista[chave + ["player_name"]].drop_duplicates(chave).set_index(chave)
    for m in MODELOS:
        sub = lista[lista.modelo == m].set_index(chave)[
            ["expected_years", "median_years", "risk", "rank", "tau_effective"]
        ]
        sub.columns = [f"{c}_{m}" for c in sub.columns]
        base = base.join(sub)
    return base.reset_index()


def valor_painel(painel: dict, pos: str, pid: str, season: int, col: str) -> float:
    """Valor cru de `col` na linha jogador x temporada do painel inferencial (NaN se ausente)."""
    df = painel[pos]
    linha = df[(df.player_id == pid) & (df.season == season)]
    if linha.empty or col not in linha.columns:
        return np.nan
    v = linha.iloc[0][col]
    return float(v) if pd.notna(v) else np.nan


def anos_crus(painel: dict, pos: str, pid: str, season: int) -> float:
    return valor_painel(painel, pos, pid, season, "years_exp")


# ---------------------------------------------------------------- trajetoria
class Carreiras:
    """Fatos da trajetoria de cada jogador numa posicao, a partir da DataBase e do painel."""

    def __init__(self, pos: str, anual: pd.DataFrame, painel: pd.DataFrame):
        self.pos = pos
        an = anual[anual.position == pos]
        self.por_temp = (
            an.groupby(["player_id", "season"])
            .agg(
                team=("team", lambda s: "/".join(sorted(set(s)))),
                passing_yards=("passing_yards", "sum"),
                rushing_yards=("rushing_yards", "sum"),
                receiving_yards=("receiving_yards", "sum"),
            )
            .reset_index()
        )
        self.por_temp["stat"] = estat_principal(pos, self.por_temp)
        self.stat_map = {(r.player_id, int(r.season)): float(r.stat) for r in self.por_temp.itertuples()}
        self.mediana_stat = self.por_temp.groupby("season")["stat"].median()
        self.painel = painel.set_index(["player_id", "season"])
        self.mediana_off = painel.groupby("season")["offense_pct"].median()
        self.fatos = {}
        for pid, g in self.por_temp.groupby("player_id"):
            seasons = sorted(g.season.astype(int))
            self.fatos[pid] = {
                "primeira": seasons[0],
                "ultima": seasons[-1],
                "temporadas": seasons,
                "gap": (seasons[-1] - seasons[0] + 1) > len(seasons),
            }

    def stat(self, pid: str, season: int) -> float:
        return self.stat_map.get((pid, season), np.nan)

    def off(self, pid: str, season: int):
        try:
            v = self.painel.loc[(pid, season), "offense_pct"]
        except KeyError:
            return np.nan
        return float(v) if pd.notna(v) else np.nan

    def trajetoria(self, pid: str) -> str:
        if pid not in self.fatos:
            return ""
        partes = []
        linhas = self.por_temp[self.por_temp.player_id == pid].sort_values("season")
        for _, r in linhas.iterrows():
            s = int(r.season)
            try:
                gm = float(self.painel.loc[(pid, s), "games_missed"])
            except KeyError:
                gm = np.nan
            off = self.off(pid, s)
            gm_txt = "NA" if pd.isna(gm) else f"{gm:.0f}"
            off_txt = "NA" if pd.isna(off) else f"{off:.2f}"
            partes.append(f"{s}|{r.team}|off={off_txt}|jogos_perdidos={gm_txt}")
        return ";".join(partes)

    def tipo(self, pid: str, residual_event: int, identificavel: bool) -> str:
        if int(residual_event) == 0 and bool(identificavel):
            return "alem_do_previsto"
        fato = self.fatos.get(pid)
        if fato is None:
            return "sem_trajetoria"
        if fato["gap"]:
            return "hiato"
        L = fato["ultima"]
        o2, o1, o0 = self.off(pid, L - 2), self.off(pid, L - 1), self.off(pid, L)
        if all(pd.notna(x) for x in (o2, o1, o0)) and o2 > o1 > o0:
            return "saida_apos_queda"
        linha = self.por_temp[(self.por_temp.player_id == pid) & (self.por_temp.season == L)]
        if linha.empty or pd.isna(o0):
            return "sem_trajetoria"
        stat_L = float(linha.iloc[0]["stat"])
        alta_off = pd.notna(o0) and o0 >= self.mediana_off.get(L, np.inf)
        alta_stat = stat_L >= self.mediana_stat.get(L, np.inf)
        if alta_off or alta_stat:
            return "saida_abrupta_apos_alta"
        return "saida_outra"


# ---------------------------------------------------------------- (a) casos no auge
def melhor_modelo_por_posicao() -> dict[str, tuple[str, float]]:
    """Modelo de maior C-index (walk-forward 2018-2021) por posicao."""
    res = pd.read_csv(RESUMO_WF)
    res = res[res.modelo.isin(MODELOS)]
    out = {}
    for pos in POSICOES:
        g = res[res.posicao == pos].sort_values(["c_index", "modelo"], ascending=[False, True])
        out[pos] = (str(g.iloc[0].modelo), float(g.iloc[0].c_index))
    return out


def ranking_auge(cf: "Carreiras", painel: pd.DataFrame, season: int) -> dict[str, int]:
    """Posicao (1 = melhor) de cada titular na temporada, pela estatistica principal."""
    tit = painel[(painel.season == season) & (painel.offense_pct >= TITULAR_MIN_OFF)]
    pares = []
    for pid in tit.player_id:
        v = cf.stat(pid, season)
        if pd.notna(v):
            pares.append((-v, pid))
    return {pid: i + 1 for i, (_, pid) in enumerate(sorted(pares))}


def pares_no_auge(dados: dict, carreiras: dict, pos: str, modelo: str) -> pd.DataFrame:
    w = dados["_w"]
    cf = carreiras[pos]
    painel = dados["painel"]
    wp = w[(w.posicao == pos) & w.error_identifiable.astype(bool)
           & w.snapshot_season.between(*ORIGENS_OBS)].copy()
    cache: dict[int, dict[str, int]] = {}
    rank = []
    for pid, s in zip(wp.player_id, wp.snapshot_season.astype(int)):
        if s not in cache:
            cache[s] = ranking_auge(cf, painel[pos], s)
        rank.append(cache[s].get(pid, np.nan))
    wp["rank_origem"] = pd.to_numeric(pd.Series(rank, index=wp.index), errors="coerce")
    wp = wp[wp.rank_origem <= TOP_AUGE].copy()
    pares = list(zip(wp.player_id, wp.snapshot_season.astype(int)))
    wp["stat_origem"] = [cf.stat(p, s) for p, s in pares]
    wp["idade_origem"] = [valor_painel(painel, pos, p, s, "age") for p, s in pares]
    wp["exp_origem"] = [anos_crus(painel, pos, p, s) for p, s in pares]
    wp["offense_origem"] = [valor_painel(painel, pos, p, s, "offense_pct") for p, s in pares]
    wp["tipo_trajetoria"] = [cf.tipo(p, int(e), True) for p, e in zip(wp.player_id, wp.residual_event)]
    wp["carreira_total"] = wp["exp_origem"] + wp["residual_duration_model"]
    return wp


def escolhe_casos(pool: pd.DataFrame, modelo: str) -> dict[str, pd.Series]:
    col = f"erro_{modelo}"
    usados: set[str] = set()
    escolhidos: dict[str, pd.Series] = {}

    def pega(cands: pd.DataFrame, regra: str) -> None:
        cands = cands[~cands.player_id.isin(usados)]
        if cands.empty:
            raise SystemExit(f"regra {regra}: nenhum par no auge disponivel")
        r = cands.iloc[0]
        usados.add(r.player_id)
        escolhidos[regra] = r

    # Ordem de escolha: as regras mais restritas primeiro; a figura usa ORDEM_CASOS.
    saida = pool[(pool.residual_event == 1) & (pool[col] > 0) & (pool.error_target_years >= 1)]
    saida = saida.assign(_nao_abrupta=(saida.tipo_trajetoria != "saida_abrupta_apos_alta").astype(int))
    pega(saida.sort_values(["_nao_abrupta", col, "player_id"], ascending=[True, False, True]),
         "saida_abrupta")
    pega(pool[pool.residual_event == 0].sort_values([col, "player_id"], ascending=[True, True]),
         "durou_alem")
    pega(pool.sort_values(["exp_origem", "player_id"], ascending=[False, True]), "veterano")
    acerta = pool[pool.error_target_years >= 3].assign(_abs=lambda d: d[col].abs())
    pega(acerta.sort_values(["_abs", "player_id"], ascending=[True, True]), "acerta_bem")
    return escolhidos


def linha_caso(carreiras: dict, pos: str, modelo: str, regra: str, r: pd.Series) -> dict:
    pid = r.player_id
    fato = carreiras[pos].fatos.get(pid, {})
    linha = {
        "tipo_caso": regra,
        "descricao_regra": CASE_RULES[regra],
        "posicao": pos,
        "player_id": pid,
        "jogador": r.player_name,
        "origem": int(r.snapshot_season),
        "idade_origem": r.idade_origem,
        "experiencia_origem_anos": r.exp_origem,
        "estatistica_chave": ESTAT_CHAVE[pos],
        "estatistica_valor_origem": r.stat_origem,
        "rank_no_auge": int(r.rank_origem),
        "offense_pct_origem": r.offense_origem,
        "modelo_principal": modelo,
        "tau": float(r.tau),
    }
    for m in MODELOS:
        linha[f"previsao_{m}"] = r[f"expected_years_{m}"]
    linha["desfecho_min_T_tau"] = float(r.error_target_years)
    linha["saiu_evento"] = int(r.residual_event)
    linha["tempo_residual_obs"] = float(r.residual_duration_model)
    for m in MODELOS:
        linha[f"erro_{m}"] = r[f"erro_{m}"]
    linha["erro_principal"] = r[f"erro_{modelo}"]
    linha["tipo_trajetoria"] = r.tipo_trajetoria
    linha["carreira_total_anos"] = r.carreira_total
    linha["temporadas_REG"] = f"{fato.get('primeira', '')}-{fato.get('ultima', '')} ({len(fato.get('temporadas', []))})"
    linha["trajetoria_completa"] = carreiras[pos].trajetoria(pid)
    return linha


def construir_casos(dados: dict, carreiras: dict) -> pd.DataFrame:
    linhas = []
    for pos in POSICOES:
        modelo, _c = dados["principal"][pos]
        pool = pares_no_auge(dados, carreiras, pos, modelo)
        print(f"{pos}: pool no auge = {len(pool)} pares ({pool.player_id.nunique()} jogadores), modelo principal {modelo}", flush=True)
        escolhidos = escolhe_casos(pool, modelo)
        for regra in ORDEM_CASOS:
            linhas.append(linha_caso(carreiras, pos, modelo, regra, escolhidos[regra]))
    return arredonda(pd.DataFrame(linhas))


# ---------------------------------------------------------------- (b) tipologia
def tipologia(dados: dict, carreiras: dict, w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    w["tipo"] = [
        carreiras[p].tipo(pid, int(ev), bool(ident))
        for p, pid, ev, ident in zip(w.posicao, w.player_id, w.residual_event, w.error_identifiable)
    ]
    saida = []
    for m in MODELOS:
        cols = ["posicao", "player_id", "tipo", "error_identifiable"]
        lo = w[cols].copy()
        lo["modelo"] = m
        lo["erro"] = w[f"erro_{m}"].to_numpy()
        saida.append(lo)
    longo = pd.concat(saida, ignore_index=True)
    longo = longo[longo.error_identifiable.astype(bool)]
    linhas = []
    for (pos, mod), g in longo.groupby(["posicao", "modelo"]):
        for tipo in ("todos",) + TIPOS:
            sub = g if tipo == "todos" else g[g.tipo == tipo]
            if sub.empty:
                continue
            e = sub["erro"].to_numpy(float)
            linhas.append(
                {
                    "posicao": pos,
                    "modelo": mod,
                    "tipo": tipo,
                    "n_pares": len(sub),
                    "n_jogadores": sub.player_id.nunique(),
                    "erro_medio_anos": e.mean(),
                    "erro_mediano_anos": np.median(e),
                    "mae_anos": np.abs(e).mean(),
                    "pct_superestima": 100 * (e > 0).mean(),
                    "pct_subestima": 100 * (e < 0).mean(),
                }
            )
    return arredonda(pd.DataFrame(linhas))


# ---------------------------------------------------------------- (c) concordancia
def concordancia(dados: dict, w: pd.DataFrame):
    oof_p = w[["posicao", "player_id", "player_name", "snapshot_season"]].copy()
    for m in MODELOS:
        oof_p[f"esp_{m}"] = w[f"expected_years_{m}"].to_numpy()
    oof_p = oof_p.rename(columns={"snapshot_season": "fold"})
    lp = dados["_lista_w"][["posicao", "player_id", "player_name"]].copy()
    for m in MODELOS:
        lp[f"esp_{m}"] = dados["_lista_w"][f"expected_years_{m}"].to_numpy()
    lp["fold"] = ANO_FIM
    todos = pd.concat([oof_p, lp], ignore_index=True)
    todos["fold"] = todos["fold"].astype(int)

    sp = []
    div = []
    for (pos, fold), g in todos.groupby(["posicao", "fold"]):
        g = g.reset_index(drop=True)
        rho = {}
        for a, b in (("cox", "gbsa"), ("cox", "rsf"), ("gbsa", "rsf")):
            rho[f"rho_{a}_{b}"] = spearmanr(g[f"esp_{a}"], g[f"esp_{b}"]).statistic
        sp.append({"posicao": pos, "fold": fold, "n_jogadores": len(g), **rho})
        pct = pd.DataFrame({m: g[f"esp_{m}"].rank(pct=True) for m in MODELOS})
        g["divergencia_pp"] = 100 * (pct.max(axis=1) - pct.min(axis=1))
        for m in MODELOS:
            g[f"pct_{m}"] = pct[m]
        div.append(g)
    sp_df = arredonda(pd.DataFrame(sp))
    div_df = pd.concat(div, ignore_index=True)
    # Retrato e desfecho para cada linha de divergencia
    anotado = []
    for _, r in div_df.iterrows():
        pid = r.player_id
        origem = int(r.fold)
        pos = r.posicao
        ret = _retrato(dados, pos, pid, origem)
        desf = w[(w.posicao == pos) & (w.player_id == pid) & (w.snapshot_season == origem)]
        anotado.append(
            {
                "posicao": pos,
                "fold": origem,
                "player_id": pid,
                "player_name": r.player_name,
                "divergencia_pp": r.divergencia_pp,
                "esp_cox": r.esp_cox,
                "esp_gbsa": r.esp_gbsa,
                "esp_rsf": r.esp_rsf,
                "pct_cox": r.pct_cox,
                "pct_gbsa": r.pct_gbsa,
                "pct_rsf": r.pct_rsf,
                "retrato_origem": ret,
                "desfecho_alvo_restrito": (
                    float(desf.error_target_years.iloc[0]) if len(desf) and pd.notna(desf.error_target_years.iloc[0]) else np.nan
                ),
                "desfecho_saiu": (int(desf.residual_event.iloc[0]) if len(desf) else np.nan),
            }
        )
    top = (
        pd.DataFrame(anotado)
        .sort_values(["posicao", "divergencia_pp", "player_id"], ascending=[True, False, True])
        .groupby("posicao")
        .head(10)
    )
    return sp_df, arredonda(top)


def _retrato(dados: dict, pos: str, pid: str, origem: int) -> str:
    partes = []
    for c in ["years_exp", "offense_pct", "games_missed"] + dados["cox_finais"][pos]:
        v = valor_painel(dados["painel"], pos, pid, origem, c)
        partes.append(f"{c}={v:.3f}" if pd.notna(v) else f"{c}=NA")
    return ";".join(partes)


# ---------------------------------------------------------------- inferencia
def carregar_inferencia(perm: pd.DataFrame) -> dict:
    cox_finais, aalen_finais = {}, {}
    cox_hr, aalen_info = {}, {}
    for pos in POSICOES:
        hr = pd.read_csv(COX_DIR / f"cox_hr_{pos}.csv")
        cox_hr[pos] = hr.set_index("covariate")
        cox_finais[pos] = list(hr["covariate"])
        covs_a = json.loads((AALEN_DIR / f"aalen_final_covs_{pos}.json").read_text(encoding="utf-8"))
        cum = pd.read_csv(AALEN_DIR / f"aalen_cum_coefs_{pos}.csv")
        ultimo = cum.dropna().iloc[-1]
        scr = pd.read_csv(AALEN_DIR / f"aalen_screening_{pos}.csv").set_index("covariate")
        aalen_info[pos] = {"covs": covs_a, "cum": cum, "ultimo": ultimo, "screening": scr}
        aalen_finais[pos] = covs_a
    return {
        "cox_finais": cox_finais,
        "aalen_finais": aalen_finais,
        "cox_hr": cox_hr,
        "aalen": aalen_info,
    }


def inferencia_vs_importancia(dados: dict) -> pd.DataFrame:
    perm = dados["perm"]
    inf = dados["inf"]
    linhas = []
    for pos in POSICOES:
        pool = sorted(set(perm.loc[perm.posicao == pos, "feature"]))
        covs_cox = inf["cox_finais"][pos]
        covs_aal = inf["aalen_finais"][pos]
        todas = pool + [c for c in covs_cox if c not in pool]
        rk = {}
        for m in MODELOS:
            sub = perm[(perm.posicao == pos) & (perm.modelo == m)].copy()
            sub["rank"] = sub["perm_drop"].rank(ascending=False, method="min")
            rk[m] = sub.set_index("feature")
        scr = inf["aalen"][pos]["screening"]
        hr = inf["cox_hr"][pos]
        ultimo = inf["aalen"][pos]["ultimo"]
        for cov in todas:
            hr_val = float(hr.loc[cov, "HR"]) if cov in hr.index else np.nan
            p_cox = float(hr.loc[cov, "p_value"]) if cov in hr.index else np.nan
            if cov in covs_aal:
                bval = float(ultimo[cov])
                b_sent = "aumenta risco de saida" if bval > 0 else "reduz risco de saida"
                p_aal = float(scr.loc[cov, "p_value_null"]) if cov in scr.index else np.nan
            else:
                bval, b_sent, p_aal = np.nan, "", np.nan
            linha = {
                "posicao": pos,
                "covariavel": cov,
                "esta_no_pool_preditivo": cov in pool,
                "esta_no_cox_final": cov in covs_cox,
                "cox_HR": hr_val,
                "cox_sentido": (
                    "" if np.isnan(hr_val) else ("aumenta risco de saida" if hr_val > 1 else "reduz risco de saida")
                ),
                "cox_p": p_cox,
                "esta_no_aalen_final": cov in covs_aal,
                "aalen_B_ultimo_tempo": bval,
                "aalen_sentido": b_sent,
                "aalen_p_nulo_triagem": p_aal,
            }
            for m in MODELOS:
                if cov in rk[m].index:
                    linha[f"rank_perm_{m}"] = int(rk[m].loc[cov, "rank"])
                    linha[f"perm_drop_{m}"] = float(rk[m].loc[cov, "perm_drop"])
                else:
                    linha[f"rank_perm_{m}"] = np.nan
                    linha[f"perm_drop_{m}"] = np.nan
            linhas.append(linha)
    return arredonda(pd.DataFrame(linhas))


def escalonador_cox(pos: str) -> dict[str, float]:
    """DP do StandardScaler do Cox inferencial, por covariavel.

    Replica os passos 11-13: imputacao por survival_utils.fit_imputation_stats /
    apply_imputation_stats sobre o pkl do painel, covariaveis = todas as colunas exceto
    identificadores, tempo e draft_round (antes do filtro de colinearidade), e
    StandardScaler().fit. O DP do sklearn e o populacional (ddof=0); scale_ e esse DP.
    """
    import sys

    from sklearn.preprocessing import StandardScaler

    shared = str(ROOT / "Inferência" / "shared")
    if shared not in sys.path:
        sys.path.append(shared)
    import survival_utils as su

    df = pd.read_pickle(PAINEL_DIR / f"df_cox_{pos}.pkl")
    id_cols = ["player_id", "player_name", "season"]
    time_cols = ["start", "stop", "event", "entry"]
    cov = [c for c in df.columns if c not in id_cols + time_cols and c != "draft_round"]
    stats = su.fit_imputation_stats(df, cov)
    dm = su.apply_imputation_stats(df, cov, stats)
    scaler = StandardScaler().fit(dm[cov])
    return dict(zip(cov, (float(v) for v in scaler.scale_)))


def dp_covariaveis(dados: dict) -> pd.DataFrame:
    linhas = []
    for pos in POSICOES:
        pool = sorted(set(dados["perm"].loc[dados["perm"].posicao == pos, "feature"]))
        cox = dados["inf"]["cox_finais"][pos]
        painel = dados["painel"][pos]
        feats = pd.read_pickle(FEAT_DIR / f"df_cox_{pos}_features.pkl")
        hr = dados["inf"]["cox_hr"][pos]
        escal = escalonador_cox(pos)
        for cov in sorted(set(pool) | set(cox)):
            if cov in painel.columns:
                s, fonte = painel[cov], "Inferência/Preparação Modelagem/resultados/df_cox_%s.csv" % pos
            elif cov in feats.columns:
                s, fonte = feats[cov], "Predição/shared/df_cox_%s_features.pkl (ausente no painel inferencial)" % pos
            else:
                continue
            s = pd.to_numeric(s, errors="coerce")
            media, dp = float(s.mean()), float(s.std(ddof=1))
            dp_esc = escal.get(cov, np.nan)
            linha = {
                "posicao": pos,
                "covariavel": cov,
                "media_crua": media,
                "dp_cru": dp,
                "dp_escalonador": dp_esc,
                "n_observado": int(s.notna().sum()),
                "n_faltante": int(s.isna().sum()),
                "minimo": float(s.min()),
                "maximo": float(s.max()),
                "em_cox_final": cov in cox,
                "em_pool_preditivo": cov in pool,
                "fonte": fonte,
                "cox_HR_por_DP": np.nan,
                "cox_HR_por_unidade": np.nan,
            }
            if cov in cox and cov in hr.index and not np.isnan(dp_esc) and dp_esc > 0:
                h = float(hr.loc[cov, "HR"])
                linha["cox_HR_por_DP"] = h
                linha["cox_HR_por_unidade"] = h ** (1.0 / dp_esc)
            linhas.append(linha)
    return arredonda(pd.DataFrame(linhas))


def ausencias_retorno(anual: pd.DataFrame) -> pd.DataFrame:
    """Ausencias iniciadas em t (presente em t-1, ausente em t, t <= 2023), temporada regular.

    Populacao QB_RB_WR: linhas de temporada regular de jogadores QB, RB ou WR.
    Populacao todas_posicoes: todas as linhas de temporada regular do CSV bruto, sem filtro
    de posicao (reconcilia com 258 de 1.227 citado antes da reescrita).
    Retorno = presenca em alguma temporada posterior a t.
    """
    linhas = []
    populacoes = [
        ("QB_RB_WR", anual[anual.position.isin(POSICOES)], "temporada regular, QB, RB ou WR"),
        ("todas_posicoes", anual, "temporada regular, todas as posicoes do CSV"),
    ]
    for nome, base, descricao in populacoes:
        n_aus = n_ret = 0
        jogadores: set[str] = set()
        for pid, g in base.groupby("player_id"):
            temporadas = set(int(x) for x in g.season)
            for t in range(min(temporadas) + 1, 2024):
                if (t - 1) in temporadas and t not in temporadas:
                    n_aus += 1
                    jogadores.add(pid)
                    n_ret += int(any(x > t for x in temporadas))
        linhas.append(
            {
                "populacao": nome,
                "descricao": descricao,
                "n_jogadores_com_ausencia": len(jogadores),
                "n_ausencias_uma_temporada": n_aus,
                "n_com_retorno_depois": n_ret,
                "fracao_com_retorno": (n_ret / n_aus) if n_aus else np.nan,
            }
        )
    return arredonda(pd.DataFrame(linhas), casas=4)


# ---------------------------------------------------------------- figura
NAVY_FIG = "#2E4057"  # mesmas cores de documentos/estudo_diagnostico_erro/run_diagnostico.py
ORANGE_FIG = "#D85A30"
COR_MEDIANA_FIG = "#7B3FA0"  # linha da mediana da lista 2024 (cor propria, distinta dos dois grupos)
COR_RSF = "#1D9E75"
# (modelo, marcador, cor, deslocamento vertical da previsao dentro da linha do jogador)
DESLOC_MODELO = (("cox", "o", NAVY_FIG, 0.24), ("gbsa", "s", ORANGE_FIG, 0.0), ("rsf", "^", COR_RSF, -0.24))


def figura_casos(casos: pd.DataFrame, principal: dict[str, tuple[str, float]]) -> Path:
    """Casos no auge: uma linha por jogador, previsoes de cada modelo ligadas ao desfecho.

    Ponto preto cheio = saida observada (anos ate a saida); ponto vazado = ainda ativo ao fim de
    tau (observado = tau, valor minimo). Um painel por posicao, linhas ordenadas por erro do modelo
    principal (maior superestimacao no topo).
    """
    import sys

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if str(FEAT_DIR) not in sys.path:
        sys.path.insert(0, str(FEAT_DIR))
    from thesis_figure_style import (  # noqa: E402
        apply_thesis_rcparams,
        eixos_virgula,
        figsize_full,
        num_br,
        save_thesis_figure,
        thesis_model_label,
    )

    apply_thesis_rcparams()
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    })
    df = casos.copy()
    num = ["previsao_cox", "previsao_gbsa", "previsao_rsf", "desfecho_min_T_tau", "erro_principal"]
    for col in num + ["saiu_evento"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    fw = figsize_full()[0]
    fh = 6.0  # altura util da pagina (0,6 da altura de texto): escala ~1 no LaTeX
    fig, axes = plt.subplots(len(POSICOES), 1, figsize=(fw, fh), sharex=True)
    xmax = float(df[num[:4]].max().max()) + 1.0

    for ax, pos in zip(axes, POSICOES):
        sub = df[df.posicao == pos].sort_values(["erro_principal", "jogador"], ascending=[False, True])
        n = len(sub)
        modelo, c = principal[pos]
        yticks, ylabels = [], []
        for i, (_, r) in enumerate(sub.iterrows()):
            y = n - 1 - i
            yticks.append(y)
            ylabels.append(f"{r.jogador} ({int(r.origem)})")
            obs = r.desfecho_min_T_tau
            for mod, marc, cor, desl in DESLOC_MODELO:
                pred = r[f"previsao_{mod}"]
                ax.plot([obs, pred], [y, y + desl], color=cor, linewidth=0.8, alpha=0.7, zorder=2)
                ax.scatter(pred, y + desl, marker=marc, s=34, color=cor, edgecolor="white",
                           linewidth=0.4, zorder=3)
            if r.saiu_evento == 1:
                ax.scatter(obs, y, marker="o", s=44, color="black", zorder=4)
            else:
                ax.scatter(obs, y, marker="o", s=44, facecolors="white", edgecolors="black",
                           linewidth=1.1, zorder=4)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_title(f"{pos}: modelo principal {thesis_model_label(modelo)} (C = {num_br(c, 3)})",
                     loc="left", fontweight="bold")
        ax.xaxis.grid(True, linestyle=":", linewidth=0.45, color="0.65")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlim(-0.3, xmax)
    for ax in axes:
        eixos_virgula(ax, x=True, y=False)
    axes[-1].set_xlabel("anos restantes a partir da temporada de origem (média restrita até τ)")

    handles = [
        Line2D([], [], marker=marc, linestyle="-", color=cor, markersize=4.5, label=thesis_model_label(mod))
        for mod, marc, cor, _d in DESLOC_MODELO
    ] + [
        Line2D([], [], marker="o", linestyle="", color="black", label="saiu (observado)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black",
               label="ativo até τ (observado ≥ τ)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.005), handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.8)
    caminho = OUT / "tgb_casos_leitura.pdf"
    save_thesis_figure(fig, caminho, also_png=False)
    plt.close(fig)
    return caminho


# ---------------------------------------------------------------- (d) lista 2024
N_EXTREMOS = 6  # jogadores ativos em 2024 com mais e com menos anos previstos, por posicao
NOMES_REF = ("Brock Purdy", "Lamar Jackson")  # marcados se nao entrarem em top/bottom


def extremos_lista_2024(dados: dict, carreiras: dict) -> pd.DataFrame:
    """Top e bottom N_EXTREMOS da lista 2024 pelo modelo principal de cada posicao.

    Previsao = media restrita ate tau (anos restantes a partir de 2024). Idade e experiencia
    sao cruas (painel inferencial, temporada 2024); estatistica-chave e da DataBase em 2024.
    """
    w = dados["_lista_w"]
    painel = dados["painel"]
    linhas = []
    for pos in POSICOES:
        modelo, _c = dados["principal"][pos]
        sub = w[w.posicao == pos].copy()
        sub["prev_principal"] = sub[f"expected_years_{modelo}"]
        sub = sub.sort_values(["prev_principal", "player_id"], ascending=[False, True]).reset_index(drop=True)
        sub["rank_principal"] = np.arange(1, len(sub) + 1)
        if len(sub) < 2 * N_EXTREMOS:
            raise SystemExit(f"{pos}: lista 2024 com {len(sub)} jogadores, insuficiente para top/bottom")
        grupos = [
            ("maior_previsao", sub.head(N_EXTREMOS)),
            ("menor_previsao", sub.tail(N_EXTREMOS)),
        ]
        ja = set(pd.concat([g for _n, g in grupos]).player_id)
        ref = sub[sub.player_name.isin(NOMES_REF) & ~sub.player_id.isin(ja)]
        grupos.append(("referencia_nominal", ref))
        for grupo, g in grupos:
            for _, r in g.iterrows():
                pid = r.player_id
                linhas.append({
                    "grupo": grupo,
                    "posicao": pos,
                    "rank_principal_posicao": int(r.rank_principal),
                    "player_id": pid,
                    "jogador": r.player_name,
                    "modelo_principal": modelo,
                    "previsao_principal_anos": float(r.prev_principal),
                    "previsao_cox_anos": float(r.expected_years_cox),
                    "previsao_gbsa_anos": float(r.expected_years_gbsa),
                    "previsao_rsf_anos": float(r.expected_years_rsf),
                    "tau_efetivo": float(r[f"tau_effective_{modelo}"]),
                    "idade_crua_2024": valor_painel(painel, pos, pid, ANO_FIM, "age"),
                    "experiencia_crua_2024_anos": anos_crus(painel, pos, pid, ANO_FIM),
                    "estatistica_chave": ESTAT_CHAVE[pos],
                    "estatistica_valor_2024": carreiras[pos].stat(pid, ANO_FIM),
                    "nome_referencia_nominal": bool(r.player_name in NOMES_REF),
                })
    return arredonda(pd.DataFrame(linhas))


def figura_lista_2024(extremos: pd.DataFrame, principal: dict[str, tuple[str, float]]) -> Path:
    """Maiores e menores previsoes de anos restantes por posicao (lista 2024, modelo principal).

    Um painel por posição. Bolinha = previsão do modelo principal (média restrita até τ).
    Os dois grupos ficam separados por linha tracejada.
    """
    import sys

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if str(FEAT_DIR) not in sys.path:
        sys.path.insert(0, str(FEAT_DIR))
    from thesis_figure_style import (  # noqa: E402
        apply_thesis_rcparams,
        eixos_virgula,
        figsize_full,
        num_br,
        save_thesis_figure,
        thesis_model_label,
    )

    apply_thesis_rcparams()
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    })
    # A referencia nominal fica so na tabela: a figura mostra os dois extremos.
    df = extremos[extremos.grupo.isin(("maior_previsao", "menor_previsao"))].copy()
    fw = figsize_full()[0]
    grupos_por_pos = {}
    for pos in POSICOES:
        sub = df[df.posicao == pos]
        grupos_por_pos[pos] = {g: sub[sub.grupo == g] for g in ("maior_previsao", "menor_previsao")}
    n_linhas = {p: sum(len(v) for v in grupos_por_pos[p].values()) for p in POSICOES}
    tau = float(df.tau_efetivo.max())
    alturas = [n_linhas[p] + 2.0 for p in POSICOES]  # mesma altura por linha em todos os paineis
    fig, axes = plt.subplots(
        len(POSICOES), 1, figsize=(fw, 0.135 * sum(alturas) + 0.8),
        gridspec_kw={"height_ratios": alturas}, sharex=True,
    )

    lista_2024 = pd.read_csv(LISTA_2024)
    for ax, pos in zip(axes, POSICOES):
        modelo, c = principal[pos]
        grupos = grupos_por_pos[pos]
        # ordem de cima para baixo: maiores e depois menores
        ordem = [(g, r) for g in ("maior_previsao", "menor_previsao") for _, r in grupos[g].iterrows()]
        n = len(ordem)
        nomes = [r.jogador for _g, r in ordem]
        ylabels, yticks = [], []
        for i, (g, r) in enumerate(ordem):
            yy = n - 1 - i
            rotulo = f"{r.jogador} [{r.player_id}]" if nomes.count(r.jogador) > 1 else r.jogador
            ylabels.append(rotulo)
            yticks.append(yy)
            v = r.previsao_principal_anos
            cor = {"maior_previsao": NAVY_FIG, "menor_previsao": ORANGE_FIG}[g]
            ax.scatter(v, yy, marker="o", s=30, color=cor, edgecolor="white", linewidth=0.4, zorder=3)
            ax.text(v + 0.18, yy, num_br(v, 2), va="center", ha="left", fontsize=7.5, color="0.15")
        # separadores tracejados entre os grupos
        acc = 0
        for g in ("maior_previsao", "menor_previsao"):
            acc += len(grupos[g])
            if acc < n:  # fronteira apos o ultimo item de cada grupo, se ainda houver linhas abaixo
                ax.axhline(n - 1 - (acc - 1) - 0.5, color="0.55", linestyle="--", linewidth=0.6, zorder=1)
        # Mediana da lista inteira da posicao, e nao so dos extremos desenhados.
        lista_pos = lista_2024[(lista_2024.posicao == pos) & (lista_2024.modelo == modelo)]
        med = float(lista_pos.expected_years.median())
        ax.axvline(med, color=COR_MEDIANA_FIG, linestyle="--", linewidth=1.1, zorder=2)
        ax.text(med + 0.15, n + 0.15, f"mediana da lista = {num_br(med, 2)}", fontsize=8, color=COR_MEDIANA_FIG,
                va="center", ha="left", fontweight="bold", zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_ylim(-0.6, n + 0.6)  # folga no topo para o texto da mediana, acima da primeira linha
        ax.set_title(f"{pos}: modelo principal {thesis_model_label(modelo)} (C = {num_br(c, 3)})",
                     loc="left", fontweight="bold")
        ax.xaxis.grid(False)  # sem grade vertical: a linha da mediana e a unica referencia vertical
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlim(0, tau)
    for ax in axes:
        eixos_virgula(ax, x=True, y=False)
    axes[-1].set_xlabel(f"anos restantes previstos a partir de 2024 (média restrita até τ = {tau:.0f})")
    handles = [
        Line2D([], [], marker="o", linestyle="", color=NAVY_FIG, markersize=5,
               label=f"{N_EXTREMOS} maiores previsões"),
        Line2D([], [], marker="o", linestyle="", color=ORANGE_FIG, markersize=5,
               label=f"{N_EXTREMOS} menores previsões"),
        Line2D([], [], color=COR_MEDIANA_FIG, linestyle="--", linewidth=1.1, label="mediana da lista"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.005), handlelength=1.5, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=0.9)
    caminho = OUT / "tgb_lista_2024.pdf"
    save_thesis_figure(fig, caminho, also_png=False)
    plt.close(fig)
    return caminho


# ---------------------------------------------------------------- (d2) lista 2024 por quintil de risco
FATIA_EQUIPE = {"QB": "team_pass_attempts_share", "RB": "team_rushing_yards_share", "WR": "team_targets_share"}
COLS_QUINTIL = (
    ("fatia", "Fatia da equipe"),
    ("offense_pct", "Participação ofensiva"),
    ("games_missed", "Jogos perdidos"),
    ("years_exp", "Experiência (anos)"),
)


def quintis_lista_2024(dados: dict) -> pd.DataFrame:
    """Mediana de covariaveis de 2024 por quintil de risco previsto pelo modelo principal.

    Usa a lista inteira de ativos em 2024, e nao so os extremos. Quintil 1 = mais anos
    previstos (menor risco); quintil 5 = menos anos previstos (maior risco). Valores crus
    da temporada de 2024, do painel preditivo.
    """
    w = dados["_lista_w"]
    linhas = []
    for pos in POSICOES:
        modelo, _c = dados["principal"][pos]
        sub = w[w.posicao == pos][["player_id", f"expected_years_{modelo}"]].rename(
            columns={f"expected_years_{modelo}": "previsao"})
        painel = pd.read_pickle(FEAT_DIR / f"df_cox_{pos}_features.pkl")
        painel = painel[painel.season == ANO_FIM].rename(columns={FATIA_EQUIPE[pos]: "fatia"})
        cols = [c for c, _r in COLS_QUINTIL]
        x = sub.merge(painel[["player_id"] + cols], on="player_id", how="left", validate="1:1")
        if x[cols].isna().any().any():
            raise SystemExit(f"{pos}: lista 2024 com covariavel ausente no painel de 2024")
        x["quintil"] = pd.qcut((-x["previsao"]).rank(method="first"), 5, labels=False) + 1
        for q, g in x.groupby("quintil"):
            linha = {"posicao": pos, "modelo_principal": modelo, "quintil_risco": int(q), "n": len(g),
                     "previsao_mediana_anos": float(g["previsao"].median())}
            linha.update({c: float(g[c].median()) for c in cols})
            linhas.append(linha)
    return arredonda(pd.DataFrame(linhas))


def figura_quintis_2024(q: pd.DataFrame) -> Path:
    """Mediana de quatro covariaveis por quintil de risco previsto, uma linha por posicao."""
    import sys

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if str(FEAT_DIR) not in sys.path:
        sys.path.insert(0, str(FEAT_DIR))
    from thesis_figure_style import apply_thesis_rcparams, eixos_virgula, figsize_full, save_thesis_figure  # noqa: E402

    apply_thesis_rcparams()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
    cor = {"QB": NAVY_FIG, "RB": ORANGE_FIG, "WR": COR_RSF}
    marca = {"QB": "o", "RB": "s", "WR": "^"}
    fw = figsize_full()[0]
    fig, axes = plt.subplots(2, 2, figsize=(fw, fw * 0.62), sharex=True)
    for ax, (col, rotulo) in zip(axes.ravel(), COLS_QUINTIL):
        for pos in POSICOES:
            s = q[q.posicao == pos].sort_values("quintil_risco")
            ax.plot(s.quintil_risco, s[col], marker=marca[pos], color=cor[pos], linewidth=1.4, markersize=4.5)
        ax.set_title(rotulo, loc="left", fontweight="bold")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        eixos_virgula(ax, x=False, y=True)
    fig.supxlabel("quintil de risco previsto em 2024 (1 = menor risco, 5 = maior risco)", y=0.085, fontsize=9)
    handles = [Line2D([], [], color=cor[p], marker=marca[p], linewidth=1.4, markersize=4.5,
                      label=f"{p} ({'RSF' if q[q.posicao == p].modelo_principal.iloc[0] == 'rsf' else 'Cox'})")
               for p in POSICOES]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.11, 1, 1), h_pad=1.0, w_pad=1.6)
    caminho = OUT / "tgb_lista_quintis.pdf"
    save_thesis_figure(fig, caminho, also_png=False)
    plt.close(fig)
    return caminho


# ---------------------------------------------------------------- (e) termos do Aalen que violaram PH
AALEN_CI_Z = 1.96  # banda pontual de 95% sobre B(t) acumulado (mesma regra de 52_series_figuras.py)


def termos_violacao(pos: str) -> list[str]:
    """Covariaveis do Aalen que violaram proporcionalidade no Schoenfeld do Cox (violacao == TRUE).

    Filtra pelos termos presentes no Aalen final da posicao, na ordem do Schoenfeld.
    """
    sch = pd.read_csv(COX_DIR / f"schoenfeld_{pos}.csv")
    sch = sch[sch["term"] != "GLOBAL"]
    viol = sch.loc[sch["violacao"].astype(str).str.upper() == "TRUE", "term"].tolist()
    colunas = pd.read_csv(AALEN_DIR / f"aalen_cum_coefs_{pos}.csv", nrows=0).columns
    return [t for t in viol if t in colunas]


def tabela_termos_violacao() -> pd.DataFrame:
    """B(t) acumulado do Aalen final e banda pontual de 95% (robvar, cluster por jogador)."""
    linhas = []
    for pos in POSICOES:
        cum = pd.read_csv(AALEN_DIR / f"aalen_cum_coefs_{pos}.csv")
        var = pd.read_csv(AALEN_DIR / f"aalen_robvar_cum_{pos}.csv")
        tempo = cum.columns[0]
        for termo in termos_violacao(pos):
            b = cum[termo].to_numpy(dtype=float)
            ep = np.sqrt(np.maximum(var[termo].to_numpy(dtype=float), 0.0))
            for t, bb, e in zip(cum[tempo], b, ep):
                linhas.append({
                    "posicao": pos,
                    "termo": termo,
                    "ano_carreira": int(t),
                    "B_acumulado": bb,
                    "erro_padrao": e,
                    "ic95_inf": bb - AALEN_CI_Z * e,
                    "ic95_sup": bb + AALEN_CI_Z * e,
                })
    return arredonda(pd.DataFrame(linhas), 6)


def figura_termos_violacao(tab: pd.DataFrame) -> Path:
    """2x2: linhas RB e WR; colunas os dois termos de cada posicao que violaram PH.

    Linha continua = B(t) acumulado; faixa = IC pontual de 95%; linha preta = zero (sem efeito
    acumulado). Eixo x = ano de carreira dentro da janela do Aalen de cada posicao.
    """
    import sys

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if str(FEAT_DIR) not in sys.path:
        sys.path.insert(0, str(FEAT_DIR))
    from thesis_figure_style import (  # noqa: E402
        TEXTWIDTH_IN,
        apply_thesis_rcparams,
        eixos_virgula,
        save_thesis_figure,
        thesis_var_label,
    )

    apply_thesis_rcparams()
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    })
    posicoes = [p for p in POSICOES if (tab.posicao == p).any()]
    fig, axes = plt.subplots(len(posicoes), 2, figsize=(TEXTWIDTH_IN, 1.95 * len(posicoes) + 1.0),
                             sharex="col", squeeze=False)
    for i, pos in enumerate(posicoes):
        termos = list(dict.fromkeys(tab.loc[tab.posicao == pos, "termo"]))
        if len(termos) != 2:
            raise SystemExit(f"{pos}: esperados 2 termos violadores, encontrados {len(termos)}")
        for j, termo in enumerate(termos):
            ax = axes[i, j]
            sub = tab[(tab.posicao == pos) & (tab.termo == termo)].sort_values("ano_carreira")
            t = sub["ano_carreira"].to_numpy()
            ax.fill_between(t, sub["ic95_inf"], sub["ic95_sup"], color=NAVY_FIG, alpha=0.18, linewidth=0, zorder=1)
            ax.plot(t, sub["B_acumulado"], color=NAVY_FIG, marker="o", markersize=3.2, linewidth=1.4, zorder=3)
            ax.axhline(0.0, color="black", linewidth=0.7, zorder=2)
            ax.set_title(f"{pos}: {thesis_var_label(termo)}", loc="left", fontweight="bold")
            ax.set_xticks(t)
            eixos_virgula(ax, x=False, y=True)
            ax.xaxis.grid(True, linestyle=":", linewidth=0.45, color="0.65")
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if j == 0:
                ax.set_ylabel("$\\hat{B}(t)$ acumulado\npor desvio-padrão\nda covariável")
            if i == len(posicoes) - 1:
                ax.set_xlabel("Ano de carreira")
    handles = [
        Line2D([], [], color=NAVY_FIG, marker="o", markersize=3.2, linewidth=1.4, label=r"$\hat{B}(t)$ acumulado"),
        Patch(facecolor=NAVY_FIG, alpha=0.18, linewidth=0, label="IC pontual 95%"),
        Line2D([], [], color="black", linewidth=0.7, label="zero (sem efeito acumulado)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=2.0, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.9, w_pad=1.2)
    caminho = OUT / "aalen_termos_violacao.pdf"
    save_thesis_figure(fig, caminho, also_png=False)
    plt.close(fig)
    return caminho


# ---------------------------------------------------------------- execucao
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dados = carregar()
    verificar_erro(dados["oof"])
    dados["inf"] = carregar_inferencia(dados["perm"])
    dados["cox_finais"] = dados["inf"]["cox_finais"]
    w = wide_oof(dados["oof"])
    dados["_w"] = w
    dados["_lista_w"] = wide_lista(dados["lista"])
    w = dados["_w"]
    carreiras = {
        pos: Carreiras(pos, dados["anual"], dados["painel"][pos]) for pos in POSICOES
    }

    dados["principal"] = melhor_modelo_por_posicao()
    casos = construir_casos(dados, carreiras)
    casos.to_csv(OUT / "casos_auge.csv", index=False)
    (OUT / "casos.csv").unlink(missing_ok=True)
    figura_casos(casos, dados["principal"])

    extremos = extremos_lista_2024(dados, carreiras)
    extremos.to_csv(OUT / "lista_2024_extremos.csv", index=False)
    figura_lista_2024(extremos, dados["principal"])
    quintis = quintis_lista_2024(dados)
    quintis.to_csv(OUT / "lista_2024_quintis.csv", index=False)
    figura_quintis_2024(quintis)

    tip = tipologia(dados, carreiras, w)
    tip.to_csv(OUT / "tipologia_erro.csv", index=False)

    sp, div = concordancia(dados, w)
    sp.to_csv(OUT / "concordancia_modelos.csv", index=False)
    div.to_csv(OUT / "concordancia_divergentes.csv", index=False)

    inferencia_vs_importancia(dados).to_csv(OUT / "inferencia_vs_importancia.csv", index=False)
    dp_covariaveis(dados).to_csv(OUT / "dp_covariaveis.csv", index=False)
    ausencias_retorno(dados["anual"]).to_csv(OUT / "ausencias_retorno.csv", index=False)

    termos = tabela_termos_violacao()
    termos.to_csv(OUT / "aalen_termos_violacao.csv", index=False)
    figura_termos_violacao(termos)

    print(f"casos: {len(casos)} | tipologia: {len(tip)} linhas | concordancia: {len(sp)} folds", flush=True)


if __name__ == "__main__":
    main()
