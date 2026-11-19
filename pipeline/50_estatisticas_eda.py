#!/usr/bin/env python3
"""Estatisticas descritivas, testes univariados e funil de covariáveis.

O capitulo 3 cita medianas de carreira, idade de estreia, medianas de
Kaplan-Meier, p-valores de Kruskal-Wallis, Mann-Whitney e log-rank, duas
correlacoes de colinearidade e as contagens do funil de covariáveis
(42/36/39 candidatas; 37/33/35 sem colunas nao identificaveis; 26/20/25 apos colinearidade; 23/19/23 saturado do Cox). Esses numeros existiam apenas
como anotacao de figura, saida de tela ou conferencia manual.

Este passo recalcula cada um com as MESMAS funcoes que os passos das figuras
usam (`load_clean_position_data`, `add_gap_aware_survival_structure` via o
painel ja preparado, `fit_imputation_stats`, `apply_imputation_stats`,
`filter_collinear_covariates`) e grava o resultado em artefato.

Saidas:

- ``Análise Exploratória de Dados/resultados/estatisticas_eda.csv``
- ``Análise Exploratória de Dados/resultados/testes_eda.csv``
- ``Inferência/Preparação Modelagem/resultados/funil_covariaveis.json``
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.preprocessing import StandardScaler
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
for candidato in [ROOT, *ROOT.parents]:
    if (candidato / "run_all.py").exists() and (candidato / "DataBase").exists():
        ROOT = candidato
        break
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "Inferência" / "shared"))

from survival_utils import (  # noqa: E402
    logrank_entrada_tardia,
    apply_imputation_stats,
    covariaveis_identificaveis_cox,
    filter_collinear_covariates,
    fit_imputation_stats,
    force_missing_flags,
    load_clean_position_data,
)

POSICOES = ("QB", "RB", "WR")
PREP_RES = ROOT / "Inferência" / "Preparação Modelagem" / "resultados"
EDA_RES = ROOT / "Análise Exploratória de Dados" / "resultados"
ID_COLS = ["player_id", "player_name", "season"]
TIME_COLS = ["start", "stop", "event", "entry"]

# Pares citados na escrita para ilustrar a colinearidade grave.
PARES_CITADOS = {
    "QB": [("observed_career_lag_passing_yards", "observed_career_lag_pass_attempts")],
    "RB": [("team_rush_attempts_share", "team_rushing_yards_share")],
    "WR": [],
}


def painel_jogador(df_cox: pd.DataFrame) -> pd.DataFrame:
    """Agregacao por jogador, igual a do passo das curvas de Kaplan-Meier."""
    return (
        df_cox.groupby("player_id")
        .agg(
            duration=("stop", "max"),
            event=("event", "max"),
            entry=("start", "min"),
            age_entry=("age", "first"),
            draft_round=("draft_round", "first"),
            depth_entry=("depth_team", "first"),
        )
        .reset_index()
    )


def ic_mediana(kmf: KaplanMeierFitter) -> tuple[float | None, float | None, str]:
    """IC 95% da mediana a partir do IC da sobrevivencia (Greenwood exponencial).

    O limite inferior e o primeiro instante em que a banda INFERIOR cai abaixo de
    0,5; o superior e o primeiro em que a banda SUPERIOR cai abaixo de 0,5.
    """
    banda = kmf.confidence_interval_
    if banda is None or banda.empty:
        return None, None, "indisponivel"
    tempos = banda.index.to_numpy(dtype=float)
    # lifelines: primeira coluna e a banda inferior, segunda a superior
    inferior = banda.iloc[:, 0].to_numpy(dtype=float)
    superior = banda.iloc[:, 1].to_numpy(dtype=float)
    metodo = "Greenwood exponencial (lifelines confidence_interval_) invertido para a mediana"

    def primeiro(mascara: np.ndarray) -> float | None:
        idx = np.flatnonzero(mascara)
        return float(tempos[idx[0]]) if len(idx) else None

    return primeiro(inferior <= 0.5), primeiro(superior <= 0.5), metodo


def detalhe_colinearidade(df_fit: pd.DataFrame, covs: list[str], threshold: float = 0.85) -> list[dict]:
    """Repete a remocao gulosa registrando, por variavel removida, os pares acima do limiar.

    A regra do pipeline nao e "reter a de maior relevancia preditiva": remove-se,
    a cada passo, a covariavel com o MAIOR NUMERO de pares acima do limiar. O
    relatorio existe para que a escrita possa descrever a regra que o codigo
    executa, e nao a que se gostaria de executar.
    """
    substantivas = [c for c in covs if not c.endswith("_missing")]
    numericas = df_fit[substantivas].apply(pd.to_numeric, errors="coerce")
    matriz = numericas.corr().abs()
    np.fill_diagonal(matriz.values, 0)
    removidas: list[dict] = []
    restantes = set(substantivas)
    while True:
        contagem = {
            c: int((matriz.loc[c, sorted(restantes)] > threshold).sum()) for c in restantes
        }
        if max(contagem.values(), default=0) == 0:
            break
        alvo = max(sorted(contagem), key=contagem.get)
        pares = [
            {"com": outro, "r": round(float(matriz.loc[alvo, outro]), 6)}
            for outro in sorted(restantes)
            if outro != alvo and matriz.loc[alvo, outro] > threshold
        ]
        removidas.append(
            {"variavel": alvo, "n_pares_acima": contagem[alvo], "pares": pares}
        )
        restantes.remove(alvo)
    return removidas


def protocolo_inferencial(linhas: list[dict]) -> dict:
    """Constantes metodologicas da inferencia, conferidas contra os proprios scripts.

    Cada valor declarado aqui e procurado como literal no arquivo que o executa.
    Divergencia entre o declarado e o encontrado vira `confere: false`, de modo
    que o artefato acusa deriva em vez de documentar uma intencao.
    """
    declarado = [
        ("cox.penalizer", 0.0, "pipeline/11_cox_tdc_qb.py", "penalizer=0.0"),
        ("cox.threshold_p", 0.05, "pipeline/11_cox_tdc_qb.py", "threshold = 0.05 if n_events >= 50 else 0.10"),
        ("cox.threshold_p_fallback", 0.10, "pipeline/11_cox_tdc_qb.py", "else 0.10"),
        ("cox.max_vars", "max(2, piso(E/10))", "pipeline/11_cox_tdc_qb.py", "max_vars = max(2, int(n_events / 10))"),
        ("cox.ordem_colinearidade", "identificaveis antes do filtro |r|>0,85", "pipeline/11_cox_tdc_qb.py", "covariaveis_identificaveis_cox(df_fit, cov_cols)\\n'\n    'cov_cols_filtered = filter_collinear_covariates"),
        ("colinearidade.limiar", 0.85, "Inferência/shared/survival_utils.py", "threshold=0.85"),
        ("taxa_lag.denominador_minimo", 50, "pipeline/04_prep_cox_qb.py", "den_lag] > 50"),
        ("aalen.triagem", "survival::aareg univariado; sem sorteio nem semente", "Inferência/Modelagem de Aalen/shared/run_aalen_univariate_screening.R", "fit <- survival::aareg(as.formula("),
        ("aalen.metodo", "survival::aareg; empates nativos; sem sorteio", "Inferência/Modelagem de Aalen/shared/run_aalen_tied_inference.R", "survival::aareg(formula, data = data, cluster = player_id, dfbeta = TRUE, taper = 1)"),
        ("aalen.janela", "janela de suporte por covariavel (posto completo)", "Inferência/Modelagem de Aalen/shared/run_aalen_tied_inference.R", "select_supported_window(df, covariates)"),
        ("cox_aalen.metodo", "timereg::cox.aalen (Scheike & Zhang 2002)", "Inferência/Modelagem Cox-Aalen/shared/run_cox_aalen_inference.R", "do.call(cox.aalen, c(list("),
        ("cox_aalen.n_sim", 1000, "Inferência/Modelagem Cox-Aalen/shared/run_cox_aalen_inference.R", "n.sim = 1000"),
        ("cox_aalen.desempate", "dentro da temporada (stop - 0,5; stop)", "Inferência/Modelagem Cox-Aalen/shared/run_cox_aalen_inference.R", "desempata_saidas(df)"),
        ("cox_aalen.semente_desempate", 20241012, "Inferência/Modelagem Cox-Aalen/shared/run_cox_aalen_inference.R", "set.seed(20241012L)"),
        ("predicao.seed", 42, "documentos/estudo_modelos_finais/run_oficial.py", "SEED = 42"),
        ("km.entrada_tardia", "entry = start", "pipeline/08_curvas_km_qb.py", "entry=df_player['entry']"),
    ]
    verificacao = []
    for campo, valor, arquivo, literal in declarado:
        caminho = ROOT / arquivo
        texto = caminho.read_text(encoding="utf-8", errors="replace") if caminho.exists() else ""
        verificacao.append(
            {
                "campo": campo,
                "valor": valor,
                "arquivo": arquivo,
                "literal_procurado": literal,
                "confere": bool(texto) and literal in texto,
            }
        )
    divergentes = [v["campo"] for v in verificacao if not v["confere"]]
    for v in verificacao:
        linhas.append(
            {
                "posicao": "todas",
                "metrica": f"protocolo_{v['campo']}",
                "valor": v["valor"] if not isinstance(v["valor"], bool) else int(v["valor"]),
            }
        )
    return {
        "gerado_por": "pipeline/50_estatisticas_eda.py",
        "constantes": {v["campo"]: v["valor"] for v in verificacao},
        "verificacao": verificacao,
        "divergencias": divergentes,
    }


def main() -> int:
    EDA_RES.mkdir(parents=True, exist_ok=True)
    linhas: list[dict] = []
    testes: list[dict] = []
    preenchimento: list[dict] = []
    funil: dict[str, dict] = {}

    def anota(posicao: str, metrica: str, valor) -> None:
        linhas.append({"posicao": posicao, "metrica": metrica, "valor": valor})

    for posicao in POSICOES:
        # ---------------------------------------------------------- EDA
        df = load_clean_position_data(ROOT, posicao)
        for col in ("years_exp", "draft_round", "age", "games_missed", "season"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["draft_round"] = df["draft_round"].fillna(8)
        carreira = df.groupby("player_id")["season"].nunique()

        anota(posicao, "eda_jogadores", int(df["player_id"].nunique()))
        anota(posicao, "eda_registros_jogador_temporada", int(len(df)))
        anota(posicao, "carreira_mediana_temporadas", float(carreira.median()))
        anota(posicao, "carreira_media_temporadas", round(float(carreira.mean()), 4))
        anota(posicao, "carreira_p25_temporadas", float(carreira.quantile(0.25)))
        anota(posicao, "carreira_p75_temporadas", float(carreira.quantile(0.75)))
        anota(posicao, "carreira_pct_ate_3_temporadas", round(float((carreira <= 3).mean() * 100), 4))
        anota(posicao, "carreira_max_temporadas", int(carreira.max()))

        # Estreia = temporada do draft (years_exp == 0; season == draft_year em 100% dos casos).
        debut = df[df["years_exp"] == 0]["age"].dropna()
        anota(posicao, "estreia_idade_mediana", round(float(debut.median()), 4) if len(debut) else None)
        anota(posicao, "estreia_idade_media", round(float(debut.mean()), 4) if len(debut) else None)
        anota(posicao, "estreia_n", int(len(debut)))

        # Kruskal-Wallis da longevidade entre estratos de draft (mesma definicao
        # do painel esquerdo da figura de longevidade por faixas de draft).
        jogador = df.drop_duplicates("player_id").copy()
        jogador["career_length"] = jogador["player_id"].map(carreira)
        jogador["draft_group"] = jogador["draft_round"].apply(
            lambda r: "Round 1" if r == 1 else ("Round 2" if r == 2 else (
                f"Rounds {int(r)}" if r < 8 else "UDFA"))
        )
        grupos = [g["career_length"].dropna().values for _, g in jogador.groupby("draft_group")]
        grupos = [g for g in grupos if len(g) > 1]
        if len(grupos) >= 2:
            h, p = stats.kruskal(*grupos)
            testes.append(
                {
                    "posicao": posicao,
                    "teste": "kruskal_wallis",
                    "comparacao": "longevidade por estrato de draft",
                    "estatistica": round(float(h), 4),
                    "p": float(p),
                    "n": int(sum(len(g) for g in grupos)),
                }
            )

        # Mann-Whitney: longevidade entre <=4 e >4 jogos perdidos no maximo da carreira.
        gm_max = df.groupby("player_id")["games_missed"].max()
        jogador["gm_max"] = jogador["player_id"].map(gm_max)
        alto = jogador[jogador["gm_max"] > 4]["career_length"].dropna()
        baixo = jogador[jogador["gm_max"] <= 4]["career_length"].dropna()
        if len(alto) > 5 and len(baixo) > 5:
            u, p = stats.mannwhitneyu(alto, baixo, alternative="two-sided")
            testes.append(
                {
                    "posicao": posicao,
                    "teste": "mann_whitney",
                    "comparacao": "longevidade: >4 vs <=4 jogos perdidos",
                    "estatistica": round(float(u), 4),
                    "p": float(p),
                    "n": int(len(alto) + len(baixo)),
                }
            )

        # -------------------------------------------------- painel e Kaplan-Meier
        df_cox = pd.read_pickle(PREP_RES / f"df_cox_{posicao}.pkl")
        df_player = painel_jogador(df_cox)
        kmf = KaplanMeierFitter()
        kmf.fit(df_player["duration"], event_observed=df_player["event"], entry=df_player["entry"])
        mediana = float(kmf.median_survival_time_)
        inf, sup, metodo = ic_mediana(kmf)
        anota(posicao, "km_mediana_anos", mediana)
        anota(posicao, "km_mediana_ic95_inferior", inf)
        anota(posicao, "km_mediana_ic95_superior", sup)
        anota(posicao, "km_mediana_metodo_ic", metodo)
        anota(posicao, "km_n_jogadores", int(len(df_player)))
        anota(posicao, "km_n_eventos", int(df_player["event"].sum()))
        anota(posicao, "painel_intervalos", int(len(df_cox)))
        anota(
            posicao,
            "painel_censura_jogador_pct",
            round(100 * (1 - float(df_player["event"].mean())), 4),
        )
        anota(
            posicao,
            "pre_window_veterano_pct",
            round(100 * float(df_cox.groupby("player_id")["pre_window_veteran"].max().mean()), 4),
        )

        # Log-rank por estrato de draft, com entrada tardia (o mesmo risco das curvas KM).
        tier = df_player["draft_round"].map(lambda r: "elite" if r <= 2 else ("mid" if r <= 5 else "late"))
        res = logrank_entrada_tardia(df_player["duration"], tier, df_player["event"], entry=df_player["entry"])
        testes.append(
            {
                "posicao": posicao,
                "teste": "logrank_draft_tiers",
                "comparacao": "elite (1-2) vs mid (3-5) vs late (6+)",
                "estatistica": round(float(res.test_statistic), 4),
                "p": float(res.p_value),
                "n": int(len(df_player)),
            }
        )

        # Log-rank por faixa etaria de entrada (bins da figura de idade).
        faixa = pd.cut(
            df_player["age_entry"], bins=[18, 22, 25, 40], labels=["<=22", "23-25", ">25"]
        )
        valido = df_player.assign(faixa=faixa).dropna(subset=["faixa"])
        if valido["faixa"].nunique() >= 2:
            res = logrank_entrada_tardia(
                valido["duration"], valido["faixa"], valido["event"], entry=valido["entry"]
            )
            testes.append(
                {
                    "posicao": posicao,
                    "teste": "logrank_faixa_etaria",
                    "comparacao": "<=22 vs 23-25 vs >25 anos",
                    "estatistica": round(float(res.test_statistic), 4),
                    "p": float(res.p_value),
                    "n": int(len(valido)),
                }
            )

        # Log-rank por grupo de profundidade no elenco (titular, reserva, profundo).
        com_prof = df_player.dropna(subset=["depth_entry"])
        if com_prof["depth_entry"].nunique() >= 2:
            grupo_prof = com_prof["depth_entry"].map(
                lambda v: "titular (1)" if v == 1 else ("reserva (2)" if v == 2 else "profundo (3+)")
            )
            res = logrank_entrada_tardia(
                com_prof["duration"], grupo_prof, com_prof["event"], entry=com_prof["entry"]
            )
            testes.append(
                {
                    "posicao": posicao,
                    "teste": "logrank_profundidade",
                    "comparacao": "titular (depth=1) vs reserva (2) vs profundo (3+)",
                    "estatistica": round(float(res.test_statistic), 4),
                    "p": float(res.p_value),
                    "n": int(len(com_prof)),
                }
            )

        # Log-rank Rounds 1-2 vs UDFA (anotacao da figura de KM por draft do EDA).
        r1 = df_player[df_player["draft_round"] <= 2]
        udfa = df_player[df_player["draft_round"] == 8]
        if len(r1) >= 5 and len(udfa) >= 5:
            dois = pd.concat([r1.assign(_g="r1r2"), udfa.assign(_g="udfa")])
            lr = logrank_entrada_tardia(dois["duration"], dois["_g"], dois["event"], entry=dois["entry"])
            testes.append(
                {
                    "posicao": posicao,
                    "teste": "logrank_r1r2_vs_udfa",
                    "comparacao": "rounds 1-2 vs nao draftados",
                    "estatistica": round(float(lr.test_statistic), 4),
                    "p": float(lr.p_value),
                    "n": int(len(r1) + len(udfa)),
                }
            )

        # ------------------------------------------------- colinearidade e funil
        cov_cols = [c for c in df_cox.columns if c not in ID_COLS + TIME_COLS]
        cov_cols = [c for c in cov_cols if c != "draft_round"]
        candidatas = list(cov_cols)
        stats_imp = fit_imputation_stats(df_cox, cov_cols)
        df_model = apply_imputation_stats(df_cox, cov_cols, stats_imp)
        escalador = StandardScaler()
        df_model[cov_cols] = escalador.fit_transform(df_model[cov_cols])
        df_fit = df_model[["player_id", "start", "stop", "event"] + cov_cols].copy()
        # Mesma cadeia do Cox (pipeline/11, passos 8 e 9): identificabilidade,
        # colinearidade 0,85, reinclusao das _missing mantidas e nova
        # identificabilidade. O funil cita cada etapa.
        identificaveis = covariaveis_identificaveis_cox(df_fit, cov_cols, verbose=False)
        pos_colinearidade = filter_collinear_covariates(df_fit, identificaveis, threshold=0.85, verbose=False)
        pos_colinearidade = force_missing_flags(pos_colinearidade, df_fit.columns.tolist())
        mantidas = covariaveis_identificaveis_cox(df_fit, pos_colinearidade, verbose=False)
        detalhe = detalhe_colinearidade(df_fit, identificaveis, threshold=0.85)

        numericas = df_cox[[c for c in df_cox.columns if c not in ID_COLS + TIME_COLS]].apply(
            pd.to_numeric, errors="coerce"
        )
        matriz = numericas.corr().abs()
        pares = []
        for a, b in PARES_CITADOS.get(posicao, []):
            if a in matriz.columns and b in matriz.columns:
                pares.append({"a": a, "b": b, "r": round(float(matriz.loc[a, b]), 6)})
        anota(posicao, "funil_candidatas", len(candidatas))
        anota(posicao, "funil_sem_nao_identificaveis", len(identificaveis))
        anota(posicao, "funil_apos_colinearidade", len(pos_colinearidade))
        anota(posicao, "funil_saturado_cox", len(mantidas))
        for par in pares:
            anota(posicao, f"correlacao_{par['a']}_x_{par['b']}", par["r"])

        funil[posicao] = {
            "limiar_colinearidade": 0.85,
            "candidatas": candidatas,
            "n_candidatas": len(candidatas),
            "n_sem_nao_identificaveis": len(identificaveis),
            "n_apos_colinearidade": len(pos_colinearidade),
            "removidas_por_colinearidade": sorted(set(identificaveis) - set(pos_colinearidade)),
            "removidas_detalhe": detalhe,
            "mantidas": sorted(mantidas),
            "n_mantidas": len(mantidas),
            "pares_citados": pares,
            "gerado_por": "pipeline/50_estatisticas_eda.py",
        }

        # Preenchimento das taxas derivadas por periodo de carreira: a escrita
        # afirma que no primeiro periodo elas nao estao completas.
        taxas = [c for c in df_cox.columns if c.startswith("observed_career_lag_")
                 and not c.endswith("_missing")]
        taxas = [c for c in taxas if any(x in c for x in (
            "comp_pct", "ypa", "int_pct", "td_pct", "ypc", "yptarget", "ypr",
            "catch_rate", "adot_acc", "rush_td_pct", "yptouch", "reception_rate"))]
        if taxas:
            for periodo, grupo in df_cox.groupby("start"):
                preenchimento.append(
                    {
                        "posicao": posicao,
                        "periodo": int(periodo),
                        "n_linhas": int(len(grupo)),
                        "n_taxas": len(taxas),
                        "pct_preenchido_taxas": round(
                            100 * float(grupo[taxas].notna().to_numpy().mean()), 4
                        ),
                    }
                )
        print(
            f"{posicao}: candidatas={len(candidatas)} sem_nao_identificaveis={len(identificaveis)} "
            f"apos_colinearidade={len(pos_colinearidade)} saturado_cox={len(mantidas)} "
            f"km_mediana={mediana:.1f} carreira_mediana={carreira.median():.0f}",
            flush=True,
        )

    protocolo = protocolo_inferencial(linhas)
    pd.DataFrame(linhas).to_csv(EDA_RES / "estatisticas_eda.csv", index=False)
    pd.DataFrame(testes).to_csv(EDA_RES / "testes_eda.csv", index=False)
    pd.DataFrame(preenchimento).to_csv(
        PREP_RES / "preenchimento_por_periodo.csv", index=False
    )
    (PREP_RES / "funil_covariaveis.json").write_text(
        json.dumps(funil, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (ROOT / "Inferência" / "protocolo_inferencial.json").write_text(
        json.dumps(protocolo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"gravado: {(EDA_RES / 'estatisticas_eda.csv').relative_to(ROOT)}")
    print(f"gravado: {(EDA_RES / 'testes_eda.csv').relative_to(ROOT)}")
    print(f"gravado: {(PREP_RES / 'funil_covariaveis.json').relative_to(ROOT)}")
    print(f"gravado: {(PREP_RES / 'preenchimento_por_periodo.csv').relative_to(ROOT)}")
    print(f"gravado: Inferência/protocolo_inferencial.json")
    if protocolo["divergencias"]:
        print(f"ATENCAO: protocolo divergente em {protocolo['divergencias']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
