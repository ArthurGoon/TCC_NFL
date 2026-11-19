#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Persiste o lastro numerico das figuras do EDA e a rastreabilidade das figuras.

Problema que este passo resolve
-------------------------------
As figuras do EDA (`pipeline/01_eda_qb.py`, `02_eda_rb.py`, `03_eda_wr.py`) sao
desenhadas a partir de series calculadas em memoria e nunca gravadas. Sem a
serie gravada nao existe artefato do pipeline que comprove os numeros lidos na
figura: a figura deixa de ser auditavel contra a escrita.

Este passo recalcula, com as MESMAS receitas do EDA (mesmo loader
`load_clean_position_data`, mesmos filtros, mesmos rotulos), as series plotadas
e grava:

    Analise Exploratoria de Dados/resultados/series_figuras/
        fig_dist_carreira_{POS}.csv          histograma de tamanho de carreira
        fig_draft_media_{POS}.csv            barras: media de temporadas por faixa de draft
        fig_draft_boxplot_{POS}.csv          boxplot de draft_ovr por longevidade
        fig_games_missed_serie_{POS}.csv     mediana de jogos perdidos por anos_exp
        fig_games_missed_grupos_{POS}.csv    grupos <=4 / >4 jogos (boxplot direito)
        fig_depth_contagem_{POS}.csv         frequencia de depth_team
        fig_depth_por_exp_{POS}.csv          media de depth_team por anos_exp
        fig_corr_matrix_{POS}.csv            matriz de correlacao efetivamente plotada
        fig_career_lag_evo_{POS}.csv         media/mediana/dp por anos_exp (variaveis de carreira)
        fig_testes.csv                       Kruskal-Wallis (draft) e Mann-Whitney (jogos perdidos)

Rastreabilidade
---------------
Le as figuras citadas na escrita compilada (`.tex`, incluindo os macros
`\\thesisfigpanel{...}{...}{caminho}`), o produtor de cada figura no manifesto e
o lastro declarado em `LASTRO`. Grava:

    reprodutibilidade/rastreabilidade_figuras.csv   figura x produtor x lastro x existencia
    reprodutibilidade/figuras_nao_citadas.csv       figuras geradas e nao citadas (ruido)

O lastro e declarado, nao adivinhado: cada regra aponta para artefatos que o
proprio pipeline gera. A checagem de existencia acontece no pos-run
(`reprodutibilidade/reproducibility_audit.py`), porque num run parcial o lastro
de passos ainda nao executados legitimamente nao existe.

Uso:
    python pipeline/52_series_figuras.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
for candidato in [ROOT, *ROOT.parents]:
    if (candidato / "Inferência" / "shared" / "survival_utils.py").exists():
        ROOT = candidato
        break
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "Inferência" / "shared"))

from survival_utils import load_clean_position_data  # noqa: E402

POSICOES = ("QB", "RB", "WR")
EDA_RES = ROOT / "Análise Exploratória de Dados" / "resultados"
SERIES_DIR = EDA_RES / "series_figuras"
MANIFESTO = ROOT / "reprodutibilidade" / "REPRODUCIBILITY_MANIFEST.json"
ESCRITA = ROOT / "Escrita Final" / "texto"
RASTREABILIDADE = ROOT / "reprodutibilidade" / "rastreabilidade_figuras.csv"
NAO_CITADAS = ROOT / "reprodutibilidade" / "figuras_nao_citadas.csv"

# Figuras ilustrativas: nao plotam dado nenhum, logo nao exigem lastro numerico.
ILUSTRATIVAS = {
    "figuras/campo_posicoes_nfl.pdf": (
        "esquema de posicoes em campo (desenho conceitual, sem dado amostrado)"
    ),
}

# Lastro declarado por familia de figura. A primeira regra que casa decide.
# Cada entrada: (regex sobre o caminho citado, [artefatos de lastro], observacao)
LASTRO: list[tuple[str, list[str], str]] = [
    (
        r"figuras/dist_carreira_",
        ["Análise Exploratória de Dados/resultados/series_figuras/fig_dist_carreira_{POS}.csv",
         "Análise Exploratória de Dados/resultados/estatisticas_eda.csv"],
        "histograma de carreira + mediana/media/P25/P75/percentual ate 3 temporadas",
    ),
    (
        r"figuras/draft_",
        ["Análise Exploratória de Dados/resultados/series_figuras/fig_draft_media_{POS}.csv",
         "Análise Exploratória de Dados/resultados/series_figuras/fig_draft_boxplot_{POS}.csv",
         "Análise Exploratória de Dados/resultados/series_figuras/fig_testes.csv"],
        "barras de media por faixa de draft, boxplot de draft_ovr e Kruskal-Wallis",
    ),
    (
        r"figuras/games_missed_",
        ["Análise Exploratória de Dados/resultados/series_figuras/fig_games_missed_serie_{POS}.csv",
         "Análise Exploratória de Dados/resultados/series_figuras/fig_games_missed_grupos_{POS}.csv",
         "Análise Exploratória de Dados/resultados/series_figuras/fig_testes.csv"],
        "mediana de jogos perdidos por experiencia, grupos <=4/>4 e Mann-Whitney",
    ),
    (
        r"figuras/depth_",
        ["Análise Exploratória de Dados/resultados/series_figuras/fig_depth_contagem_{POS}.csv",
         "Análise Exploratória de Dados/resultados/series_figuras/fig_depth_por_exp_{POS}.csv"],
        "frequencia de depth_team e media de depth_team por anos de experiencia",
    ),
    (
        r"figuras/corr_matrix_",
        ["Análise Exploratória de Dados/resultados/series_figuras/fig_corr_matrix_{POS}.csv"],
        "matriz de Pearson jogador-nivel exatamente como plotada (variaveis mantidas)",
    ),
    (
        r"figuras/career_lag_evo_",
        ["Análise Exploratória de Dados/resultados/series_figuras/fig_career_lag_evo_{POS}.csv"],
        "media/mediana/dp por anos de experiencia das variaveis de carreira (QB)",
    ),
    (
        r"figuras/km_",
        ["Análise Exploratória de Dados/resultados/estatisticas_eda.csv",
         "Análise Exploratória de Dados/resultados/testes_eda.csv"],
        "medianas de KM com IC95 e testes log-rank de draft, idade e profundidade",
    ),
    (
        r"figuras/schoenfeld_",
        ["Inferência/Modelagem de COX/resultados/schoenfeld_{POS}.csv",
         "Inferência/Modelagem de COX/resultados/final_covs_{POS}.json"],
        "residuos de Schoenfeld por covariavel do modelo final congelado",
    ),
    (
        r"figuras/forest_plot_",
        ["Inferência/Modelagem de COX/resultados/cox_hr_{POS}.csv",
         "Inferência/Modelagem de COX/resultados/lr_test_{POS}.csv"],
        "HR, IC95 e p por covariavel do modelo Cox final",
    ),
    (
        r"figuras/aalen_termos_violacao",
        ["documentos/estudo_leitura_aplicada/resultados/aalen_termos_violacao.csv",
         "Inferência/Modelagem de Aalen/resultados/aalen_cum_coefs_RB.csv",
         "Inferência/Modelagem de Aalen/resultados/aalen_cum_coefs_WR.csv",
         "Inferência/Modelagem de Aalen/resultados/aalen_summary_RB.json"],
        "curvas B(t) do Aalen final para os termos que violam o Schoenfeld (RB e WR), com IC pontual de 95%",
    ),
    (
        r"figuras/aalen_cum_coefs_",
        ["Inferência/Modelagem de Aalen/resultados/aalen_cum_coefs_{POS}.csv",
         "Inferência/Modelagem de Aalen/resultados/aalen_robvar_cum_{POS}.csv",
         "Inferência/Modelagem de Aalen/resultados/aalen_summary_{POS}.json"],
        "coeficientes acumulados B(t) com bandas de variancia robusta",
    ),
    (
        r"figuras/comparacao_cox_aalen_",
        ["Inferência/Modelagem de Aalen/resultados/comparacao_cox_aalen_{POS}.json",
         "Inferência/Modelagem de Aalen/resultados/aalen_summary_{POS}.json"],
        "comparacao descritiva entre Cox e Aalen no mesmo conjunto congelado",
    ),
    (
        r"tgb_cindex_walkforward|tgb_acerto_horizontes|tgb_previsto_vs_observado"
        r"|tgb_casos_individuais|tgb_dist_erro_abs|tgb_cauda_erro|tgb_importancia_nativa",
        ["documentos/estudo_modelos_finais/resultados/pred_oof_walkforward_2018_2021.csv",
         "documentos/estudo_modelos_finais/resultados/resumo_walkforward_2018_2021.csv",
         "documentos/estudo_modelos_finais/resultados/auditoria/acerto_horizontes.csv",
         "documentos/estudo_leitura_preditiva/resultados/perm_importance_media.csv",
         "documentos/pacote_preditivo_tgb/figuras/importancia_familias.csv"],
        "predicoes out-of-fold do walk-forward e importancias nativas/permutacao",
    ),
    (
        r"tgb_lista_2024",
        ["documentos/estudo_leitura_aplicada/resultados/lista_2024_extremos.csv",
         "documentos/estudo_modelos_finais/resultados/predicao_2024_modelos.csv"],
        "previsoes 2024 do modelo principal por posicao (extremos da lista), com idade e experiencia cruas de 2024",
    ),
    (
        r"tgb_lista_quintis",
        ["documentos/estudo_leitura_aplicada/resultados/lista_2024_quintis.csv",
         "documentos/estudo_modelos_finais/resultados/predicao_2024_modelos.csv"],
        "mediana de covariaveis de 2024 por quintil de risco previsto pelo modelo principal",
    ),
    (
        r"tgb_landmark_quintis",
        ["documentos/estudo_leitura_aplicada/resultados/landmark_quintis.csv"],
        "saida prevista e observada em w anos por quintil de risco (landmarking do modelo principal)",
    ),
    (
        r"tgb_casos_leitura",
        ["documentos/estudo_leitura_aplicada/resultados/casos_auge.csv"],
        "casos escolhidos por regra (casos_auge.csv): participacao, passer rating, previsoes por modelo e desfecho",
    ),
    (
        r"tgb_calibracao_quintil",
        ["documentos/estudo_diagnostico_erro/resultados/calibracao_quintil.csv",
         "documentos/estudo_diagnostico_erro/resultados/pontos_figura_calibracao.json"],
        "calibracao por quintil de risco previsto: S(3) prevista media contra Kaplan-Meier do OOF (folds 2018-2020)",
    ),
    (
        r"c_walkforward_folds",
        ["documentos/estudo_modelos_finais/resultados/metricas_walkforward_folds.csv",
         "documentos/estudo_modelos_finais/resultados/resumo_walkforward_2018_2021.csv",
         "documentos/estudo_ic_preditivo/resultados/ic_metricas.csv"],
        "C de Harrell por origem (2018 a 2021), por posicao e modelo, com a media e o IC 95% do walk-forward",
    ),
]


# --------------------------------------------------------------------------
# Parte 1 — series do EDA
# --------------------------------------------------------------------------
def _boxplot_stats(valores: np.ndarray) -> dict | None:
    """Quartis e whiskers como o matplotlib.boxplot desenha (1.5 * IQR)."""
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    iqr = q3 - q1
    interno = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
    return {
        "n": int(v.size),
        "q1": q1,
        "mediana": med,
        "q3": q3,
        "whislo": float(interno.min()) if interno.size else q1,
        "whishi": float(interno.max()) if interno.size else q3,
        "minimo": float(v.min()),
        "maximo": float(v.max()),
    }


def series_eda(df: pd.DataFrame, posicao: str, testes: list[dict]) -> list[str]:
    """Grava as series plotadas nas figuras do EDA. Devolve os arquivos escritos."""
    escritos: list[str] = []
    df = df.copy()
    # O EDA cria career_length na primeira celula (transform nunique por jogador).
    df["career_length"] = df.groupby("player_id")["season"].transform("nunique")

    def grava(nome: str, tabela: pd.DataFrame) -> None:
        destino = SERIES_DIR / nome.format(POS=posicao)
        tabela.to_csv(destino, index=False)
        escritos.append(destino.name)

    # --- dist_carreira: histograma de temporadas por jogador -----------------
    cl = df.groupby("player_id")["season"].nunique()
    bins = list(range(1, int(cl.max()) + 2))
    hist = cl.value_counts().reindex(bins[:-1], fill_value=0).sort_index()
    grava(
        "fig_dist_carreira_{POS}.csv",
        pd.DataFrame({"temporadas": hist.index.astype(int), "n_jogadores": hist.values}),
    )

    # --- draft: barras por faixa e boxplot por longevidade -------------------
    cl_player = df.drop_duplicates("player_id").copy()
    cl_player["draft_round"] = cl_player["draft_round"].fillna(8)
    cl_player["draft_group"] = cl_player["draft_round"].apply(
        lambda x: f"Round {int(x)}" if x <= 7 else "Undrafted"
    )
    medias = cl_player.groupby("draft_group")["career_length"].agg(["mean", "count"])
    grava(
        "fig_draft_media_{POS}.csv",
        medias.reset_index().rename(
            columns={"draft_group": "faixa_draft", "mean": "media_temporadas", "count": "n_jogadores"}
        ),
    )
    cl_player["longevidade_grupo"] = pd.cut(
        cl_player["career_length"], bins=[0, 3, 7, 12, 99], labels=["≤3", "4–7", "8–12", ">12"]
    )
    linhas_bp = []
    for grupo in ["≤3", "4–7", "8–12", ">12"]:
        val = cl_player[cl_player["longevidade_grupo"] == grupo]["draft_ovr"].dropna().values
        st = _boxplot_stats(val)
        if st:
            linhas_bp.append({"longevidade": grupo, **st})
    grava("fig_draft_boxplot_{POS}.csv", pd.DataFrame(linhas_bp))

    grupos_kw = [g["career_length"].values for _, g in cl_player.groupby("draft_group")]
    grupos_kw = [g for g in grupos_kw if len(g) > 1]
    if len(grupos_kw) >= 2:
        h, p = stats.kruskal(*grupos_kw)
        testes.append({
            "posicao": posicao, "figura": f"draft_{posicao}", "teste": "kruskal_longevidade_por_faixa_draft",
            "estatistica": round(float(h), 6), "p_valor": float(p), "n_grupos": len(grupos_kw),
            "n_observacoes": int(sum(len(g) for g in grupos_kw)),
        })

    # --- games_missed: serie por experiencia + grupos <=4 / >4 jogos ---------
    if "games_missed" in df.columns:
        por_exp = df.groupby("years_exp")["games_missed"].agg(["median", "count"])
        por_exp = por_exp[por_exp["count"] >= 5]
        grava(
            "fig_games_missed_serie_{POS}.csv",
            por_exp.reset_index().rename(
                columns={"years_exp": "anos_experiencia", "median": "mediana_jogos_perdidos", "count": "n"}
            ),
        )
        cl_p = df.drop_duplicates("player_id").set_index("player_id")
        cl_p["gm_max"] = df.groupby("player_id")["games_missed"].max()
        alto = cl_p[cl_p["gm_max"] > 4]["career_length"].dropna()
        baixo = cl_p[cl_p["gm_max"] <= 4]["career_length"].dropna()
        linhas_gm = []
        for rotulo, serie in (("≤4 jogos", baixo), (">4 jogos", alto)):
            st = _boxplot_stats(serie.values)
            if st:
                linhas_gm.append({"grupo": rotulo, **st})
        grava("fig_games_missed_grupos_{POS}.csv", pd.DataFrame(linhas_gm))
        if len(alto) > 5 and len(baixo) > 5:
            u, p_mw = stats.mannwhitneyu(alto, baixo, alternative="two-sided")
            testes.append({
                "posicao": posicao, "figura": f"games_missed_{posicao}",
                "teste": "mannwhitney_longevidade_por_jogos_perdidos",
                "estatistica": round(float(u), 6), "p_valor": float(p_mw), "n_grupos": 2,
                "n_observacoes": int(len(alto) + len(baixo)),
            })

    # --- depth: frequencia e media por experiencia ---------------------------
    if "depth_team" in df.columns:
        contagem = df["depth_team"].value_counts().sort_index()
        grava(
            "fig_depth_contagem_{POS}.csv",
            pd.DataFrame({"depth_team": contagem.index.astype(int), "n_registros": contagem.values}),
        )
        dt_exp = df.groupby("years_exp")["depth_team"].agg(["mean", "count"])
        dt_exp = dt_exp[dt_exp["count"] >= 5]
        grava(
            "fig_depth_por_exp_{POS}.csv",
            dt_exp.reset_index().rename(
                columns={"years_exp": "anos_experiencia", "mean": "media_depth_team", "count": "n"}
            ),
        )

    # --- corr_matrix: matriz de Pearson exatamente como plotada --------------
    main_corr_vars = [
        "years_exp", "age", "height", "weight", "draft_ovr",
        "offense_pct", "depth_team", "games_missed", "career_length",
    ]
    position_specific = [
        "season_rushing_yards", "season_rush_attempts", "season_receptions",
        "season_receiving_yards", "season_pass_attempts", "season_passing_yards",
        "season_targets", "season_receiving_touchdown", "season_total_yards",
    ]
    main_corr_vars += [v for v in position_specific if v in df.columns]
    main_corr_vars = [v for v in main_corr_vars if v in df.columns]
    primeiro = ["years_exp", "age", "height", "weight", "draft_ovr", "career_length"]
    player_data = df.groupby("player_id")[main_corr_vars].agg(
        {v: ("first" if v in primeiro else "max") for v in main_corr_vars}
    )
    corr = player_data.corr(method="pearson")
    if "career_length" in corr.columns:
        top = corr["career_length"].drop("career_length").abs().sort_values(ascending=False).head(11).index.tolist()
        keep = top + ["career_length"]
        corr = corr.loc[keep, keep]
    corr.to_csv(SERIES_DIR / f"fig_corr_matrix_{posicao}.csv", index_label="variavel")
    escritos.append(f"fig_corr_matrix_{posicao}.csv")

    # --- career_lag_evo: media/mediana/dp por experiencia --------------------
    main_lags = [
        "career_passing_yards", "career_pass_touchdown", "career_pass_attempts",
        "career_rushing_yards",
    ]
    lag_linhas = []
    for var in main_lags:
        if var not in df.columns:
            continue
        evo = df.groupby("years_exp")[var].agg(["mean", "median", "std", "count"]).reset_index()
        for _, r in evo.iterrows():
            lag_linhas.append({
                "variavel": var, "anos_experiencia": int(r["years_exp"]),
                "media": r["mean"], "mediana": r["median"], "dp": r["std"], "n": int(r["count"]),
            })
    if lag_linhas:
        grava("fig_career_lag_evo_{POS}.csv", pd.DataFrame(lag_linhas))

    return escritos


# --------------------------------------------------------------------------
# Parte 2 — rastreabilidade das figuras citadas
# --------------------------------------------------------------------------
PADRAO_CMD = re.compile(r"\\(includegraphics|thesisfig[a-zA-Z]*)")
PADRAO_FIG = re.compile(r"\.(pdf|png|jpe?g|eps|svg)$", re.IGNORECASE)


def figuras_citadas(escrita: Path, recursivo: bool = False) -> dict[str, set[str]]:
    """Figuras efetivamente incluídas nos `.tex`.

    Le os `.tex` do diretorio (ou de toda a arvore, com `recursivo=True`) e
    percorre os grupos `{...}` que seguem um comando de figura. Necessario
    porque a escrita usa o macro de tres argumentos
    `\\thesisfigpanel{letra}{estilo}{caminho}`, em que o caminho e o terceiro
    grupo.
    """
    citadas: dict[str, set[str]] = {}
    arquivos = escrita.rglob("*.tex") if recursivo else escrita.glob("*.tex")
    for tex in sorted(arquivos):
        if ".git" in tex.parts:
            continue
        texto = tex.read_text(encoding="utf-8", errors="replace")
        for cmd in PADRAO_CMD.finditer(texto):
            i = cmd.end()
            while i < len(texto) and texto[i] == " ":
                i += 1
            while i < len(texto) and texto[i] in "[":  # argumentos opcionais
                prof = 1
                i += 1
                while i < len(texto) and prof:
                    prof += (texto[i] == "[") - (texto[i] == "]")
                    i += 1
                while i < len(texto) and texto[i] == " ":
                    i += 1
            while i < len(texto) and texto[i] == "{":
                prof, j = 0, i
                while j < len(texto):
                    if texto[j] == "{":
                        prof += 1
                    elif texto[j] == "}":
                        prof -= 1
                        if prof == 0:
                            break
                    j += 1
                valor = texto[i + 1:j].strip()
                if PADRAO_FIG.search(valor):
                    citadas.setdefault(valor, set()).add(tex.name)
                i = j + 1
                while i < len(texto) and texto[i] in " \n%":
                    i += 1
    return citadas


def corpo_dos_tex(raiz: Path) -> str:
    """Texto concatenado de todos os `.tex` da arvore (para busca por nome)."""
    return "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(raiz.rglob("*.tex"))
        if ".git" not in p.parts
    )


def lastro_de(caminho: str, posicao: str) -> tuple[list[str], str] | None:
    for padrao, artefatos, obs in LASTRO:
        if re.search(padrao, caminho):
            return [a.format(POS=posicao) for a in artefatos], obs
    return None


def posicao_da_figura(caminho: str) -> str | None:
    m = re.search(r"_(QB|RB|WR)", caminho)
    return m.group(1) if m else None


def main() -> int:
    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    testes: list[dict] = []
    escritos: list[str] = []

    for posicao in POSICOES:
        df = load_clean_position_data(ROOT, posicao)
        for col in ("years_exp", "draft_round", "age", "games_missed", "season"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["draft_round"] = df["draft_round"].fillna(8)
        escritos += series_eda(df, posicao, testes)

    pd.DataFrame(testes).to_csv(SERIES_DIR / "fig_testes.csv", index=False)
    escritos.append("fig_testes.csv")
    for nome in escritos:
        print(f"gravado: {(SERIES_DIR / nome).relative_to(ROOT)}")

    # ---------------------------------------------------------------- rastreio
    manifesto = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    produtores: dict[str, set[str]] = {}
    geradas: list[str] = []
    for artefato in manifesto["files"]:
        caminho = artefato["path"]
        if artefato.get("producer"):
            produtores.setdefault(caminho, set()).add(str(artefato["producer"]))
        if artefato.get("category") == "generated" and PADRAO_FIG.search(caminho):
            geradas.append(caminho)

    citadas = figuras_citadas(ESCRITA)
    linhas = []
    for caminho, onde in sorted(citadas.items()):
        posicao = posicao_da_figura(caminho)
        prod = sorted(produtores.get(caminho, set()))
        if not prod:  # o manifesto pode registrar so o caminho absoluto/exportado
            prod = sorted({p for c, ps in produtores.items() if c.endswith(caminho) for p in ps})
        if caminho in ILUSTRATIVAS:
            lastro, obs, tipo = [], ILUSTRATIVAS[caminho], "ilustrativa"
        else:
            achado = lastro_de(caminho, posicao or "")
            lastro, obs = achado if achado else ([], "")
            tipo = "dados"
        ausentes = [a for a in lastro if not (ROOT / a).exists()]
        linhas.append({
            "figura": caminho,
            "citada_em": ",".join(sorted(onde)),
            "posicao": posicao or "",
            "tipo": tipo,
            "produtor": ",".join(prod),
            "lastro_declarado": ";".join(lastro),
            "lastro_ausente_no_momento": ";".join(ausentes),
            "observacao": obs,
        })
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(RASTREABILIDADE, index=False)
    print(f"gravado: {RASTREABILIDADE.relative_to(ROOT)} ({len(tabela)} figuras citadas)")

    # ------------------------------------------------------- nao citadas (ruido)
    # Criterio: nenhum `.tex` do repositorio referencia a figura, nem pelo
    # caminho nem pelo nome do arquivo.
    citadas_repo = figuras_citadas(ROOT, recursivo=True)
    nomes_repo = {Path(c).name for c in citadas_repo}
    corpo = corpo_dos_tex(ROOT)
    nao_citadas = [
        c for c in sorted(geradas)
        if Path(c).name not in nomes_repo and Path(c).stem not in corpo
    ]
    pd.DataFrame({
        "figura_gerada_nao_citada": nao_citadas,
        "diretorio": [str(Path(c).parent) for c in nao_citadas],
    }).to_csv(NAO_CITADAS, index=False)
    print(f"gravado: {NAO_CITADAS.relative_to(ROOT)} ({len(nao_citadas)} figuras geradas"
          f" e nao citadas em nenhum .tex do repo)")

    sem_produtor = tabela[tabela["produtor"] == ""]
    sem_lastro = tabela[(tabela["tipo"] == "dados") & (tabela["lastro_declarado"] == "")]
    print(f"citadas com produtor: {len(tabela) - len(sem_produtor)}/{len(tabela)}")
    print(f"citadas com lastro declarado: {len(tabela) - len(sem_lastro) - len(ILUSTRATIVAS)}/"
          f"{len(tabela) - len(ILUSTRATIVAS)}")
    for _, r in sem_produtor.iterrows():
        print(f"AVISO sem produtor no manifesto: {r['figura']}")
    for _, r in sem_lastro.iterrows():
        print(f"AVISO sem lastro declarado: {r['figura']}")
    print(f"lastro ainda ausente em disco: {int(tabela['lastro_ausente_no_momento'].ne('').sum())} "
          f"(esperado em run parcial; o pos-run cobra)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
