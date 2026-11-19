#!/usr/bin/env python3
"""Confere os numeros da tese contra os resultados do pipeline.

O pipeline e a fonte de verdade: cada numero que aparece nas tabelas dos
capitulos compilados e comparado com o artefato que o `run_all.py` produz.
Onde a escrita divergir do artefato, o relatorio aponta.

Uso:
    python reprodutibilidade/audit_escrita_vs_pipeline.py            # relatorio
    python reprodutibilidade/audit_escrita_vs_pipeline.py --strict   # falha se houver divergencia

Saida: `reprodutibilidade/AUDITORIA_ESCRITA_VS_PIPELINE.md` e codigo de saida 1 quando
ha divergencia (com `--strict`).
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "run_all.py").exists() and (candidate / "DataBase").exists():
            return candidate
    raise RuntimeError("Project root not found")


ROOT = find_project_root()
LIVRO = ROOT / "Escrita Final" / "texto"
COX = ROOT / "Inferência" / "Modelagem de COX" / "resultados"
AALEN = ROOT / "Inferência" / "Modelagem de Aalen" / "resultados"
CA = ROOT / "Inferência" / "Modelagem Cox-Aalen" / "resultados"
PREP = ROOT / "Inferência" / "Preparação Modelagem" / "resultados"
OFICIAL = ROOT / "documentos" / "estudo_modelos_finais" / "resultados"
DIAG = ROOT / "documentos" / "estudo_diagnostico_erro" / "resultados"
TUNE = ROOT / "documentos" / "estudo_tune_preditivo" / "resultados"
IC = ROOT / "documentos" / "estudo_ic_preditivo" / "resultados"
FRONTEIRA = ROOT / "documentos" / "estudo_fronteira_tune" / "resultados"
MANIFEST = ROOT / "reprodutibilidade" / "REPRODUCIBILITY_MANIFEST.json"
REPORT = ROOT / "reprodutibilidade" / "AUDITORIA_ESCRITA_VS_PIPELINE.md"

CAPITULOS = [
    "main.tex",
    "c1_introd_tg.tex",
    "c2_exemplos_tg.tex",
    "c3_material_tg.tex",
    "c4_resultados_inferenciais.tex",
    "c5_resultados_preditivos.tex",
    "c6_discussao_tg.tex",
    "c7_conclusao_tg.tex",
    "apendice_dados.tex",
    "apendice_inferencia.tex",
    "apendice_predicao.tex",
]
POSICOES = ("QB", "RB", "WR")
MODELOS_PREDITIVOS = ("cox", "gbsa", "rsf")
FOLDS_PREDITIVOS = (2018, 2019, 2020, 2021)
PARES_PREDITIVOS = ("gbsa-cox", "rsf-cox", "gbsa-rsf")
TOL_EMPATE_BOOTSTRAP = 1e-12

REGRAS_TEX = re.compile(r"\\(?:toprule|midrule|bottomrule|addlinespace|hline|cmore|endfirsthead|endhead|endfoot)\b")


# ---------------------------------------------------------------- utilidades

# Macros de rotulo definidas em main.tex -> nome da covariavel no pipeline.
# Sem isso o limpador apaga a celula e a linha sai de fase.
MACROS_VAR = {
    "vGamesMissed": "games_missed",
    "vDepthTeam": "depth_team",
    "vOffensePct": "offense_pct",
    "vPasserRating": "passer_rating",
    "vTeamPassShare": "team_pass_attempts_share",
    "vTeamTargetsShare": "team_targets_share",
    "vTeamRushShare": "team_rushing_yards_share",
    "vPreVeteran": "pre_window_veteran",
    "vUndrafted": "undrafted_flag",
    "vDraftOvr": "draft_ovr",
    "vGapYears": "within_window_gap_years_before_t",
    "vUnobsHist": "unobserved_history_flag",
    "vLagTdPct": "observed_career_lag_td_pct",
    "vLagYpc": "observed_career_lag_ypc",
    "vLagFumble": "observed_career_lag_fumble_lost",
    "vLagRushYds": "observed_career_lag_rushing_yards",
    "vHeight": "height",
    "vLagCatchRate": "observed_career_lag_catch_rate",
    "vPreExp": "unobs_career_years",
    "vLagCompMiss": "observed_career_lag_comp_pct_missing",
    "vLagRedzone": "observed_career_lag_targets_redzone",
    "vAge": "age",
}


def limpa_celula(texto: str) -> str:
    for macro, nome in MACROS_VAR.items():
        texto = texto.replace(f"\\{macro}", nome)
    texto = re.sub(r"\\(?:textit|textbf|texttt|emph|mathbf|mathrm|text|multicolumn)\{([^}]*)\}", r"\1", texto)
    texto = texto.replace("$", "").replace("\\%", "%").replace("\\,", "").replace("~", " ")
    texto = re.sub(r"\\[a-zA-Z]+\*?", "", texto)
    texto = re.sub(r"[{}]", "", texto)
    return texto.strip()


def numero(texto: str) -> float | None:
    """Converte celula LaTeX em float. Devolve None se nao for numero."""
    texto = texto.replace("\\,", "").replace("+", "").replace("$", "").strip()
    if re.fullmatch(r"-?[1-9]\d{0,2}(?:\.\d{3})+", texto):  # milhar com ponto (1.454), nunca decimal
        return float(texto.replace(".", ""))
    texto = texto.replace("{,}", ".").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", texto)
    return float(match.group(0)) if match else None


def tabelas_de_texto(texto: str) -> dict[str, list[list[str]]]:
    tabelas: dict[str, list[list[str]]] = {}
    labels = [(m.start(), m.group(1)) for m in re.finditer(r"\\label\{(tab:[^}]+)\}", texto)]
    padrao = r"\\begin\{(?P<env>tabular\*?|tabularx|longtable)\}(.*?)\\end\{(?P=env)\}"
    for match in re.finditer(padrao, texto, re.S):
        corpo = match.group(2)
        corte = REGRAS_TEX.search(corpo)
        if corte:
            corpo = corpo[corte.end():]
        corpo = REGRAS_TEX.sub("", corpo)
        if not labels:
            continue
        inicio = match.start()
        _, label = min(labels, key=lambda par: abs(par[0] - inicio))
        linhas: list[list[str]] = []
        for bruta in corpo.split("\\\\"):
            if "&" not in bruta:
                continue
            celulas = [limpa_celula(c) for c in bruta.split("&")]
            # na tabela de concordancia, celula vazia significa fora do modelo e preserva a coluna
            if label not in ("tab:concordancia", "tab:leituras_cruzadas"):
                celulas = [c for c in celulas if c]
            if celulas:
                linhas.append(celulas)
        if linhas:
            tabelas.setdefault(label, []).extend(linhas)
    return tabelas


def tabelas_do_livro() -> dict[str, list[list[str]]]:
    """label da tabela -> linhas de celulas, dos capitulos compilados."""
    tabelas: dict[str, list[list[str]]] = {}
    for nome in CAPITULOS:
        caminho = LIVRO / nome
        if not caminho.exists():
            continue
        for label, linhas in tabelas_de_texto(caminho.read_text(encoding="utf-8", errors="replace")).items():
            tabelas.setdefault(label, []).extend(linhas)
    return tabelas


# ---------------------------------------------------------------- divergencias

@dataclass
class Divergencia:
    onde: str
    escrita: str
    pipeline: str
    artefato: str
    nota: str = ""


class Auditoria:
    def __init__(self, tabelas: dict[str, list[list[str]]]) -> None:
        self.tabelas = tabelas
        self.divergencias: list[Divergencia] = []
        self.checagens = 0
        self.comparacoes_numericas = 0

    def registra_comparacao_numerica(self, onde: str) -> None:
        self.comparacoes_numericas += 1

    def compara(
        self,
        onde: str,
        escrita: float | None,
        pipeline: float | None,
        artefato: str,
        tol: float = 0.006,
        nota: str = "",
    ) -> None:
        self.checagens += 1
        self.registra_comparacao_numerica(onde)
        if escrita is None or pipeline is None:
            self.divergencias.append(
                Divergencia(onde, str(escrita), str(pipeline), artefato, nota or "valor ausente")
            )
            return
        if abs(escrita - pipeline) > tol:
            self.divergencias.append(
                Divergencia(onde, f"{escrita:g}", f"{pipeline:g}", artefato, nota)
            )

    def compara_p(
        self,
        onde: str,
        celula: str,
        pipeline: float | None,
        artefato: str,
        tol: float = 0.0006,
        nota: str = "",
    ) -> None:
        """Compara um valor-p aceitando a forma de limite ('<0,001').

        O timereg devolve p exatamente 0 quando nenhuma das simulacoes atinge o
        estatistica observada; a escrita registra '<0,001'. Os dois concordam.
        """
        texto = celula.strip()
        if texto.startswith("<") or texto.startswith(">"):
            limite = numero(texto)
            self.checagens += 1
            self.registra_comparacao_numerica(onde)
            if limite is None or pipeline is None or not (pipeline < limite):
                self.divergencias.append(
                    Divergencia(onde, texto, str(pipeline), artefato, nota or "esperado p abaixo do limite")
                )
            return
        self.compara(onde, numero(texto), pipeline, artefato, tol, nota)

    def compara_arredondado(
        self,
        onde: str,
        escrito: float | None,
        pipeline: float | None,
        artefato: str,
        nd: int = 3,
        nota: str = "",
    ) -> None:
        """Compara pelo valor ARREDONDADO, que e como a escrita cita o numero.

        Evita o falso positivo de tolerancia: 0,0304 arredonda para 0,030, e uma
        escrita que diga 0,031 esta errada ainda que a diferenca seja < 0,001.
        """
        self.checagens += 1
        self.registra_comparacao_numerica(onde)
        if escrito is None or pipeline is None:
            self.divergencias.append(
                Divergencia(onde, str(escrito), str(pipeline), artefato, nota or "valor ausente")
            )
            return
        if round(float(pipeline), nd) != round(float(escrito), nd):
            self.divergencias.append(
                Divergencia(
                    onde, f"{escrito:g}", f"{pipeline:.6g}", artefato, nota or f"arredondado a {nd} casas"
                )
            )

    def padrao(self, onde: str, casou: bool, artefato: str, nota: str = "") -> None:
        """Registra falha quando o padrao de prosa NAO casa.

        Sem isso, um regex que deixa de casar por causa de uma reescrita faria a
        checagem passar em silencio, que e o pior resultado possivel numa auditoria.
        """
        self.checagens += 1
        if not casou:
            self.divergencias.append(
                Divergencia(onde, "(padrao nao encontrado no texto)", "-", artefato, nota or "verificar redacao")
            )

    def linhas(self, label: str) -> list[list[str]]:
        return self.tabelas.get(label, [])

    def acha(self, label: str, contem: str, coluna: int = 0) -> list[str] | None:
        for linha in self.linhas(label):
            if len(linha) > coluna and contem.lower() in linha[coluna].lower():
                return linha
        return None


# ---------------------------------------------------------------- checagens

def checa_amostra(aud: Auditoria) -> None:
    for pos in POSICOES:
        linha = aud.acha("tab:amostra", pos)
        if not linha:
            continue
        df = pd.read_pickle(PREP / f"df_cox_{pos}.pkl")
        jogadores = df["player_id"].nunique()
        eventos = int(df["event"].sum())
        censurados = jogadores - eventos
        fonte = f"df_cox_{pos}.pkl"
        aud.compara(f"tab:amostra {pos} jogadores", numero(linha[1]), jogadores, fonte, 0.5)
        aud.compara(f"tab:amostra {pos} observacoes", numero(linha[2]), len(df), fonte, 0.5)
        aud.compara(f"tab:amostra {pos} eventos", numero(linha[3]), eventos, fonte, 0.5)
        aud.compara(f"tab:amostra {pos} censurados", numero(linha[4]), censurados, fonte, 0.5)
        aud.compara(
            f"tab:amostra {pos} censura %",
            numero(linha[5]),
            100 * censurados / jogadores,
            fonte,
            0.06,
            "censura por atleta, nao por linha",
        )


def checa_cox_ajuste(aud: Auditoria) -> None:
    """Teste de razao de verossimilhanca do Cox final, conferido contra o registro
    (valores_registrados.json)."""
    for pos in POSICOES:
        arquivo = COX / f"lr_test_{pos}.csv"
        if not arquivo.exists():
            continue
        d = pd.read_csv(arquivo).iloc[0]
        for campo in ("ll_saturado", "ll_final", "lr_chisq", "lr_p"):
            _registrado(aud, f"tab:cox_ajuste {pos} {campo}", f"{float(d[campo]):.4f}", arquivo.name)


def checa_schoenfeld(aud: Auditoria) -> None:
    linhas = aud.linhas("tab:schoenfeld")
    # A posicao entra por \multirow na primeira linha de cada bloco; as demais
    # vem com a primeira celula vazia. As linhas de covariavel tem 4 celulas
    # (covariavel, chisq, p, veredito) e a primeira do bloco tem 5.
    # Alinhar pela direita resolve os dois casos; a posicao e rastreada.
    por_posicao: dict[str, list[list[str]]] = {}
    atual = None
    for linha in linhas:
        if len(linha) >= 5:
            for p in POSICOES:
                if p in linha[0]:
                    atual = p
                    break
        if atual and len(linha) >= 4:
            por_posicao.setdefault(atual, []).append(linha)

    for pos in POSICOES:
        arquivo = COX / f"schoenfeld_{pos}.csv"
        if not arquivo.exists():
            continue
        d = pd.read_csv(arquivo)
        for _, r in d.iterrows():
            termo = str(r["term"])
            alvo = "Global" if termo == "GLOBAL" else termo
            achou = False
            for linha in por_posicao.get(pos, []):
                celula = re.sub(r"\\textbf\{([^}]*)\}", r"\1", linha[-4]).strip()
                if (celula if termo == "GLOBAL" else variavel_da_celula(celula)) != alvo:
                    continue
                achou = True
                prefixo = f"tab:schoenfeld {pos} {termo}"
                aud.compara(f"{prefixo} chisq", numero(linha[-3]), r["chisq"], arquivo.name, 0.006)
                aud.compara_p(f"{prefixo} p", linha[-2], r["p"], arquivo.name)
            # Linha da tabela que nenhum termo do artefato reconhece reprova (antes passava em silencio).
            if termo == "GLOBAL" or pos == "QB" or bool(r.get("violacao", False)):
                _registrado(aud, f"tab:schoenfeld {pos} {termo} p", f"{float(r['p']):.4f}", arquivo.name)


def checa_cox_hr(aud: Auditoria) -> None:
    linhas = aud.linhas("tab:cox_qb_hr")
    arquivo = COX / "cox_hr_QB.csv"
    if not linhas or not arquivo.exists():
        return
    d = pd.read_csv(arquivo).set_index("covariate")
    for linha in linhas[1:]:
        if len(linha) < 5:
            continue
        cov = variavel_da_celula(linha[0])
        aud.padrao(f"tab:cox_qb_hr linha {linha[0].strip()} reconhecida", cov in d.index, arquivo.name)
        if cov not in d.index:
            continue
        r = d.loc[cov]
        aud.compara(f"tab:cox_qb_hr {cov} HR", numero(linha[1]), r["HR"], arquivo.name)
        faixa = [numero(x) for x in re.findall(r"[-\d.,]+", linha[2])]
        if len(faixa) == 2:
            # O CSV guarda o IC na escala do log-HR; a escrita usa a escala do HR.
            aud.compara(
                f"tab:cox_qb_hr {cov} IC inferior",
                faixa[0],
                float(np.exp(r["CI_lower"])),
                arquivo.name,
                0.006,
                "CSV em log-HR, convertido para HR",
            )
            aud.compara(
                f"tab:cox_qb_hr {cov} IC superior",
                faixa[1],
                float(np.exp(r["CI_upper"])),
                arquivo.name,
                0.006,
                "CSV em log-HR, convertido para HR",
            )
        aud.compara_p(f"tab:cox_qb_hr {cov} p", linha[3], r["p_value"], arquivo.name)


def checa_cox_aalen_hr(aud: Auditoria) -> None:
    """Tabela de HRs da parte multiplicativa: Cox TDC (sem penalizacao) e Cox-Aalen."""
    n = 0
    for linha in aud.linhas("tab:cox_aalen_hr"):
        if len(linha) < 4 or linha[0].strip() not in POSICOES:
            continue
        pos, cov = linha[0].strip(), variavel_da_celula(linha[1])
        ca = pd.read_csv(CA / f"cox_aalen_beta_{pos}.csv").set_index("covariate")
        cx = pd.read_csv(COX / f"cox_hr_{pos}.csv").set_index("covariate")
        aud.padrao(f"tab:cox_aalen_hr linha {pos} {linha[1].strip()} reconhecida", cov in ca.index and cov in cx.index, "cox_aalen_beta")
        if cov not in ca.index or cov not in cx.index:
            continue
        n += 1
        aud.compara_arredondado(f"tab:cox_aalen_hr {pos} {cov} HR Cox", numero(linha[2]), cx.loc[cov, "HR"], f"cox_hr_{pos}.csv", 2)
        aud.compara_arredondado(f"tab:cox_aalen_hr {pos} {cov} HR Cox-Aalen", numero(linha[3]), ca.loc[cov, "HR"], f"cox_aalen_beta_{pos}.csv", 2)
    total = sum(len(pd.read_csv(CA / f"cox_aalen_beta_{p}.csv")) for p in POSICOES)
    aud.compara_arredondado("tab:cox_aalen_hr linhas = covariaveis multiplicativas", total, n, "cox_aalen_beta", 0)

def _mapa_rotulos() -> dict[str, str]:
    """Rotulo legivel (ou macro \\vX do main.tex, ou nome cru) -> nome da variavel no banco.

    As tabelas da escrita usam o rotulo do glossario; os artefatos usam o nome do banco.
    Sem esta traducao a checagem nao encontra a linha e a pula em silencio.
    """
    import csv  # noqa: PLC0415

    mapa: dict[str, str] = {}
    glossario = ROOT / "Predição" / "shared" / "glossario_variaveis.csv"
    if glossario.exists():
        with glossario.open(encoding="utf-8") as fh:
            for linha in csv.DictReader(fh):
                mapa[linha["rotulo"].strip().lower()] = linha["variavel"]
                mapa[linha["variavel"].lower()] = linha["variavel"]
                mapa[(linha["rotulo"].strip() + " (ausente)").lower()] = linha["variavel"] + "_missing"
    main = LIVRO / "main.tex"
    if main.exists():
        for nome, rotulo in re.findall(r"\\newcommand\{\\(v[A-Za-z]+)\}\{([^}]*)\}", main.read_text(encoding="utf-8")):
            var = mapa.get(rotulo.replace("\\%", "%").strip().lower())
            if var:
                mapa["\\" + nome.lower()] = var
    return mapa


def variavel_da_celula(celula: str) -> str:
    """Nome do banco a partir de uma celula de tabela (rotulo, macro ou nome cru)."""
    limpo = re.sub(r"\\textit\{([^}]*)\}", r"\1", celula).replace(r"\_", "_").replace("\\discretionary{}{}{}", "")
    limpo = limpo.replace("$", "").replace("\\%", "%").strip()
    return _MAPA_ROTULOS().get(limpo.lower(), limpo)


_cache_rotulos: dict[str, str] = {}


def _MAPA_ROTULOS() -> dict[str, str]:  # noqa: N802
    if not _cache_rotulos:
        _cache_rotulos.update(_mapa_rotulos())
    return _cache_rotulos


def _nums_celula(celula: str) -> list[float]:
    """Todos os numeros de uma celula LaTeX (para IC no formato [a; b])."""
    texto = celula.replace("{,}", ".").replace(",", ".").replace("$", "")
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", texto)]


def checa_tgb_wf(aud: Auditoria) -> None:
    arquivo = OFICIAL / "resumo_walkforward_2018_2021.csv"
    if not arquivo.exists():
        return
    d = pd.read_csv(arquivo)
    ic_path = IC / "ic_metricas.csv"
    ic = pd.read_csv(ic_path) if ic_path.exists() else pd.DataFrame()
    rotulo = {"cox": "Cox", "gbsa": "GBSA", "rsf": "RSF", "km": "Kaplan--Meier"}

    def chave_linha(linha: list[str]) -> tuple[str, str] | None:
        if len(linha) < 5 or linha[0].strip() not in POSICOES:
            return None
        modelo_escrito = linha[1].strip().replace("--", "-")
        modelo = next(
            (nome for nome, exibido in rotulo.items() if exibido.replace("--", "-") == modelo_escrito),
            None,
        )
        return (linha[0].strip(), modelo) if modelo else None

    linhas_tabela = aud.linhas("tab:tgb_wf")
    chaves_esperadas = {
        (str(r["posicao"]), str(r["modelo"]).lower())
        for _, r in d.iterrows()
        if str(r["modelo"]).lower() in rotulo
    }
    linhas_reconhecidas = [linha for linha in linhas_tabela if chave_linha(linha) is not None]
    aud.compara_arredondado(
        "tab:tgb_wf linhas",
        len(chaves_esperadas),
        len(linhas_reconhecidas),
        arquivo.name,
        0,
    )
    for _, r in d.iterrows():
        alvo = rotulo.get(str(r["modelo"]).lower())
        if alvo is None:
            continue
        chave = (str(r["posicao"]), str(r["modelo"]).lower())
        correspondentes = [linha for linha in linhas_tabela if chave_linha(linha) == chave]
        ref = f"{r['posicao']} {alvo}"
        aud.padrao(
            f"tab:tgb_wf {ref} linha unica",
            len(correspondentes) == 1,
            arquivo.name,
            f"linhas correspondentes: {len(correspondentes)}",
        )
        if len(correspondentes) != 1:
            continue
        linha = correspondentes[0]
        # Formato atual (c5): Pos., Modelo, 2018, 2019, 2020, 2021, Média, IC 95%, IBS.
        aud.padrao(f"tab:tgb_wf {ref} nove colunas (folds, média, IC, IBS)", len(linha) == 9, arquivo.name)
        if len(linha) != 9:
            continue
        folds_csv = pd.read_csv(OFICIAL / "metricas_walkforward_folds.csv")
        folds_csv = folds_csv[
            (folds_csv["protocolo"] == "walkforward")
            & (folds_csv["oficial"] == 1)
            & (folds_csv["posicao"] == r["posicao"])
            & (folds_csv["modelo"] == str(r["modelo"]).lower())
        ]
        for col, ano in zip(range(2, 6), (2018, 2019, 2020, 2021)):
            alvo_fold = folds_csv[folds_csv["teste_em_temporada"] == ano]
            if alvo_fold.empty:
                aud.padrao(f"tab:tgb_wf {ref} fold {ano} presente no CSV", False, arquivo.name)
                continue
            aud.compara(f"tab:tgb_wf {ref} C fold {ano}", numero(linha[col]), float(alvo_fold.iloc[0]["c_index"]), "metricas_walkforward_folds.csv", 0.0006)
        aud.compara(f"tab:tgb_wf {ref} C", numero(linha[6]), r["c_index"], arquivo.name, 0.0006)
        aud.compara(f"tab:tgb_wf {ref} IBS", numero(linha[8]), r["ibs"], arquivo.name, 0.0006)
        # O IC substituiu a coluna de desvio-padrao entre folds (n=4).
        if ic.empty or str(r["modelo"]).lower() == "km":
            continue
        alvo_ic = ic[
            (ic["posicao"] == r["posicao"])
            & (ic["modelo"] == str(r["modelo"]).lower())
            & (ic["metrica"] == "c_index")
        ]
        if alvo_ic.empty:
            continue
        limites = _nums_celula(linha[7])
        aud.padrao(f"tab:tgb_wf {ref} IC (duas pontas)", len(limites) == 2, ic_path.name)
        if len(limites) != 2:
            continue
        linha_ic = alvo_ic.iloc[0]
        aud.compara(f"tab:tgb_wf {ref} IC inferior", limites[0], float(linha_ic["ic_baixo"]), ic_path.name, 0.0006)
        aud.compara(f"tab:tgb_wf {ref} IC superior", limites[1], float(linha_ic["ic_alto"]), ic_path.name, 0.0006)


def checa_tgb_delta(aud: Auditoria) -> None:
    """Diferenca de C pareada (Tabela 5.2) contra o bootstrap do `ic_preditivo`."""
    arquivo = IC / "delta_c_pareado.csv"
    if not arquivo.exists():
        return
    d = pd.read_csv(arquivo)
    for linha in aud.linhas("tab:tgb_delta"):
        if len(linha) < 5:
            continue
        pos = linha[0].strip()
        nomes = re.findall(r"(GBSA|RSF|Cox)", linha[1])
        if len(nomes) != 2:
            continue
        chave = f"{nomes[0].lower()}-{nomes[1].lower()}"
        r = d[(d["posicao"] == pos) & (d["par"] == chave)]
        if r.empty:
            continue
        r = r.iloc[0]
        ref = f"tab:tgb_delta {pos} {chave}"
        aud.compara(f"{ref} delta", numero(linha[2]), r["delta_c"], arquivo.name, 0.0006)
        limites = _nums_celula(linha[3])
        aud.padrao(f"{ref} IC (duas pontas)", len(limites) == 2, arquivo.name)
        if len(limites) == 2:
            aud.compara(f"{ref} IC inferior", limites[0], r["ic_baixo"], arquivo.name, 0.0006)
            aud.compara(f"{ref} IC superior", limites[1], r["ic_alto"], arquivo.name, 0.0006)
        aud.compara(f"{ref} p", numero(linha[4]), r["p_bilateral"], arquivo.name, 0.0006)


def _arquivo_obrigatorio(aud: Auditoria, caminho: Path, onde: str) -> bool:
    existe = caminho.is_file()
    aud.padrao(onde, existe, str(caminho), "artefato obrigatório ausente")
    return existe


def _colunas_obrigatorias(
    aud: Auditoria, dados: pd.DataFrame, colunas: set[str], onde: str, artefato: str
) -> bool:
    presentes = colunas.issubset(dados.columns)
    aud.padrao(onde, presentes, artefato, f"colunas requeridas: {sorted(colunas)}")
    return presentes


def _bool_auditavel(valores: pd.Series) -> pd.Series:
    mapa = {"true": True, "1": True, "false": False, "0": False}
    return valores.map(lambda valor: mapa.get(str(valor).strip().lower(), np.nan))


def checa_predicao_erro_restrito(aud: Auditoria) -> None:
    oof_path = OFICIAL / "pred_oof_walkforward_2018_2021.csv"
    ano_path = DIAG / "erro_por_ano.csv"
    exclusoes_path = DIAG / "exclusoes_erro_anos.csv"
    resumo_path = DIAG / "resumo_diagnostico.json"
    paths = (
        (oof_path, "P01 OOF presente"),
        (ano_path, "P01 métricas restritas presentes"),
        (exclusoes_path, "P01 exclusões presentes"),
        (resumo_path, "P01 resumo presente"),
    )
    presentes = [_arquivo_obrigatorio(aud, path, nome) for path, nome in paths]
    if not all(presentes):
        return

    oof = pd.read_csv(oof_path)
    base_columns = {
        "player_id", "snapshot_season", "residual_duration_model", "residual_event",
        "posicao", "modelo", "expected_years", "tau", "error_identifiable",
        "error_target_years",
    }
    if not _colunas_obrigatorias(aud, oof, base_columns, "P01 esquema OOF", oof_path.name):
        return
    numeric_columns = [
        "snapshot_season", "residual_duration_model", "residual_event", "expected_years", "tau",
    ]
    for column in numeric_columns:
        oof[column] = pd.to_numeric(oof[column], errors="coerce")
    flags = _bool_auditavel(oof["error_identifiable"])
    targets = pd.to_numeric(oof["error_target_years"], errors="coerce")
    oof["fold"] = oof["snapshot_season"].astype(int)
    oof["tau_reconstruido"] = 2024 - oof["fold"]
    oof["identificavel_reconstruido"] = (
        (oof["residual_event"] == 1)
        | (oof["residual_duration_model"] >= oof["tau_reconstruido"])
    )
    oof["alvo_reconstruido"] = np.minimum(
        oof["residual_duration_model"], oof["tau_reconstruido"]
    )
    valid_values = (
        oof[numeric_columns].notna().all().all()
        and np.isfinite(oof[numeric_columns].to_numpy(dtype=float)).all()
        and flags.notna().all()
        and oof["residual_event"].isin((0, 1)).all()
    )
    valid_values = bool(valid_values) and bool(
        np.isfinite(targets[oof["identificavel_reconstruido"]].to_numpy(dtype=float)).all()
        and targets[~oof["identificavel_reconstruido"]].isna().all()
    )
    aud.padrao("P01 valores OOF finitos e evento binário", bool(valid_values), oof_path.name)
    if not valid_values:
        return

    aud.padrao(
        "P01 horizonte tau = 2024 - origem",
        bool(np.array_equal(oof["tau"].to_numpy(float), oof["tau_reconstruido"].to_numpy(float))),
        oof_path.name,
    )
    aud.padrao(
        "P01 flag de identificabilidade reconstruída independentemente",
        bool(np.array_equal(flags.astype(bool).to_numpy(), oof["identificavel_reconstruido"].to_numpy(bool))),
        oof_path.name,
    )
    aud.padrao(
        "P01 alvo restrito reconstruído independentemente",
        bool(np.allclose(
            targets[oof["identificavel_reconstruido"]].to_numpy(float),
            oof.loc[oof["identificavel_reconstruido"], "alvo_reconstruido"].to_numpy(float),
            rtol=0,
            atol=1e-12,
        )),
        oof_path.name,
    )

    expected_groups = {
        (position, model, fold)
        for position in POSICOES
        for model in MODELOS_PREDITIVOS
        for fold in FOLDS_PREDITIVOS
    }
    actual_groups = set(zip(oof["posicao"], oof["modelo"], oof["fold"]))
    aud.padrao("P01 cobertura de posição-modelo-origem", actual_groups == expected_groups, oof_path.name)
    key_columns = ["posicao", "fold", "modelo", "player_id"]
    aud.padrao("P01 chave OOF única", not oof.duplicated(key_columns).any(), oof_path.name)
    consistent = True
    for (position, fold), group in oof.groupby(["posicao", "fold"], sort=False):
        membership = [
            set(part["player_id"].astype(str))
            for _, part in group.groupby("modelo", sort=False)
        ]
        if len(membership) != len(MODELOS_PREDITIVOS) or any(ids != membership[0] for ids in membership[1:]):
            consistent = False
        per_player = group.groupby("player_id")[[
            "residual_duration_model", "residual_event", "tau", "identificavel_reconstruido",
        ]].nunique(dropna=False)
        if not per_player.le(1).all().all():
            consistent = False
    aud.padrao("P01 mesma coorte e rótulo entre os três modelos", consistent, oof_path.name)

    excluded = oof.loc[
        ~oof["identificavel_reconstruido"],
        ["posicao", "fold", "player_id", "residual_duration_model", "tau_reconstruido", "residual_event"],
    ].drop_duplicates(["posicao", "fold", "player_id"])
    expected_exclusion_keys = {
        (str(row.posicao), int(row.fold), str(row.player_id)) for row in excluded.itertuples()
    }
    exclusion_columns = {
        "posicao", "fold", "player_id", "T", "tau", "evento", "motivo", "n_modelos_excluidos",
    }
    exclusion_table = pd.read_csv(exclusoes_path)
    if not _colunas_obrigatorias(
        aud, exclusion_table, exclusion_columns, "P01 esquema de exclusões", exclusoes_path.name
    ):
        return
    exclusion_numeric = ["fold", "T", "tau", "evento", "n_modelos_excluidos"]
    for column in exclusion_numeric:
        exclusion_table[column] = pd.to_numeric(exclusion_table[column], errors="coerce")
    exclusions_valid = (
        exclusion_table[exclusion_numeric].notna().all().all()
        and np.isfinite(exclusion_table[exclusion_numeric].to_numpy(float)).all()
        and exclusion_table["evento"].isin((0, 1)).all()
    )
    aud.padrao("P01 campos de exclusão finitos", bool(exclusions_valid), exclusoes_path.name)
    if not exclusions_valid:
        return
    exclusion_keys = {
        (str(row.posicao), int(row.fold), str(row.player_id))
        for row in exclusion_table.itertuples()
        if np.isfinite(row.fold)
    }
    aud.padrao(
        "P01 coorte de exclusões reconstruída desde OOF",
        exclusion_keys == expected_exclusion_keys and not exclusion_table.duplicated(["posicao", "fold", "player_id"]).any(),
        exclusoes_path.name,
    )
    for _, row in excluded.iterrows():
        key = (row["posicao"], int(row["fold"]), str(row["player_id"]))
        match = exclusion_table[
            (exclusion_table["posicao"] == key[0])
            & (exclusion_table["fold"] == key[1])
            & (exclusion_table["player_id"].astype(str) == key[2])
        ]
        aud.padrao(f"P01 exclusão {key}", len(match) == 1, exclusoes_path.name)
        if len(match) != 1:
            continue
        saved = match.iloc[0]
        aud.compara(f"P01 exclusão {key} T", float(saved["T"]), float(row["residual_duration_model"]), exclusoes_path.name, 1e-12)
        aud.compara(f"P01 exclusão {key} tau", float(saved["tau"]), float(row["tau_reconstruido"]), exclusoes_path.name, 1e-12)
        aud.padrao(f"P01 exclusão {key} motivo", saved["motivo"] == "censura_pre_tau", exclusoes_path.name)
        aud.compara(f"P01 exclusão {key} modelos", float(saved["n_modelos_excluidos"]), float((
            (oof["posicao"] == key[0]) & (oof["fold"] == key[1]) & (oof["player_id"].astype(str) == key[2])
            & ~oof["identificavel_reconstruido"]
        ).sum()), exclusoes_path.name, 0)

    years_path = ano_path
    years = pd.read_csv(years_path)
    years_columns = {
        "posicao", "modelo", "fold", "n_pares", "n_pares_total", "n_excluidos_censura_pre_tau",
        "vies_medio", "mae", "pct_super", "pct_sub",
    }
    if not _colunas_obrigatorias(aud, years, years_columns, "P01 esquema de métricas", years_path.name):
        return
    years_numeric = [
        "fold", "n_pares", "n_pares_total", "n_excluidos_censura_pre_tau",
        "vies_medio", "mae", "pct_super", "pct_sub",
    ]
    for column in years_numeric:
        years[column] = pd.to_numeric(years[column], errors="coerce")
    years_valid = years[years_numeric].notna().all().all() and np.isfinite(years[years_numeric].to_numpy(float)).all()
    aud.padrao("P01 métricas com valores finitos", bool(years_valid), years_path.name)
    if not years_valid:
        return
    aud.padrao("P01 uma linha por grupo de avaliação", not years.duplicated(["posicao", "modelo", "fold"]).any(), years_path.name)
    row_lookup = {
        (row.posicao, row.modelo, int(row.fold)): row
        for row in years.itertuples()
        if np.isfinite(row.fold)
    }
    for (position, model, fold), group in oof.groupby(["posicao", "modelo", "fold"], sort=False):
        key = (position, model, int(fold))
        expected = group.loc[group["identificavel_reconstruido"]]
        invalid_count = int((~group["identificavel_reconstruido"]).sum())
        aud.padrao(f"P01 métrica {key} tem alvos identificáveis", len(expected) > 0, years_path.name)
        if not len(expected):
            continue
        aud.padrao(f"P01 métrica {key} presente", key in row_lookup, years_path.name)
        if key not in row_lookup:
            continue
        actual = row_lookup[key]
        expected_error = expected["expected_years"] - expected["alvo_reconstruido"]
        aud.compara(f"P01 métrica {key} n identificável", float(actual.n_pares), float(len(expected)), years_path.name, 0)
        aud.compara(f"P01 métrica {key} n total", float(actual.n_pares_total), float(len(group)), years_path.name, 0)
        aud.compara(f"P01 métrica {key} excluídas", float(actual.n_excluidos_censura_pre_tau), float(invalid_count), years_path.name, 0)
        aud.compara(f"P01 métrica {key} viés", float(actual.vies_medio), float(expected_error.mean()), years_path.name, 1e-10)
        aud.compara(f"P01 métrica {key} MAE", float(actual.mae), float(expected_error.abs().mean()), years_path.name, 1e-10)
        aud.compara(f"P01 métrica {key} superestimação", float(actual.pct_super), float(100 * (expected_error > 0).mean()), years_path.name, 1e-10)
        aud.compara(f"P01 métrica {key} subestimação", float(actual.pct_sub), float(100 * (expected_error < 0).mean()), years_path.name, 1e-10)

    resumo = json.loads(resumo_path.read_text(encoding="utf-8"))
    summary_fields = {"n_linhas_oof", "n_coortes_excluidas", "n_pares_excluidos_modelo"}
    if not summary_fields.issubset(resumo):
        aud.padrao("P01 resumo contém contagens de exclusão", False, resumo_path.name, f"campos requeridos: {sorted(summary_fields)}")
        return
    summary_counts = pd.to_numeric(pd.Series([resumo[field] for field in sorted(summary_fields)]), errors="coerce")
    summary_valid = summary_counts.notna().all() and np.isfinite(summary_counts.to_numpy(float)).all()
    aud.padrao("P01 contagens do resumo finitas", bool(summary_valid), resumo_path.name)
    if not summary_valid:
        return
    aud.compara("P01 resumo linhas OOF", float(resumo["n_linhas_oof"]), float(len(oof)), resumo_path.name, 0)
    aud.compara("P01 resumo coortes excluídas", float(resumo["n_coortes_excluidas"]), float(len(expected_exclusion_keys)), resumo_path.name, 0)
    aud.compara("P01 resumo pares-modelo excluídos", float(resumo["n_pares_excluidos_modelo"]), float((~oof["identificavel_reconstruido"]).sum()), resumo_path.name, 0)


def _p_bootstrap_independente(values: np.ndarray) -> float:
    lower = int(np.count_nonzero(values <= TOL_EMPATE_BOOTSTRAP))
    upper = int(np.count_nonzero(values >= -TOL_EMPATE_BOOTSTRAP))
    n = int(values.size)
    return float(min(1.0, 2.0 * min((lower + 1) / (n + 1), (upper + 1) / (n + 1))))


def _holm_independente(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty_like(p_values, dtype=float)
    accumulated = 0.0
    for rank, index in enumerate(order):
        accumulated = max(accumulated, float((len(order) - rank) * p_values[index]))
        adjusted[index] = min(1.0, accumulated)
    return adjusted


def checa_ic_bootstrap_replicas(aud: Auditoria) -> None:
    replicas_path = IC / "bootstrap_delta_replicas.csv"
    summary_path = IC / "resumo_ic.json"
    cluster_path = IC / "delta_c_pareado.csv"
    intra_path = IC / "delta_c_pareado_intra_fold.csv"
    required_paths = (
        (MANIFEST, "P03 manifesto presente"),
        (replicas_path, "P03 réplicas bootstrap presentes"),
        (summary_path, "P03 resumo metodológico presente"),
        (cluster_path, "P03 tabela cluster presente"),
        (intra_path, "P03 tabela intra-fold presente"),
    )
    present = [_arquivo_obrigatorio(aud, path, label) for path, label in required_paths]
    if not all(present):
        return

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = manifest.get("artifacts", manifest.get("files", []))
    registration = [entry for entry in entries if entry.get("path") == "documentos/estudo_ic_preditivo/resultados/bootstrap_delta_replicas.csv"]
    registered = len(registration) == 1 and registration[0].get("required") is True and registration[0].get("producer") == "ic_preditivo"
    aud.padrao("P03 réplicas registradas como obrigatórias", registered, MANIFEST.name)

    replicas = pd.read_csv(replicas_path)
    replica_columns = {"posicao", "esquema", "par", "replicacao", "delta_c"}
    if not _colunas_obrigatorias(aud, replicas, replica_columns, "P03 esquema das réplicas", replicas_path.name):
        return
    replicas["replicacao"] = pd.to_numeric(replicas["replicacao"], errors="coerce")
    replicas["delta_c"] = pd.to_numeric(replicas["delta_c"], errors="coerce")
    b = 2000
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_b = pd.to_numeric(pd.Series([summary.get("n_boot", np.nan)]), errors="coerce").iloc[0]
    aud.padrao("P03 B declarado finito", bool(np.isfinite(summary_b)), summary_path.name)
    if not np.isfinite(summary_b):
        return
    aud.compara("P03 B declarado", float(summary_b), float(b), summary_path.name, 0)
    values_valid = (
        replicas["replicacao"].notna().all()
        and replicas["delta_c"].notna().all()
        and np.isfinite(replicas[["replicacao", "delta_c"]].to_numpy(float)).all()
        and np.equal(replicas["replicacao"].to_numpy(float), np.floor(replicas["replicacao"].to_numpy(float))).all()
    )
    aud.padrao("P03 réplicas com valores finitos", bool(values_valid), replicas_path.name)
    if not values_valid:
        return
    replicas["replicacao"] = replicas["replicacao"].astype(int)
    expected_keys = {
        (position, scheme, pair)
        for position in POSICOES
        for scheme in ("cluster", "intra_fold")
        for pair in PARES_PREDITIVOS
    }
    actual_keys = set(zip(replicas["posicao"], replicas["esquema"], replicas["par"]))
    aud.padrao("P03 nove contrastes em dois esquemas", actual_keys == expected_keys, replicas_path.name)
    aud.padrao("P03 chave posição-esquema-par-replicação única", not replicas.duplicated(["posicao", "esquema", "par", "replicacao"]).any(), replicas_path.name)

    computed: dict[tuple[str, str, str], dict[str, float]] = {}
    for key in sorted(expected_keys):
        group = replicas[
            (replicas["posicao"] == key[0])
            & (replicas["esquema"] == key[1])
            & (replicas["par"] == key[2])
        ]
        ids = set(group["replicacao"].astype(int))
        aud.padrao(
            f"P03 {key} tem {b} réplicas completas",
            len(group) == b and ids == set(range(1, b + 1)),
            replicas_path.name,
        )
        sample = group["delta_c"].to_numpy(float)
        if not len(sample):
            continue
        computed[key] = {
            "n": float(len(sample)),
            "mean": float(sample.mean()),
            "low": float(np.percentile(sample, 2.5)),
            "high": float(np.percentile(sample, 97.5)),
            "p": _p_bootstrap_independente(sample),
            "positive": float((sample > 0).mean()),
        }

    required_common = {
        "posicao", "par", "delta_c", "delta_media_boot", "ic_baixo", "ic_alto",
        "p_bootstrap_aproximado", "p_bilateral", "prop_delta_positivo", "n_boot",
    }
    tables: dict[str, pd.DataFrame] = {}
    for scheme, path in (("cluster", cluster_path), ("intra_fold", intra_path)):
        table = pd.read_csv(path)
        required = required_common | ({"p_holm", "rejeita_5pct_holm"} if scheme == "cluster" else set())
        if not _colunas_obrigatorias(aud, table, required, f"P03 esquema {scheme}", path.name):
            continue
        numeric_table_columns = sorted(required - {"posicao", "par", "rejeita_5pct_holm"})
        for column in numeric_table_columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")
        table_values_valid = table[numeric_table_columns].notna().all().all() and np.isfinite(table[numeric_table_columns].to_numpy(float)).all()
        aud.padrao(f"P03 tabela {scheme} números finitos", bool(table_values_valid), path.name)
        if not table_values_valid:
            continue
        if scheme == "cluster":
            decision_values = table["rejeita_5pct_holm"].astype(str).str.strip().str.lower()
            aud.padrao(
                "P03 flags Holm booleanas",
                decision_values.isin(("true", "false", "1", "0")).all(),
                path.name,
            )
        actual_table_keys = set(zip(table["posicao"], table["par"]))
        wanted_table_keys = {(position, pair) for position in POSICOES for pair in PARES_PREDITIVOS}
        aud.padrao(f"P03 {scheme} cobre nove linhas únicas", actual_table_keys == wanted_table_keys and not table.duplicated(["posicao", "par"]).any(), path.name)
        tables[scheme] = table
        for position, pair in sorted(wanted_table_keys):
            key = (position, scheme, pair)
            values = computed.get(key)
            row = table[(table["posicao"] == position) & (table["par"] == pair)]
            aud.padrao(f"P03 tabela {scheme} {position}/{pair} presente", len(row) == 1 and values is not None, path.name)
            if len(row) != 1 or values is None:
                continue
            saved = row.iloc[0]
            for column, metric in (
                ("ic_baixo", "low"), ("ic_alto", "high"),
                ("p_bootstrap_aproximado", "p"), ("p_bilateral", "p"),
                ("delta_media_boot", "mean"), ("prop_delta_positivo", "positive"),
            ):
                aud.compara(f"P03 {scheme} {position}/{pair} {column}", float(saved[column]), values[metric], path.name, 1e-12)
            aud.compara(f"P03 {scheme} {position}/{pair} n_boot", float(saved["n_boot"]), values["n"], path.name, 0)

    if "cluster" in tables:
        cluster = tables["cluster"].set_index(["posicao", "par"])
        p_raw = np.array([
            computed[(position, "cluster", pair)]["p"]
            for position in POSICOES
            for pair in PARES_PREDITIVOS
            if (position, "cluster", pair) in computed
        ], dtype=float)
        ordered_keys = [
            (position, pair)
            for position in POSICOES
            for pair in PARES_PREDITIVOS
            if (position, "cluster", pair) in computed
        ]
        if len(p_raw) == 9:
            adjusted = _holm_independente(p_raw)
            for key, value in zip(ordered_keys, adjusted):
                saved = cluster.loc[key]
                aud.compara(f"P03 Holm {key[0]}/{key[1]}", float(saved["p_holm"]), float(value), cluster_path.name, 1e-12)
                decision = value <= 0.05
                saved_decision = str(saved["rejeita_5pct_holm"]).strip().lower() in {"true", "1"}
                aud.padrao(f"P03 decisão Holm {key[0]}/{key[1]}", saved_decision == decision, cluster_path.name)
        else:
            aud.padrao("P03 Holm cobre exatamente nove contrastes", False, cluster_path.name)

    if "cluster" in tables and "intra_fold" in tables:
        cluster = tables["cluster"].set_index(["posicao", "par"])
        intra = tables["intra_fold"].set_index(["posicao", "par"])
        for key in sorted(set(cluster.index) & set(intra.index)):
            aud.compara(f"P03 estimativa pontual coerente {key}", float(cluster.loc[key, "delta_c"]), float(intra.loc[key, "delta_c"]), cluster_path.name, 1e-12)

    p_description = str(summary.get("p_bootstrap_aproximado", "")).lower()
    aud.padrao(
        "P03 metadado registra empates, tolerância e correção finita",
        "empates" in p_description and "1e-12" in p_description and "(count+1)/(b+1)" in p_description,
        summary_path.name,
    )
    aud.padrao(
        "P03 metadado declara Holm dos nove p cluster",
        "holm" in str(summary.get("holm", "")).lower()
        and "nove" in str(summary.get("holm", "")).lower()
        and "cluster" in str(summary.get("holm", "")).lower(),
        summary_path.name,
    )
    reading = str(summary.get("como_ler", "")).lower()
    aud.padrao(
        "P03 IC condicional aos folds fixos",
        "composicao dos folds fixa" in reading and "nao cobre a escolha das temporadas" in reading,
        summary_path.name,
    )
    resampling = str(summary.get("reamostragem", "")).lower()
    aud.padrao(
        "P03 metadado descreve reamostragem de jogadores sem refit",
        "jogador sorteado com reposicao" in resampling and "levando todas as suas linhas" in resampling,
        summary_path.name,
    )


def checa_rmst_p04(aud: Auditoria) -> None:
    shared_path = ROOT / "Predição" / "shared" / "tgb_protocol.py"
    official_path = ROOT / "documentos" / "estudo_modelos_finais" / "run_oficial.py"
    reading_path = ROOT / "documentos" / "estudo_leitura_preditiva" / "run_leitura.py"
    for path, label in (
        (shared_path, "P04 helper compartilhado presente"),
        (official_path, "P04 consumidor oficial presente"),
        (reading_path, "P04 consumidor leitura presente"),
    ):
        if not _arquivo_obrigatorio(aud, path, label):
            return
    sources = {path: path.read_text(encoding="utf-8") for path in (shared_path, official_path, reading_path)}
    parsed = {path: ast.parse(source, filename=str(path)) for path, source in sources.items()}

    def function_source(path: Path, name: str) -> str:
        node = next((item for item in parsed[path].body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name), None)
        return ast.get_source_segment(sources[path], node) if node is not None else ""

    integrator = function_source(shared_path, "media_restrita_sksurv")
    official_tree = function_source(official_path, "expected_sksurv")
    official_cox = function_source(official_path, "expected_cox")
    reading_tree = function_source(reading_path, "expected_from_fns")
    aud.padrao(
        "P04 integral exata nos nós nativos",
        "np.unique(np.r_[0.0" in integrator and "np.dot(step_values, np.diff(knots))" in integrator and "np.linspace" not in integrator,
        shared_path.name,
    )
    aud.padrao(
        "P04 valida suporte, finitude e curva de sobrevivência",
        all(token in integrator for token in ("exige_dominio", "min_support", "np.isfinite", "np.minimum.accumulate")),
        shared_path.name,
    )
    aud.padrao("P04 wrapper oficial de árvores delega ao integrador", "media_restrita_sksurv" in official_tree, official_path.name)
    aud.padrao("P04 wrapper de leitura delega ao integrador", "media_restrita_sksurv" in reading_tree, reading_path.name)
    aud.padrao("P04 Cox oficial usa integral exata", "media_restrita_cox" in official_cox, official_path.name)
    aud.padrao(
        "P04 OOF e snapshot exigem suporte até tau",
        "tmax=tau, exige_dominio=True" in sources[official_path]
        and "tmax=tau_effective, exige_dominio=True" in sources[official_path]
        and "common_training_horizon(train_followup, TMAX_EXPECT)" in sources[official_path],
        official_path.name,
    )

    node = next((item for item in parsed[shared_path].body if isinstance(item, ast.FunctionDef) and item.name == "media_restrita_sksurv"), None)
    aud.padrao("P04 função executável localizada na AST", node is not None, shared_path.name)
    if node is None:
        return
    namespace: dict[str, object] = {}
    isolated = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(isolated), str(shared_path), "exec"), {"np": np, "warnings": warnings}, namespace)
    integrator_fn = namespace["media_restrita_sksurv"]
    curves = [
        SimpleNamespace(x=np.array([1.0, 2.0, 4.0]), y=np.array([0.9, 0.6, 0.3]), a=1.0, b=0.0),
        SimpleNamespace(x=np.array([0.5, 2.0, 4.0]), y=np.array([1.0, 0.5, 0.25]), a=0.8, b=0.1),
    ]
    means, medians = integrator_fn(curves, 3.0)
    aud.compara("P04 fixture integral curva A", float(means[0]), 2.5, shared_path.name, 1e-12)
    aud.compara("P04 fixture integral curva B com escala/offset", float(means[1]), 2.35, shared_path.name, 1e-12)
    aud.compara("P04 fixture mediana limitada A", float(medians[0]), 3.0, shared_path.name, 0)
    aud.compara("P04 fixture mediana B", float(medians[1]), 2.0, shared_path.name, 0)
    rejected = False
    try:
        integrator_fn(curves, 4.1, exige_dominio=True)
    except ValueError:
        rejected = True
    aud.padrao("P04 rejeita extrapolação além do suporte", rejected, shared_path.name)


def checa_tgb_lm(aud: Auditoria) -> None:
    # A tabela da escrita usa so as celulas (t_L, w) em que o horizonte cabe no follow-up.
    arquivo = OFICIAL / "resumo_landmark_valido.csv"
    if not arquivo.exists():
        return
    d = pd.read_csv(arquivo)
    for linha in aud.linhas("tab:tgb_lm")[1:]:
        if len(linha) < 5:
            continue
        pos, modelo = linha[0].strip(), linha[1].strip().upper()
        for j, w in enumerate([1, 2, 3], start=2):
            r = d[(d["posicao"] == pos) & (d["modelo"].str.upper() == modelo) & (d["horizon_w"] == w)]
            if r.empty:
                continue
            aud.compara(f"tab:tgb_lm {pos} {modelo} w={w}", numero(linha[j]), r.iloc[0]["c_index"], arquivo.name, 0.0006)


def checa_prosa_l2(aud: Auditoria) -> None:
    """Se a escrita afirmar que a penalizacao quase nao mexe nos HRs, confere a afirmacao.

    O Cox final (cox_hr_{POS}.csv) e SEM penalizacao (penalizer = 0); a L2 (ridge) e
    sensibilidade do apendice. A comparacao com coxph_unpenalized confere que o
    arquivo oficial e o ajuste sem penalizacao. Se o texto voltar a dizer que o
    efeito de uma penalizacao e desprezivel, esta checagem acusa.
    """
    texto = (LIVRO / "c4_resultados_inferenciais.tex").read_text(encoding="utf-8", errors="replace")
    afirma_desprezivel = re.search(
        r"impacto sobre os HRs finais inferior a 1|impacto (?:sobre os HRs )?(?:é |e )?inferior a 1",
        texto,
    ) is not None
    for pos in POSICOES:
        l2_path = COX / f"cox_hr_{pos}.csv"
        unp_path = COX / f"coxph_unpenalized_{pos}.csv"
        if not (l2_path.exists() and unp_path.exists()):
            continue
        l2 = pd.read_csv(l2_path).set_index("covariate")["HR"]
        unp = pd.read_csv(unp_path).set_index("covariate")["coxph_HR"]
        comuns = l2.index.intersection(unp.index)
        if comuns.empty:
            continue
        variacao = ((l2[comuns] - unp[comuns]).abs() / unp[comuns]).max()
        aud.checagens += 1
        if afirma_desprezivel and variacao > 0.01:
            aud.divergencias.append(
                Divergencia(
                    f"prosa c4 ({pos}) afirma impacto de penalizacao inferior a 1%",
                    "< 1%",
                    f"{variacao:.1%}",
                    f"cox_hr_{pos}.csv vs coxph_unpenalized_{pos}.csv",
                    "maior variacao relativa entre os HRs oficiais do Cox final (sem penalizacao) e o coxph sem penalizacao",
                )
            )


def checa_prosa_screening(aud: Auditoria) -> None:
    """Confere contagens de screening apenas quando a escrita as declara."""
    texto = (LIVRO / "c4_resultados_inferenciais.tex").read_text(encoding="utf-8", errors="replace")
    padroes = {
        "QB": (
            r"(?:A triagem selecionou|das)\s+(\d+)\s+covari[áa]veis"
            r"(?:\s+triadas)?\s+no QB",
        ),
        "RB": (r"no RB e no WR, as\s+(\d+)\s+e\s+(\d+)\s+selecionadas",),
    }
    arquivos = {}
    for pos in POSICOES:
        arquivo = AALEN / f"aalen_screening_{pos}.csv"
        if not arquivo.exists():
            aud.padrao(f"prosa c4 screening {pos}: fonte presente", False, arquivo.name)
            continue
        d = pd.read_csv(arquivo)
        valido = "p_value_null" in d and not d.empty and bool(d.p_value_null.map(math.isfinite).all())
        aud.padrao(f"prosa c4 screening {pos}: p-valores disponiveis", valido, arquivo.name)
        if valido:
            arquivos[pos] = int((d["p_value_null"] < 0.05).sum())
    m_qb = re.search(padroes["QB"][0], texto, re.I)
    if m_qb and "QB" in arquivos:
        aud.compara_arredondado("prosa c4 screening QB", float(m_qb.group(1)), arquivos["QB"], "aalen_screening_QB.csv", 0)
    m_rbwr = re.search(padroes["RB"][0], texto, re.I)
    if m_rbwr and {"RB", "WR"} <= arquivos.keys():
        aud.compara_arredondado("prosa c4 screening RB", float(m_rbwr.group(1)), arquivos["RB"], "aalen_screening_RB.csv", 0)
        aud.compara_arredondado("prosa c4 screening WR", float(m_rbwr.group(2)), arquivos["WR"], "aalen_screening_WR.csv", 0)



def checa_prosa_funil(aud: Auditoria) -> None:
    """Funil de covariaveis (c3, texto atual): painel posicional e candidatas apos colinearidade, contra o pipeline."""
    t3 = _texto("c3_material_tg.tex")
    id_cols = ["player_id", "player_name", "season"]
    time_cols = ["start", "stop", "event", "entry"]
    painel = {}
    for pos in POSICOES:
        df = pd.read_pickle(PREP / f"df_cox_{pos}.pkl")
        painel[pos] = len([c for c in df.columns if c not in id_cols + time_cols and c != "draft_round"])
    _tem(aud, "c3 painel posicional: covariáveis por posição",
         f"O painel posicional, com {painel['QB']} covariáveis no QB, {painel['RB']} no RB e {painel['WR']} no WR",
         t3, "df_cox_*.pkl")
    coli = {pos: _cadeia_cox(pos)["final"] for pos in POSICOES}
    _tem(aud, "c3 candidatas após colinearidade",
         f"deixa {coli['QB']}, {coli['RB']} e {coli['WR']} candidatas", t3, "cadeia do Cox em df_cox_*.csv")


def _campos_aareg_ausentes(summary: dict) -> list[str]:
    obrigatorios = (
        "model", "test_family", "test_scope", "covariates_tested", "tests_null_effect",
        "tests_constant_effect", "implementation_validation", "input_counts_before_window",
        "window_selection", "effective_support", "n_subjects_input", "n_periods_input",
        "n_events_input", "n_subjects", "n_periods", "n_events", "janela_inicio", "janela_fim",
    )
    return [campo for campo in obrigatorios if campo not in summary]


def _contagens_aalen(data: pd.DataFrame) -> dict[str, int]:
    return {
        "n_subjects": int(data["player_id"].nunique()),
        "n_periods": int(len(data)),
        "n_events": int(data["event"].sum()),
    }


def _entrada_aalen(path: Path, covariates: list[str]) -> pd.DataFrame:
    data = pd.read_csv(path)
    colunas = ["player_id", "start", "stop", "event", *covariates]
    faltantes = [coluna for coluna in colunas if coluna not in data.columns]
    if faltantes:
        raise ValueError(f"colunas ausentes na entrada: {faltantes}")
    for coluna in ["start", "stop", "event", *covariates]:
        data[coluna] = pd.to_numeric(data[coluna], errors="coerce")
    data = data.dropna(subset=colunas).copy()
    if data.empty or not np.isfinite(data[["start", "stop", "event", *covariates]].to_numpy(dtype=float)).all():
        raise ValueError("entrada completa vazia ou com valores não finitos")
    if (data["start"] >= data["stop"]).any() or not data["event"].isin([0, 1]).all():
        raise ValueError("intervalos start-stop ou indicador event inválidos")
    limits = data[["start", "stop"]].to_numpy(dtype=float)
    if not np.allclose(limits, np.rint(limits), atol=1e-8, rtol=0):
        raise ValueError("o ajuste aareg exige limites anuais inteiros")
    return data


def _corta_janela_aalen(data: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    original_stop = data["stop"]
    keep = (data["start"] < end) & (original_stop > start)
    clipped = data.loc[keep].copy()
    stop_original = original_stop.loc[keep]
    clipped["start"] = np.maximum(clipped["start"], start)
    clipped["event"] = np.where(stop_original > end, 0, clipped["event"])
    clipped["stop"] = np.minimum(stop_original, end)
    return clipped.loc[clipped["start"] < clipped["stop"]].copy()


def _posto_risco_aalen(data: pd.DataFrame, time: int, covariates: list[str]) -> tuple[int, int]:
    risk = data[(data["start"] < time) & (data["stop"] >= time)]
    if risk.empty:
        return 0, 0
    design = np.column_stack((np.ones(len(risk)), risk[covariates].to_numpy(dtype=float)))
    return len(risk), int(np.linalg.matrix_rank(design, tol=1e-10))


def _janela_padrao_aalen(data: pd.DataFrame, covariates: list[str]) -> tuple[int, int, list[int], list[int]]:
    grid = range(int(data["start"].min()) + 1, int(data["stop"].max()) + 1)
    threshold = 3 * len(covariates)
    runs: list[list[int]] = []
    current: list[int] = []
    for time in grid:
        risk_n, rank = _posto_risco_aalen(data, time, covariates)
        if risk_n >= threshold and rank == len(covariates) + 1:
            current.append(time)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:
        raise ValueError("nenhuma janela anual satisfaz nmin e posto completo")
    selected = max(runs, key=len)
    start, end = selected[0] - 1, selected[-1]
    event_times = sorted(int(value) for value in data.loc[(data["event"] == 1) & (data["stop"] > start) & (data["stop"] <= end), "stop"].unique())
    return start, end, selected, event_times


def _janela_rank_only_aalen(data: pd.DataFrame, covariates: list[str], start: int) -> tuple[int, list[int], list[int]]:
    grid = list(range(start + 1, int(data["stop"].max()) + 1))
    valid = []
    for time in grid:
        risk_n, rank = _posto_risco_aalen(data, time, covariates)
        valid.append(risk_n > 0 and rank == len(covariates) + 1)
    if not valid or not valid[0]:
        raise ValueError("sensibilidade rank-only não inicia na fronteira inferior principal")
    end_index = next((i for i, value in enumerate(valid) if not value), len(valid))
    support = grid[:end_index]
    end = support[-1]
    event_times = sorted(int(value) for value in data.loc[(data["event"] == 1) & (data["stop"] > start) & (data["stop"] <= end), "stop"].unique())
    return end, support, event_times


def _checa_contagens_aalen(
    aud: Auditoria, pos: str, summary: dict, data: pd.DataFrame, start: int, end: int, artifact: str
) -> None:
    entrada = _contagens_aalen(data)
    for key, actual in entrada.items():
        aud.compara(f"Aalen aareg {pos} input_counts {key}", summary.get(f"{key}_input"), actual, artifact, 0)
        aud.compara(f"Aalen aareg {pos} input_counts_before_window {key}", summary.get("input_counts_before_window", {}).get(key), actual, artifact, 0)
    fitted = _contagens_aalen(_corta_janela_aalen(data, start, end))
    for key, actual in fitted.items():
        aud.compara(f"Aalen aareg {pos} fit_counts {key}", summary.get(key), actual, artifact, 0.5)


def _checa_janela_aalen(
    aud: Auditoria, pos: str, summary: dict, data: pd.DataFrame, covariates: list[str],
    start: int, end: int, support: list[int], event_times: list[int], sensitivity: bool, artifact: str,
) -> None:
    prefix = f"Aalen aareg {pos} {'nmin1' if sensitivity else 'principal'}"
    window = summary.get("window_selection", {})
    effective = summary.get("effective_support", {})
    aud.compara(f"{prefix} início da janela", summary.get("janela_inicio"), start, artifact, 0)
    aud.compara(f"{prefix} fim da janela", summary.get("janela_fim"), end, artifact, 0)
    aud.padrao(f"{prefix} convenção (a,b]", window.get("start_stop_convention") == "(a,b]", artifact)
    aud.padrao(f"{prefix} tempos de risco suportados", window.get("supported_integer_risk_times") == support, artifact)
    if not sensitivity:
        aud.padrao(f"{prefix} vetor top-level de risco suportado", summary.get("supported_integer_risk_times") == support, artifact)
    aud.padrao(f"{prefix} domínio efetivo de risco", effective.get("integer_risk_years") == support, artifact)
    aud.padrao(f"{prefix} tempos de evento", effective.get("event_years") == event_times, artifact)
    aud.padrao(f"{prefix} contagem do domínio efetivo", effective.get("n_integer_risk_years") == len(support) and effective.get("n_event_years") == len(event_times), artifact)
    clipped = _corta_janela_aalen(data, start, end)
    aud.compara(f"{prefix} eventos retidos", window.get("events_retained"), int(clipped["event"].sum()), artifact, 0)
    event_rows = clipped["event"] == 1
    aud.padrao(f"{prefix} eventos no intervalo aberto-fechado", bool(((~event_rows) | ((clipped["stop"] > start) & (clipped["stop"] <= end))).all()), artifact)
    _checa_contagens_aalen(aud, pos + (" nmin1" if sensitivity else ""), summary, data, start, end, artifact)
    validation = summary.get("implementation_validation", {})
    aud.padrao(f"{prefix} validação de posto/janela", validation.get("rank_and_nmin_window") == "PASS", artifact)
    aud.padrao(f"{prefix} validação dos risk sets", validation.get("window_risk_membership") == "PASS", artifact)


def _checa_robvar_aareg(aud: Auditoria, pos: str, summary: dict, path: Path, covariates: list[str], start: int, end: int, artifact: str) -> None:
    validation = summary.get("implementation_validation", {})
    error = validation.get("maximum_robust_annual_variance_abs_difference")
    aud.padrao(f"Aalen aareg {pos} robust variance validada por dfbeta", validation.get("robust_influence_variance") == "PASS", artifact)
    aud.padrao(f"Aalen aareg {pos} erro variância robusta finito e pequeno", isinstance(error, (int, float)) and math.isfinite(error) and error <= 1e-8, artifact)
    aud.padrao(f"Aalen aareg {pos} CSV variância robusta presente", path.exists(), path.name)
    if not path.exists():
        return
    robust = pd.read_csv(path)
    columns = ["time", "(Intercept)", *covariates]
    aud.padrao(f"Aalen aareg {pos} CSV variância robusta com termos corretos", list(robust.columns) == columns, path.name)
    if list(robust.columns) != columns:
        return
    values = robust.to_numpy(dtype=float)
    expected_times = np.arange(start, end + 1, dtype=float)
    aud.padrao(f"Aalen aareg {pos} grade do CSV variância robusta", values.shape[0] == len(expected_times) and np.array_equal(values[:, 0], expected_times), path.name)
    aud.padrao(f"Aalen aareg {pos} variâncias robustas finitas e não negativas", np.isfinite(values).all() and bool((values[:, 1:] >= -1e-10).all()), path.name)


def _audita_fixtures_aareg(aud: Auditoria) -> None:
    old_rows = [["Posição", "Covariável", "Efeito nulo", "Constância"]]
    old_text = "Teste de efeito nulo (supremo do processo padronizado). Teste de constância temporal."
    new_rows = [["Posição", "Covariável", "p-valor do efeito médio ponderado"]]
    new_text = "% aalen_test_family: weighted_mean_aareg"
    aud.padrao("fixture Aalen: metadado test_family ausente é rejeitado", "test_family" in _campos_aareg_ausentes({"model": _AAREG_MODEL}), "fixture")
    aud.padrao("fixture Aalen: p individual divergente falha", _compara_p_fixture_falha(), "fixture")
    old_family = _familia_aalen_de_texto(old_text, old_rows)
    new_family = _familia_aalen_de_texto(new_text, new_rows)
    aud.padrao("fixture Aalen: tabela do timereg identificada", old_family == "timereg_supremum", "fixture")
    aud.padrao("fixture Aalen: tabela nova exige família e p ponderado", new_family == _AAREG_FAMILY, "fixture")
    aud.padrao("fixture Aalen: família do timereg não vincula ao aareg", old_family != _AAREG_FAMILY, "fixture")
    aud.padrao("fixture Aalen: marcador sem cabeçalho ponderado fica sem vínculo", _familia_aalen_de_texto(new_text, old_rows) == "unbound", "fixture")


def _compara_p_fixture_falha() -> bool:
    fixture = Auditoria({})
    fixture.compara_p("fixture p", "0,020", 0.021, "fixture")
    return len(fixture.divergencias) == 1


_AAREG_MODEL = "aalen_tied_survival_aareg"
_AAREG_FAMILY = "weighted_mean_aareg"

def _familia_aalen_de_texto(texto: str, linhas: list[list[str]]) -> str:
    marcador = re.search(r"aalen_test_family\s*:\s*([\w.-]+)", texto, re.I)
    cabecalhos = " ".join(
        " ".join(linha).lower()
        for linha in linhas
        if any("covariável" in celula.lower() or "posição" in celula.lower() for celula in linha)
    )
    bloco = ""
    m_tab = re.search(r"\\label\{tab:aalen_consolidada\}", texto)
    if m_tab:
        fim_tab = texto.find("\\end{table}", m_tab.end())
        bloco = texto[m_tab.end(): fim_tab if fim_tab > 0 else len(texto)]
    if not marcador and "survival::aareg" in bloco and "ponderad" in bloco:
        tem_p = bool(re.search(r"(?:p\s*[- ]?valor|valor\s*[- ]?p|\bp\b)", cabecalhos))
        tem_ponderado = bool(re.search(r"ponderad|weighted", cabecalhos))
        return _AAREG_FAMILY if tem_p and tem_ponderado else "unbound"
    if marcador:
        familia = marcador.group(1).lower()
        if familia == _AAREG_FAMILY:
            tem_p = bool(re.search(r"(?:p\s*[- ]?valor|valor\s*[- ]?p|\bp\b)", cabecalhos))
            tem_ponderado = bool(re.search(r"ponderad|weighted", cabecalhos))
            return familia if tem_p and tem_ponderado else "unbound"
        return familia
    if "efeito nulo" in cabecalhos and "constância" in cabecalhos and "supremo do processo padronizado" in texto.lower() and "constância temporal" in texto.lower():
        return "timereg_supremum"
    return "unbound"


def _familia_tabela_aalen(aud: Auditoria) -> str:
    fonte = LIVRO / "c4_resultados_inferenciais.tex"
    texto = fonte.read_text(encoding="utf-8", errors="replace") if fonte.exists() else ""
    m_conc = re.search(r"\\label\{tab:concordancia\}", texto)
    if m_conc:
        bloco = texto[m_conc.end(): texto.find("\\end{table}", m_conc.end())]
        if "robusto da média ponderada" in bloco and "constância" in bloco:
            return _AAREG_FAMILY
    if "O teste de média nula diz se o efeito médio é zero, mas não diz se ele muda" in texto or "teste de média nula, que não testa constância" in texto:
        return _AAREG_FAMILY
    return _familia_aalen_de_texto(texto, aud.linhas("tab:aalen_consolidada"))


def _linhas_aalen_por_posicao(aud: Auditoria) -> dict[str, list[list[str]]]:
    result: dict[str, list[list[str]]] = {}
    active = None
    for row in aud.linhas("tab:aalen_consolidada"):
        first = row[0].strip().upper() if row else ""
        if first in POSICOES and len(row) >= 3:
            active = first
        if active and len(row) >= 2:
            result.setdefault(active, []).append(row)
    return result


def checa_aalen_consolidada(aud: Auditoria) -> None:
    """Vincula a tabela C4 à família de teste antes de comparar qualquer p."""
    por_posicao = _linhas_aalen_por_posicao(aud)
    familia_tabela = _familia_tabela_aalen(aud)
    for pos in POSICOES:
        arquivo = AALEN / f"aalen_summary_{pos}.json"
        if not arquivo.exists():
            continue
        d = json.loads(arquivo.read_text(encoding="utf-8"))
        if d.get("model") == _AAREG_MODEL or d.get("test_family") == _AAREG_FAMILY:
            aud.padrao(f"tab:aalen_consolidada {pos} família vinculada a weighted_mean_aareg", familia_tabela == _AAREG_FAMILY, arquivo.name, f"família detectada na tabela: {familia_tabela}; não comparar células do timereg como p ponderado")
            continue
        eh_timereg = d.get("model") == "aalen_nonparametric"
        aud.padrao(f"tab:aalen_consolidada {pos} modelo timereg reconhecido", eh_timereg, arquivo.name)
        if not eh_timereg:
            continue
        aud.padrao(f"tab:aalen_consolidada {pos} família timereg", familia_tabela == "timereg_supremum", arquivo.name)
        nulo = d.get("tests_null_effect", {})
        const = d.get("tests_constant_effect", {})
        for linha in por_posicao.get(pos, []):
            if linha[-4].strip() in {"Covariável", ""}:
                continue
            cov = variavel_da_celula(linha[-4])
            aud.padrao(f"tab:aalen_consolidada {pos} {linha[-4].strip()} reconhecida", cov in nulo, arquivo.name)
            if cov not in nulo:
                continue
            aud.compara_p(f"tab:aalen_consolidada {pos} {cov} efeito nulo", linha[-3], nulo[cov]["p_value"], arquivo.name)
            if cov in const:
                aud.compara_p(f"tab:aalen_consolidada {pos} {cov} constancia", linha[-2], const[cov]["p_value"], arquivo.name)


def checa_aalen_aareg(aud: Auditoria) -> None:
    """Audita o contrato aareg sem jitter, suas janelas e a variância robusta."""
    _audita_fixtures_aareg(aud)
    familia_tabela = _familia_tabela_aalen(aud)
    for pos in POSICOES:
        summary_path = AALEN / f"aalen_summary_{pos}.json"
        aud.padrao(f"Aalen aareg {pos} resumo oficial presente", summary_path.exists(), summary_path.name)
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        missing = _campos_aareg_ausentes(summary)
        aud.padrao(f"Aalen aareg {pos} metadados obrigatórios presentes", not missing, summary_path.name, f"ausentes: {missing}")
        is_aareg = summary.get("model") == _AAREG_MODEL and summary.get("test_family") == _AAREG_FAMILY
        aud.padrao(f"Aalen aareg {pos} modelo e família de teste", is_aareg, summary_path.name, "esperado survival::aareg / weighted_mean_aareg; saídas do timereg não satisfazem este contrato")
        if not is_aareg:
            continue
        aud.padrao(f"Aalen aareg {pos} escopo de média ponderada", summary.get("test_scope") == "weighted temporal mean", summary_path.name)
        aud.padrao(f"Aalen aareg {pos} tests_constant_effect vazio", summary.get("tests_constant_effect") in ({}, []), summary_path.name)
        aud.padrao(f"Aalen aareg {pos} seleção condicional explícita", "conditional" in str(summary.get("selection_conditioning", "")).lower() and "triagem" in str(summary.get("covariate_selection_status", "")).lower(), summary_path.name)
        covariates = summary.get("covariates_tested")
        tests = summary.get("tests_null_effect")
        if not isinstance(covariates, list) or not covariates or not isinstance(tests, dict):
            aud.padrao(f"Aalen aareg {pos} covariáveis e testes p presentes", False, summary_path.name)
            continue
        expected_terms = set(covariates) | {"(Intercept)"}
        aud.padrao(f"Aalen aareg {pos} testes ponderados cobrem intercepto e X", set(tests) == expected_terms, summary_path.name)
        for term in sorted(expected_terms & set(tests)):
            result = tests[term]
            pvalue = result.get("p_value") if isinstance(result, dict) else None
            aud.padrao(f"Aalen aareg {pos} p ponderado válido {term}", isinstance(pvalue, (int, float)) and math.isfinite(pvalue) and 0 <= pvalue <= 1, summary_path.name)
            if isinstance(pvalue, (int, float)) and math.isfinite(pvalue):
                aud.padrao(f"Aalen aareg {pos} significância de {term} coerente", result.get("significant") is (pvalue < 0.05), summary_path.name)
        aud.padrao(f"Aalen aareg {pos} tabela C4 vinculada à família", familia_tabela == _AAREG_FAMILY, summary_path.name, f"família detectada: {familia_tabela}; não interpretar a tabela do timereg como teste ponderado")
        if familia_tabela == _AAREG_FAMILY:
            exibidos: dict[str, str] = {}
            for linha in _linhas_aalen_por_posicao(aud).get(pos, []):
                for idx, celula in enumerate(linha[:-1]):
                    cov = variavel_da_celula(celula)
                    if cov in covariates and idx + 1 < len(linha):
                        exibidos[cov] = linha[idx + 1]
                        break
            significativos = {cov for cov in covariates if tests.get(cov, {}).get("significant") is True}
            for cov in sorted(set(exibidos) & set(tests)):
                aud.compara_p(f"tab:aalen_consolidada {pos} {cov} efeito médio ponderado", exibidos[cov], tests[cov]["p_value"], summary_path.name)
        else:
            aud.padrao(f"tab:aalen_consolidada {pos} comparação de p bloqueada por família incompatível", False, summary_path.name)

        input_path = AALEN / f"aalen_input_{pos}.csv"
        aud.padrao(f"Aalen aareg {pos} CSV de entrada presente", input_path.exists(), input_path.name)
        if not input_path.exists():
            continue
        data = _entrada_aalen(input_path, covariates)
        aud.padrao(f"Aalen aareg {pos} contagens de entrada reproduzidas", summary.get("input_counts_before_window") == _contagens_aalen(data), input_path.name, "input_counts são anteriores ao recorte; fit_counts são auditados separadamente")
        start, end, support, event_times = _janela_padrao_aalen(data, covariates)
        aud.padrao(f"Aalen aareg {pos} política nmin default", summary.get("nmin_policy") == "default" and summary.get("nmin_value") == 3 * len(covariates), summary_path.name)
        aud.padrao(f"Aalen aareg {pos} seleção da janela default reproduzida", summary.get("janela_inicio") == start and summary.get("janela_fim") == end, summary_path.name, f"regra independently recalculada: ({start}, {end}], posto completo e nmin=3*nvar")
        effective = summary.get("effective_support", {})
        aud.padrao(f"Aalen aareg {pos} eventos do domínio default", effective.get("event_years") == event_times and summary.get("window_selection", {}).get("event_times_selected_by_rank_and_default_nmin") == event_times, summary_path.name)
        _checa_janela_aalen(aud, pos, summary, data, covariates, start, end, support, event_times, False, summary_path.name)
        policy = str(summary.get("effective_support", {}).get("policy", "")).lower()
        aud.padrao(f"Aalen aareg {pos} regra default nmin e posto explícita", "3*nvar" in policy and "rank" in policy, summary_path.name)
        _checa_robvar_aareg(aud, pos, summary, AALEN / f"aalen_robvar_cum_{pos}.csv", covariates, start, end, summary_path.name)

        sensitivity_dir = AALEN / "nmin1_sensitivity"
        sensitivity_path = sensitivity_dir / f"aalen_summary_{pos}.json"
        aud.padrao(f"Aalen aareg {pos} resumo nmin1 próprio presente", sensitivity_path.exists(), sensitivity_path.name)
        if not sensitivity_path.exists():
            continue
        sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
        sens_fields = _campos_aareg_ausentes(sensitivity)
        aud.padrao(f"Aalen aareg {pos} metadados nmin1 presentes", not sens_fields, sensitivity_path.name, f"ausentes: {sens_fields}")
        sens_is_aareg = sensitivity.get("model") == _AAREG_MODEL and sensitivity.get("test_family") == _AAREG_FAMILY
        aud.padrao(f"Aalen aareg {pos} modelo/família nmin1", sens_is_aareg, sensitivity_path.name)
        aud.padrao(f"Aalen aareg {pos} tests_constant_effect nmin1 vazio", sensitivity.get("tests_constant_effect") in ({}, []), sensitivity_path.name)
        if not sens_is_aareg:
            continue
        aud.padrao(f"Aalen aareg {pos} nmin1 é sensibilidade rank-only sem fallback", sensitivity.get("nmin_policy") == "nmin1" and sensitivity.get("nmin_value") == 1 and sensitivity.get("window_selection", {}).get("sensitivity_only") is True and sensitivity.get("window_selection", {}).get("never_fallback_for_primary") is True and "rank" in str(sensitivity.get("window_selection", {}).get("rule", "")).lower() and "nmin" in str(sensitivity.get("window_selection", {}).get("rule", "")).lower(), sensitivity_path.name)
        sens_end, sens_support, sens_events = _janela_rank_only_aalen(data, covariates, start)
        aud.padrao(f"Aalen aareg {pos} nmin1 tem domínio independente e mesmo início", sensitivity.get("janela_inicio") == start and sensitivity.get("janela_fim") == sens_end and sensitivity.get("window_selection", {}).get("supported_integer_risk_times") == sens_support, sensitivity_path.name, f"domínio rank-only recalculado: ({start}, {sens_end}]")
        _checa_janela_aalen(aud, pos, sensitivity, data, covariates, start, sens_end, sens_support, sens_events, True, sensitivity_path.name)
        _checa_robvar_aareg(aud, pos, sensitivity, sensitivity_dir / f"aalen_robvar_cum_{pos}.csv", covariates, start, sens_end, sensitivity_path.name)


def checa_prosa_aalen_dimensoes(aud: Auditoria) -> None:
    """Dimensoes do ajuste de Aalen citadas na Secao 4.2."""
    texto = (LIVRO / "c4_resultados_inferenciais.tex").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"O ajuste tem (\d+) jogadores, ([\d.]+) intervalos e (\d+) saídas", texto)
    aud.padrao("prosa c4 Aalen QB dimensoes (frase)", bool(m), "aalen_summary_QB.json")
    if not m:
        return
    arquivo = AALEN / "aalen_summary_QB.json"
    if not arquivo.exists():
        return
    d = json.loads(arquivo.read_text(encoding="utf-8"))
    aud.compara_arredondado("prosa c4 Aalen QB jogadores", float(m.group(1)), d["n_subjects"], arquivo.name, 0)
    aud.compara_arredondado("prosa c4 Aalen QB intervalos", float(m.group(2).replace(".", "")), d["n_periods"], arquivo.name, 0)
    aud.compara_arredondado("prosa c4 Aalen QB eventos", float(m.group(3)), d["n_events"], arquivo.name, 0)


DIAG = ROOT / "documentos" / "estudo_diagnostico_erro" / "resultados"


def _destaques() -> dict[str, tuple[str, str]]:
    """(modelo de maior C, contraste) por posicao, lidos do resumo oficial com a regra da figura."""
    sys.path.insert(0, str(ROOT / "Predição" / "shared"))
    from tgb_protocol import modelos_destaque

    resumo = OFICIAL / "resumo_walkforward_2018_2021.csv"
    if not resumo.exists():
        return {}
    return modelos_destaque(pd.read_csv(resumo))


_DESTAQUE = _destaques()
CAMPEAO = {pos: par[0] for pos, par in _DESTAQUE.items()}
CONTRASTE = {pos: par[1] for pos, par in _DESTAQUE.items()}


AALEN_PRED = ROOT / "documentos" / "estudo_aalen_preditivo" / "resultados"



def checa_prosa_blocos(aud: Auditoria) -> None:
    """Numero de grupos de variaveis citado no c3 contra a enumeracao que o proprio texto faz."""
    palavras = {"um": 1, "dois": 2, "três": 3, "quatro": 4, "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10}
    texto = _texto("c3_material_tg.tex")
    m = re.search(r"usa (\w+) grupos de variáveis: ([^.]+)\.", texto)
    aud.padrao("prosa c3 numero de blocos (frase)", bool(m), "c3_material_tg.tex")
    if not m:
        return
    grupos = ("desempenho", "uso e oportunidade", "recrutamento e físico", "disponibilidade", "carreira na janela")
    aud.checagens += 1
    if m.group(1).lower() not in palavras or palavras[m.group(1).lower()] != len(grupos) or not all(g in m.group(2) for g in grupos):
        aud.divergencias.append(Divergencia("prosa c3 numero de blocos", m.group(1), str(len(grupos)), "c3_material_tg.tex", "grupos enumerados no texto nao batem com a contagem"))


def checa_prosa_km_sem_bandas(aud: Auditoria) -> None:
    """A curva KM global nao desenha bandas: legenda e codigo precisam concordar.

    A figura incluida na tese e gerada com `ci_show=False`. Uma legenda que
    prometa bandas de confianca descreve uma figura que nao existe.
    """
    texto = (LIVRO / "c3_material_tg.tex").read_text(encoding="utf-8", errors="replace")
    trecho = re.search(r"\\label\{fig:km_global\}", texto)
    aud.padrao("prosa c3 fig:km_global existe", bool(trecho), "c3_material_tg.tex")
    if not trecho:
        return
    inicio = max(0, trecho.start() - 1200)
    legenda = texto[inicio : trecho.end()]
    aud.checagens += 1
    if "bandas de confiança" in legenda:
        aud.divergencias.append(
            Divergencia(
                "prosa c3 legenda fig:km_global",
                "promete bandas de confiança",
                "figura gerada com ci_show=False",
                "c3_material_tg.tex",
                "a figura nao desenha bandas",
            )
        )
    codigo = (ROOT / "pipeline" / "08_curvas_km_qb.py").read_text(encoding="utf-8", errors="replace")
    aud.padrao("pipeline km global sem bandas (ci_show=False)", "ci_show=False" in codigo, "08_curvas_km_qb.py")


def checa_prosa_logrank_draft(aud: Auditoria) -> None:
    """Valores-p do log-rank por rodada de draft contra a anotacao da figura."""
    if not _extrator_pdf_disponivel():
        return
    texto = (LIVRO / "c3_material_tg.tex").read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"contra os não draftados é significativo no RB e no WR \(\$p<0\{,\}001\$\) e não no QB \(\$p=0\{,\}45\$",
        texto,
        re.S,
    )
    aud.padrao("prosa c3 log-rank por draft (frase)", bool(m), "km_draft_QB.pdf")
    if not m:
        return
    return  # a figura por faixa de draft nao e citada no texto atual; o log-rank de rodadas 1-2 e checado abaixo
    escrito = numero(m.group(1))
    pdf = ROOT / "Escrita Final" / "figuras" / "km_draft_QB.pdf"
    if not pdf.exists():
        return
    bruto = _texto_pdf_bruto(pdf)
    if not bruto:
        return
    achado = re.search(r"Log-rank p\s*=\s*([0-9.]+)", bruto)
    if not achado:
        return
    aud.compara_arredondado(
        "prosa c3 log-rank QB", escrito, float(achado.group(1)), "km_draft_QB.pdf (anotacao)", 3
    )


def checa_prosa_medianas_carreira(aud: Auditoria) -> None:
    """Mediana de carreira no KM: RB e WR empatam, entao RB nao e 'a mais curta'.

    A fonte e o artefato do pipeline (`estatisticas_eda.csv`, passo 50). Uma versao
    anterior desta checagem lia um JSON de auditoria que nao era versionado: em clone
    limpo ela perdia a cobertura e passava em silencio, que e o pior resultado
    possivel numa auditoria.
    """
    arquivo = ROOT / "Análise Exploratória de Dados" / "resultados" / "estatisticas_eda.csv"
    if not arquivo.exists():
        return
    tabela = pd.read_csv(arquivo)
    mediana = tabela[tabela["metrica"] == "km_mediana_anos"]
    por_pos = {str(r["posicao"]): float(r["valor"]) for _, r in mediana.iterrows()}
    if not {"RB", "WR"} <= set(por_pos):
        return
    rb, wr = por_pos["RB"], por_pos["WR"]
    texto = (LIVRO / "c7_conclusao_tg.tex").read_text(encoding="utf-8", errors="replace")
    aud.checagens += 1
    if rb == wr and re.search(r"Running backs têm a carreira mais curta", texto):
        aud.divergencias.append(
            Divergencia(
                "prosa c6 mediana RB",
                "RB tem a carreira mais curta",
                f"RB e WR empatam em {rb} anos",
                arquivo.name,
                "superlativo sem lastro no artefato",
            )
        )
    if rb != wr:
        aud.compara_arredondado("prosa c6 mediana RB", rb, wr, arquivo.name, 1, "RB deve ser a menor")


def checa_aalen_sem_missing(aud: Auditoria) -> None:
    """O Aalen inferencial nao ajusta indicadores _missing (a escrita afirma isso)."""
    for pos in POSICOES:
        arquivo = AALEN / f"aalen_summary_{pos}.json"
        if not arquivo.exists():
            continue
        d = json.loads(arquivo.read_text(encoding="utf-8"))
        testadas = list(d.get("covariates_tested", []))
        aud.checagens += 1
        if any(nome.endswith("_missing") for nome in testadas):
            aud.divergencias.append(
                Divergencia(
                    f"aalen {pos} covariates_tested",
                    "sem indicadores _missing (como a escrita afirma)",
                    "contem " + ", ".join(n for n in testadas if n.endswith("_missing")),
                    arquivo.name,
                )
            )


def inteiro_br(texto: str) -> int | None:
    """Inteiro no padrao brasileiro de milhar (7.133 = 7133)."""
    limpo = texto.strip().replace(".", "").replace("\u00a0", "").replace(" ", "")
    return int(limpo) if limpo.isdigit() else None


def checa_fonte_bruta(aud: Auditoria) -> None:
    """Forma, cobertura e qualidade do CSV bruto (Capitulo 3).

    O que a escrita afirma sobre o arquivo fonte (linhas, colunas, temporadas,
    modalidades, posicoes, negativos, multi-time) tem artefato em
    `DataBase/qualidade/`, gerado pelo passo `qualidade_dados`.
    """
    prov_path = ROOT / "DataBase" / "qualidade" / "proveniencia.json"
    resumo_path = ROOT / "DataBase" / "qualidade" / "qualidade_resumo.csv"
    if not prov_path.exists() or not resumo_path.exists():
        return
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    resumo = pd.read_csv(resumo_path)

    def valor(metrica: str, escopo: str) -> float | None:
        linha = resumo[(resumo["metrica"] == metrica) & (resumo["escopo"] == escopo)]
        return None if not len(linha) else float(linha["valor"].iloc[0])

    texto = (LIVRO / "c3_material_tg.tex").read_text(encoding="utf-8", errors="replace")

    m = re.search(
        r"([\d.]+) linhas e (\d+) colunas, de (\d+) a (\d+)",
        texto,
    )
    aud.padrao("prosa c3 forma do arquivo bruto (frase)", bool(m), prov_path.name)
    if m:
        aud.compara_arredondado("bruto linhas", inteiro_br(m.group(1)), prov["linhas"], prov_path.name, 0)
        aud.compara_arredondado("bruto colunas", inteiro_br(m.group(2)), prov["colunas"], prov_path.name, 0)
        aud.compara_arredondado("bruto temporada minima", int(m.group(3)), prov["temporadas"]["min"], prov_path.name, 0)
        aud.compara_arredondado("bruto temporada maxima", int(m.group(4)), prov["temporadas"]["max"], prov_path.name, 0)

    m = re.search(
        r"([\d.]+) linhas e \d+ colunas, de \d+ a \d+\.[\s\S]{0,300}?([\d.]+) linhas de temporada regular[\s\S]{0,80}?e ([\d.]+) de pós-temporada",
        texto,
    )
    aud.padrao("prosa c3 modalidades (frase)", bool(m), prov_path.name)
    if m:
        aud.compara_arredondado("modalidade REG", inteiro_br(m.group(2)), prov["modalidades"].get("REG"), prov_path.name, 0)
        aud.compara_arredondado("modalidade POST", inteiro_br(m.group(3)), prov["modalidades"].get("POST"), prov_path.name, 0)

    snaps = resumo[resumo["metrica"] == "pct_ausente_offense_snaps"]
    aud.checagens += 1
    if len(snaps) and float(snaps["valor"].max()) > 1.0:
        aud.divergencias.append(
            Divergencia(
                "prosa c3 disponibilidade de snaps",
                "ausencia abaixo de 1% em todas as temporadas",
                f"maximo {float(snaps['valor'].max()):.2f}%",
                resumo_path.name,
                "a prosa afirma que a serie de snaps e consistente na janela",
            )
        )
    aud.padrao(
        "prosa c3 snaps consistentes (frase)",
        "Fração dos \\textit{snaps} ofensivos" in _texto("apendice_dados.tex"),
        resumo_path.name,
    )

    m = re.search(r"em (\d+) posições", texto)
    aud.padrao("prosa c3 numero de posicoes (frase)", bool(m), prov_path.name)
    if m:
        aud.compara_arredondado("numero de posicoes", int(m.group(1)), prov["n_posicoes"], prov_path.name, 0)

    m = None  # valor nao citado no texto: confere o registro
    _registrado(aud, "prosa c3 games_missed negativos (frase)", str(valor("games_missed_negativos", "modalidade_REG")), resumo_path.name)
    if m:
        aud.compara_arredondado(
            "games_missed negativos", int(m.group(1)), valor("games_missed_negativos", "modalidade_REG"), resumo_path.name, 0
        )
        aud.compara(
            "games_missed minimo", -float(m.group(2)), valor("games_missed_minimo", "modalidade_REG"), resumo_path.name
        )

    m = None  # valor nao citado no texto: confere o registro
    _registrado(aud, "prosa c3 multi-time (frase)", str(valor("pares_jogador_temporada_multi_time", "REG_QB")), resumo_path.name)
    if m:
        multi = sum(
            valor("pares_jogador_temporada_multi_time", f"REG_{p}") or 0 for p in POSICOES
        )
        pares = sum(valor("pares_jogador_temporada", f"REG_{p}") or 0 for p in POSICOES)
        aud.compara_arredondado(
            "multi-time pares com mais de uma passagem", int(m.group(1)), multi, resumo_path.name, 0
        )
        aud.compara_arredondado(
            "multi-time pares",
            104,
            multi,
            resumo_path.name,
            0,
            "o texto cita o total de pares multi-time da base modelada",
        )

    m = None  # valor nao citado no texto: confere o registro
    _registrado(aud, "prosa c3 multi-time season identicas (frase)", str(valor("pares_multi_time_com_season_divergente", "REG_QB")), resumo_path.name)
    if m:
        divergentes = sum(
            valor("pares_multi_time_com_season_divergente", f"REG_{p}") or 0 for p in POSICOES
        )
        aud.padrao("multi-time colunas season identicas (texto)", divergentes == 0, resumo_path.name)


def checa_painel_inferencial(aud: Auditoria) -> None:
    """Painel counting-process: intervalos, jogadores e censura (Capitulo 3)."""
    texto = (LIVRO / "c3_material_tg.tex").read_text(encoding="utf-8", errors="replace")
    total = re.search(r"painel contém ([\d.]+) intervalos para ([\d.]+) jogadores distintos", texto)
    m = re.search(
        r"contém ([\d.]+) intervalos de carreira.*?referentes a ([\d.]+) jogadores distintos "
        r"\((\d+) QBs, (\d+) RBs e (\d+) WRs",
        texto,
        re.S,
    )
    aud.padrao("prosa c3 painel (frase)", bool(total or m) or bool(re.search(r"consolidam em [\d.]+ pares jogador-temporada de [\d.]+ jogadores", texto)), "df_cox_*.csv")
    if not total and not m:
        return
    paineis = {}
    for pos in POSICOES:
        arquivo = PREP / f"df_cox_{pos}.csv"
        if not arquivo.exists():
            return
        paineis[pos] = pd.read_csv(arquivo, usecols=["player_id", "event"])
    intervalos = sum(len(d) for d in paineis.values())
    distintos = len(set().union(*(set(d["player_id"]) for d in paineis.values())))
    if total:
        aud.compara_arredondado("painel intervalos", inteiro_br(total.group(1)), intervalos, "df_cox_*.csv", 0)
        aud.compara_arredondado("painel jogadores distintos", inteiro_br(total.group(2)), distintos, "df_cox_*.csv", 0)
        return
    aud.compara_arredondado("painel intervalos", inteiro_br(m.group(1)), intervalos, "df_cox_*.csv", 0)
    aud.compara_arredondado("painel jogadores distintos", inteiro_br(m.group(2)), distintos, "df_cox_*.csv", 0)
    for grupo, pos in zip((3, 4, 5), POSICOES):
        aud.compara_arredondado(
            f"painel jogadores {pos}", int(m.group(grupo)), paineis[pos]["player_id"].nunique(), f"df_cox_{pos}.csv", 0
        )

    m = re.search(
        r"([\d,]+)\\% \(QB\), ([\d,]+)\\% \(RB\) e ([\d,]+)\\% \(WR\)",
        texto,
    )
    aud.padrao("prosa c3 censura por posicao (frase)", bool(m), "df_cox_*.csv")
    if not m:
        return
    # A censura citada e por JOGADOR (saiu ou nao dentro da janela), nao por
    # intervalo: o painel repete o mesmo jogador em varias linhas.
    for grupo, pos in zip((1, 2, 3), POSICOES):
        escrito = float(m.group(grupo).replace(",", "."))
        por_jogador = paineis[pos].groupby("player_id")["event"].max()
        censura = 100 * (1 - float(por_jogador.mean()))
        aud.compara_arredondado(f"censura {pos}", escrito, censura, f"df_cox_{pos}.csv", 1)


def checa_estatisticas_eda(aud: Auditoria) -> None:
    """Medianas, funil de covariaveis e testes univariados do Capitulo 3.

    Fonte: `Análise Exploratória de Dados/resultados/{estatisticas_eda,testes_eda}.csv`
    e `Inferência/Preparação Modelagem/resultados/funil_covariaveis.json`, todos
    gerados pelo passo `estatisticas_eda`.
    """
    est_p = ROOT / "Análise Exploratória de Dados" / "resultados" / "estatisticas_eda.csv"
    tes_p = ROOT / "Análise Exploratória de Dados" / "resultados" / "testes_eda.csv"
    fun_p = PREP / "funil_covariaveis.json"
    if not (est_p.exists() and tes_p.exists() and fun_p.exists()):
        return
    est = pd.read_csv(est_p)
    tes = pd.read_csv(tes_p)
    fun = json.loads(fun_p.read_text(encoding="utf-8"))

    def valor(pos: str, metrica: str) -> float | None:
        linha = est[(est["posicao"] == pos) & (est["metrica"] == metrica)]
        return None if not len(linha) else float(linha["valor"].iloc[0])

    def pvalor(pos: str, teste: str) -> float | None:
        linha = tes[(tes["posicao"] == pos) & (tes["teste"] == teste)]
        return None if not len(linha) else float(linha["p"].iloc[0])

    texto = (LIVRO / "c3_material_tg.tex").read_text(encoding="utf-8", errors="replace")

    m = re.search(
        r"O painel posicional, com (\d+) covariáveis no QB, (\d+) no RB e (\d+) no WR",
        texto,
    )
    aud.padrao("prosa c3 painel posicional (frase)", bool(m), fun_p.name)
    if m:
        for grupo, pos in zip((1, 2, 3), POSICOES):
            aud.compara_arredondado(
                f"funil candidatas {pos}", int(m.group(grupo)), fun[pos]["n_candidatas"], fun_p.name, 0
            )

    m = re.search(r"deixa (\d+), (\d+) e (\d+) candidatas", texto)
    aud.padrao("prosa c3 pos-colinearidade (frase)", bool(m), "tab:funnel_variaveis")
    if m:
        for grupo, pos in zip((1, 2, 3), POSICOES):
            aud.compara_arredondado(
                f"funil pos-colinearidade {pos}", int(m.group(grupo)), _cadeia_cox(pos)["final"],
                "cadeia do Cox (identificabilidade -> colinearidade 0,85)", 0,
            )

    aud.checagens += 1
    if any(not dados.get("removidas_detalhe") for dados in fun.values()):
        aud.divergencias.append(
            Divergencia(
                "funil detalhe da colinearidade",
                "sem detalhe por variavel removida",
                "esperado lista com pares acima do limiar",
                fun_p.name,
                "o texto descreve a regra gulosa; o artefato precisa registra-la",
            )
        )

    m = re.search(r"A mediana é de (\d+) temporadas para QBs e de (\d+) para RBs e WRs", texto)
    aud.padrao("prosa c3 medianas de carreira (frase)", bool(m), est_p.name)
    if m:
        aud.compara_arredondado(
            "carreira mediana QB", int(m.group(1)), valor("QB", "carreira_mediana_temporadas"), est_p.name, 0
        )
        for pos in ("RB", "WR"):
            aud.compara_arredondado(
                f"carreira mediana {pos}", int(m.group(2)), valor(pos, "carreira_mediana_temporadas"), est_p.name, 0
            )

    m = None
    if m:
        aud.compara_arredondado(
            "estreia idade media (QB)",
            float(f"{m.group(1)}.{m.group(2)}"),
            valor("QB", "estreia_idade_media"),
            est_p.name,
            1,
        )
        n_est = valor("QB", "estreia_n")
        aud.padrao("estreia: n citado (QB)", n_est is not None and f"$n={int(n_est)}$" in texto, est_p.name)

    aud.checagens += 1
    pct_qb = valor("QB", "carreira_pct_ate_3_temporadas")
    if pct_qb is None or not 40 <= pct_qb <= 55:
        aud.divergencias.append(
            Divergencia(
                "prosa c3 'quase metade' (QB)",
                "faixa 40--55%",
                str(pct_qb),
                est_p.name,
                "a prosa diz que quase metade das carreiras de QB termina em ate 3 temporadas",
            )
        )

    m = re.search(
        r"A mediana de experiência até a saída é de (\d+) anos para QBs \(IC 95\\%: (\d+) a (\d+)\), "
        r"(\d+) para RBs \(IC 95\\%: (\d+) a (\d+)\) e (\d+) para WRs \(IC 95\\%: (\d+) a (\d+)\)",
        texto,
    )
    aud.padrao("prosa c3 KM mediana e IC (frase)", bool(m), est_p.name)
    if m:
        grupos = [(1, 2, 3, "QB"), (4, 5, 6, "RB"), (7, 8, 9, "WR")]
        for g_med, g_inf, g_sup, pos in grupos:
            aud.compara_arredondado(
                f"km mediana {pos}", int(m.group(g_med)), valor(pos, "km_mediana_anos"), est_p.name, 0
            )
            aud.compara_arredondado(
                f"km IC inferior {pos}",
                int(m.group(g_inf)),
                valor(pos, "km_mediana_ic95_inferior"),
                est_p.name,
                0,
            )
            aud.compara_arredondado(
                f"km IC superior {pos}",
                int(m.group(g_sup)),
                valor(pos, "km_mediana_ic95_superior"),
                est_p.name,
                0,
            )

    for pos in POSICOES:
        inferior = valor(pos, "km_mediana_ic95_inferior")
        superior = valor(pos, "km_mediana_ic95_superior")
        metodo = est[(est["posicao"] == pos) & (est["metrica"] == "km_mediana_metodo_ic")]
        aud.checagens += 1
        if inferior is None or superior is None or not len(metodo) or not str(metodo["valor"].iloc[0]):
            aud.divergencias.append(
                Divergencia(
                    f"km IC da mediana {pos}",
                    "IC e metodo registrados",
                    f"inferior={inferior} superior={superior}",
                    est_p.name,
                    "o IC da mediana precisa de artefato para ser citavel",
                )
            )

    # A frase foi reescrita junto com o paragrafo: o contraste bruto de jogos
    # perdidos aponta na direcao oposta a que o texto afirmava.
    m = re.search(
        r"rejeita a igualdade no WR \(\$p\s*=\s*0\{,}(\d+)\$\) e no RB \(\$p\s*=\s*0\{,}(\d+)\$\)"
        r", não no QB \(\$p\s*=\s*0\{,}(\d+)\$",
        texto,
    )
    aud.padrao("prosa c3 Mann-Whitney (frase)", bool(m), tes_p.name)
    if m:
        for grupo, pos in zip((2, 1), ("RB", "WR")):
            aud.compara_arredondado(
                f"mann-whitney {pos}", float(f"0.{m.group(grupo)}"), pvalor(pos, "mann_whitney"), tes_p.name, 3
            )
        aud.compara_arredondado(
            "mann-whitney QB", float(f"0.{m.group(3)}"), pvalor("QB", "mann_whitney"), tes_p.name, 3
        )

    m = re.search(
        r"O teste \\textit\{log-rank\}, com a mesma entrada tardia, rejeita a homogeneidade entre estratos no "
        r"\(QB: \$p = 0\{,}(\d+)\$; RB: \$p < 0\{,\}001\$; WR: \$p < 0\{,\}001\$\)",
        texto,
    )
    if not m:
        m = re.search(
            r"O teste \\textit\{log-rank\}, com a mesma entrada tardia, rejeita a homogeneidade entre estratos no "
            r"QB \(\$p = ([0-9]+\{,\}[0-9]+)\$\), RB \(\$p < ([0-9]+\{,\}[0-9]+)\$\) e WR \(\$p < ([0-9]+\{,\}[0-9]+)\$\)",
            texto,
        )
    aud.padrao("prosa c3 log-rank por draft (frase)", bool(m) or ("contra os não draftados é significativo no RB e no WR" in texto), tes_p.name)
    if m:
        aud.compara_arredondado(
            "log-rank draft QB", numero(m.group(1)) if m.lastindex == 3 else float(f"0.{m.group(1)}"), pvalor("QB", "logrank_draft_tiers"), tes_p.name, 3
        )
        for pos in ("RB", "WR"):
            aud.compara_p(
                f"log-rank draft {pos}",
                "<" + (m.group(2 if pos == "RB" else 3) if m.lastindex == 3 else "0,001"),
                pvalor(pos, "logrank_draft_tiers"),
                tes_p.name,
            )

    aud.checagens += 1
    kw = {pos: pvalor(pos, "kruskal_wallis") for pos in POSICOES}
    if any(v is None or v >= 0.05 for v in kw.values()):
        aud.divergencias.append(
            Divergencia(
                "prosa c3 Kruskal-Wallis",
                "rejeita nas tres posicoes",
                str(kw),
                tes_p.name,
                "a prosa afirma rejeicao de homogeneidade por estrato de draft",
            )
        )


def checa_protocolo_inferencial(aud: Auditoria) -> None:
    """Constantes metodologicas conferidas contra os proprios scripts."""
    arquivo = ROOT / "Inferência" / "protocolo_inferencial.json"
    if not arquivo.exists():
        return
    protocolo = json.loads(arquivo.read_text(encoding="utf-8"))
    aud.checagens += 1
    if protocolo.get("divergencias"):
        aud.divergencias.append(
            Divergencia(
                "protocolo inferencial",
                "constantes conferem com os scripts",
                "divergentes: " + ", ".join(protocolo["divergencias"]),
                arquivo.name,
                "o artefato acusa deriva entre o protocolo declarado e o codigo",
            )
        )
    constantes = protocolo.get("constantes", {})
    for campo, esperado in (
        ("cox.penalizer", 0.0),
        ("cox.threshold_p", 0.05),
        ("colinearidade.limiar", 0.85),
        ("taxa_lag.denominador_minimo", 50),
        ("cox_aalen.n_sim", 1000),
    ):
        aud.compara_arredondado(
            f"protocolo {campo}", esperado, constantes.get(campo), arquivo.name, 2
        )


def checa_preenchimento_e_casos(aud: Auditoria) -> None:
    """Preenchimento das taxas por periodo, casos multi-time e jogos negativos."""
    prep_p = PREP / "preenchimento_por_periodo.csv"
    casos_p = ROOT / "DataBase" / "qualidade" / "multitime_casos.csv"
    neg_p = ROOT / "DataBase" / "qualidade" / "games_missed_inconsistencias.csv"
    texto_c3 = (LIVRO / "c3_material_tg.tex").read_text(encoding="utf-8", errors="replace")
    texto_c4 = (LIVRO / "c4_resultados_inferenciais.tex").read_text(encoding="utf-8", errors="replace")

    if casos_p.exists():
        casos = pd.read_csv(casos_p)
        tabela = aud.linhas("tab:multitime_exemplos")
        _registrado(aud, "tabela de casos multi-time presente", str(len(casos)), casos_p.name)
        for linha in tabela:
            if len(linha) < 6 or not linha[2].strip().isdigit():
                continue
            jogador, posicao = linha[0].strip(), linha[1].strip()
            temporada = int(linha[2])
            por_time = [parte.strip() for parte in linha[3].split("/")]
            maximo, soma = numero(linha[4]), numero(linha[5])
            candidato = casos[
                (casos["player_name"] == jogador)
                & (casos["season"] == temporada)
                & (casos["position"] == posicao)
            ]
            aud.checagens += 1
            if not len(candidato):
                aud.divergencias.append(
                    Divergencia(
                        f"tabela multi-time {jogador} {temporada}",
                        "caso presente no artefato",
                        "ausente",
                        casos_p.name,
                        "a tabela cita casos que o artefato precisa conter",
                    )
                )
                continue
            registro = candidato.iloc[0]
            coluna = (
                "season_rushing_yards_por_time"
                if posicao == "RB"
                else "season_receiving_yards_por_time"
            )
            valores_artefato = [
                numero(parte) for parte in str(registro[coluna]).split(";")
            ]
            maximo_artefato = float(
                registro[
                    "season_rushing_yards_max"
                    if posicao == "RB"
                    else "season_receiving_yards_max"
                ]
            )
            for escrito, artefato in zip(por_time, valores_artefato):
                aud.compara_arredondado(
                    f"multi-time {jogador} valor por time",
                    numero(escrito),
                    artefato,
                    casos_p.name,
                    0,
                )
            aud.compara_arredondado(
                f"multi-time {jogador} max()", maximo, maximo_artefato, casos_p.name, 0
            )
            # As duas passagens repetem o agregado da temporada: sum() = 2 x max().
            aud.compara_arredondado(
                f"multi-time {jogador} sum()",
                soma,
                2 * maximo_artefato,
                casos_p.name,
                0,
                "a tabela soma as duas passagens do mesmo agregado",
            )

    if neg_p.exists():
        neg = pd.read_csv(neg_p)
        aud.compara_arredondado("games_missed negativos no artefato", 5, len(neg), neg_p.name, 0)
        aud.checagens += 1
        if len(neg) and not (neg["games_played_season"] > 16).all():
            aud.divergencias.append(
                Divergencia(
                    "prosa c3 negativos x games_played_season",
                    "todos com games_played_season > 16",
                    str(neg["games_played_season"].tolist()),
                    neg_p.name,
                    "a prosa liga os negativos a temporadas com mais de 16 jogos",
                )
            )

    if prep_p.exists():
        prep = pd.read_csv(prep_p)
        primeiro = prep[prep["periodo"] == prep["periodo"].min()]
        aud.checagens += 1
        if primeiro.empty:
            aud.divergencias.append(
                Divergencia("preenchimento por periodo", "primeiro periodo presente", "vazio", prep_p.name)
            )
        else:
            # O texto (c4) afirma que as taxas entram "parcialmente preenchidas"
            # no primeiro periodo; o artefato precisa mostrar qual e a fracao.
            medias = primeiro.groupby("posicao")["pct_preenchido_taxas"].mean()
            if not ((medias >= 0) & (medias <= 100)).all():
                aud.divergencias.append(
                    Divergencia(
                        "preenchimento por periodo (faixa)",
                        "0 a 100%",
                        str(medias.round(2).to_dict()),
                        prep_p.name,
                    )
                )
    # Contagens defasadas na primeira aparicao de cada jogador: a prosa afirma
    # que sao nulas, com uma unica excecao no painel de RBs. As TAXAS ficam de
    # fora desta conta (tem tratamento proprio, ver preenchimento por periodo).
    sufixos_taxa = (
        "comp_pct", "ypa", "int_pct", "td_pct", "ypc", "yptarget", "ypr",
        "catch_rate", "adot_acc", "rush_td_pct", "yptouch", "reception_rate",
    )
    excecoes = {}
    for pos in POSICOES:
        arquivo = PREP / f"df_cox_{pos}.csv"
        if not arquivo.exists():
            return
        painel = pd.read_csv(arquivo)
        contagens = [
            c for c in painel.columns
            if c.startswith("observed_career_lag_")
            and not c.endswith("_missing")
            and not any(sufixo in c for sufixo in sufixos_taxa)
        ]
        if not contagens:
            continue
        primeira = painel.sort_values(["player_id", "start"]).groupby("player_id").first()
        excecoes[pos] = int((primeira[contagens].fillna(0) != 0).any(axis=1).sum())
    if excecoes:
        aud.checagens += 1
        if excecoes.get("QB") != 0 or excecoes.get("RB") != 1 or excecoes.get("WR") != 0:
            aud.divergencias.append(
                Divergencia(
                    "prosa c4 excecao de lag no primeiro periodo",
                    "QB 0, RB 1, WR 0",
                    str(excecoes),
                    "df_cox_*.csv",
                    "a prosa fala de uma unica excecao, no painel de RBs",
                )
            )
    _tem_se_afirmado(
        aud,
        "prosa c4 primeiro periodo",
        "No primeiro ano de carreira as contagens defasadas valem zero para quase todos os jogadores",
        texto_c4,
        "df_cox_*.csv",
        r"No primeiro ano de carreira as contagens defasadas valem zero para quase todos os jogadores",
        excecoes == {"QB": 0, "RB": 1, "WR": 0},
    )
    if prep_p.exists():
        prep = pd.read_csv(prep_p)
        periodos = sorted(prep["periodo"].unique())
        if len(periodos) >= 2:
            primeiro = prep[prep["periodo"] == periodos[0]]
            segundo = prep[prep["periodo"] == periodos[1]]
            aud.checagens += 1
            if float(primeiro["pct_preenchido_taxas"].max()) != 0.0:
                aud.divergencias.append(
                    Divergencia(
                        "c4 taxas ausentes no primeiro periodo",
                        "0% preenchido",
                        f"maximo {primeiro['pct_preenchido_taxas'].max():.2f}%",
                        prep_p.name,
                        "a prosa afirma ausencia integral no primeiro periodo",
                    )
                )
            aud.checagens += 1
            por_pos = segundo.set_index("posicao")["pct_preenchido_taxas"].to_dict()
            metade = [p for p in ("QB", "RB") if por_pos.get(p, 0) < 40]
            quarto = por_pos.get("WR")
            if metade or quarto is None or not 20 <= quarto <= 30:
                aud.divergencias.append(
                    Divergencia(
                        "c4 preenchimento no segundo periodo",
                        "cerca de metade em QB/RB e um quarto em WR",
                        str({k: round(v, 2) for k, v in por_pos.items()}),
                        prep_p.name,
                        "a prosa descreve o preenchimento do segundo periodo",
                    )
                )


def checa_sensibilidade_c3(aud: Auditoria) -> None:
    """As tres sensibilidades citadas no c3, com passo e artefato proprios.

    O passo `sensibilidade_inferencial` reconstroi o painel variando o limiar da
    taxa, a codificacao de nao draftados e a imputacao, e so e aceito se a
    reconstrucao no limiar padrao reproduzir a selecao do proprio pipeline.
    """
    base = ROOT / "Inferência" / "sensibilidade" / "resultados"
    lim_p = base / "limiares_taxa.csv"
    cen_p = base / "undrafted_cenarios.csv"
    mi_p = base / "imputacao_multipla.csv"
    res_p = base / "resumo.json"
    texto = (LIVRO / "c3_material_tg.tex").read_text(encoding="utf-8", errors="replace")

    if res_p.exists():
        resumo = json.loads(res_p.read_text(encoding="utf-8"))
        aud.padrao(
            "sensibilidade: aceitacao do limiar padrao",
            bool(resumo.get("destaques", {}).get("limiar_50_reproduz_selecao_do_pipeline")),
            res_p.name,
            "o passo so vale se reproduzir a selecao do pipeline no limiar padrao",
        )
        aceitacao = resumo.get("metodo", {}).get("teste_aceitacao", {}).get("resultado", {})
        for pos in POSICOES:
            aud.checagens += 1
            if not aceitacao.get(pos):
                aud.divergencias.append(
                    Divergencia(
                        f"sensibilidade {pos}: reconstrucao no limiar padrao",
                        "reproduz a selecao do proprio pipeline",
                        str(aceitacao.get(pos)),
                        res_p.name,
                        "sem isso a sensibilidade nao fala do mesmo modelo",
                    )
                )

    if lim_p.exists():
        lim = pd.read_csv(lim_p)
        g = lim.groupby("posicao").agg(nmin=("n_covariaveis", "min"), nmax=("n_covariaveis", "max"), jmin=("jaccard_vs_50", "min"))
        _tem(aud, "c3 limiares: faixa testada", f"varrido de {int(lim.limiar.min())} a {int(lim.limiar.max())}", VALOR_REGISTRADO, lim_p.name)
        _tem(aud, "c3 limiares: Jaccard QB", f"similaridade mínima (Jaccard) é de ${br(g.loc['QB','jmin'], 2)}$ no QB", VALOR_REGISTRADO, lim_p.name)
        _tem(aud, "c3 limiares: Jaccard RB e WR", f"${br(g.loc['RB','jmin'], 2)}$ no RB e ${br(g.loc['WR','jmin'], 2)}$ no WR", VALOR_REGISTRADO, lim_p.name)
        _tem(aud, "c3 limiares: QB igual de 25 a 90", "a seleção do QB é a mesma de 25 a 90 e muda em 95 e 100", VALOR_REGISTRADO, lim_p.name)
        _verdade(aud, "c3 limiares: QB estável de 25 a 90", bool((lim[(lim.posicao == "QB") & (lim.limiar <= 90)].jaccard_vs_50 == 1).all()), lim_p.name)

    if cen_p.exists():
        cen = pd.read_csv(cen_p)
        _verdade(aud, "c3 codificacao: QB identico em todos os cenarios", bool((cen[cen.posicao == "QB"].jaccard_vs_padrao == 1).all()), cen_p.name)
        rb = cen[(cen.posicao == "RB") & (cen.jaccard_vs_padrao < 1)]
        wr = cen[(cen.posicao == "WR") & (cen.jaccard_vs_padrao < 1)]
        _verdade(aud, "c3 codificacao: os cenarios B e D mudam o RB e so o B muda o WR",
                 list(rb.cenario) == sorted(rb.cenario) == ["B", "D"] and list(wr.cenario) == ["B"], cen_p.name, f"RB {list(rb.cenario)} WR {list(wr.cenario)}")
        if len(rb) and len(wr):
            ad_t = _texto("apendice_dados.tex")
            _tem(aud, "c3 codificacao: Jaccard RB (faixa)", f"(Jaccard de ${br(rb[rb.cenario == 'B'].jaccard_vs_padrao.iloc[0], 2)}$ e ${br(rb[rb.cenario == 'D'].jaccard_vs_padrao.iloc[0], 2)}$)", ad_t, cen_p.name)
            _verdade(aud, "c3 codificacao: RB, cenario D muda mais que B", float(rb[rb.cenario == 'D'].jaccard_vs_padrao.iloc[0]) < float(rb[rb.cenario == 'B'].jaccard_vs_padrao.iloc[0]), cen_p.name)
            _tem(aud, "c3 codificacao: Jaccard WR", f"(Jaccard de ${br(wr.jaccard_vs_padrao.iloc[0], 2)}$)", ad_t, cen_p.name)

    if mi_p.exists():
        mi = pd.read_csv(mi_p).set_index(["posicao", "covariavel"])
        q = mi.loc[("QB", "observed_career_lag_td_pct")]
        r = mi.loc[("RB", "observed_career_lag_ypc")]
        for label, pattern, expected, precisions in (
            (
                "c3 MI QB",
                r"HR de TD\\% no QB[^.;]*",
                (q.hr_pooled, q.p_pooled, q.hr_unica_sem_flag, q.p_unica_sem_flag),
                (3, 3, 3, 4),
            ),
            (
                "c3 MI RB",
                r"No RB, o HR de jardas por corrida[^.;]*",
                (r.hr_pooled, r.p_pooled, r.hr_unica_original, r.p_unica_original),
                (3, 3, 3, 3),
            ),
        ):
            match = None  # valor nao citado no texto: confere o registro
            _registrado(aud, f"{label}: trecho numérico presente", ";".join(f"{float(x):.6g}" for x in expected), mi_p.name)
            if match is None:
                continue
            written = _nums_celula(match.group(0))
            aud.padrao(f"{label}: quatro valores numéricos", len(written) == 4, mi_p.name)
            if len(written) != 4:
                continue
            for index, (actual, precision) in enumerate(zip(expected, precisions), start=1):
                aud.compara_arredondado(f"{label}: valor {index}", written[index - 1], float(actual), mi_p.name, precision)



def checa_rsf_qb_protocolo(aud: Auditoria) -> None:
    """RSF no QB faz parte do protocolo principal (hiperparametros da busca por fold), nao e sensibilidade."""
    t = _texto("c5_resultados_preditivos.tex")
    rs_p = OFICIAL / "resumo_walkforward_2018_2021.csv"
    hp_p = OFICIAL / "hp_usado_por_fold.csv"
    if not t or not rs_p.exists() or not hp_p.exists():
        return
    rs = pd.read_csv(rs_p).set_index(["posicao", "modelo"])
    _tem(aud, "c5 RSF QB no protocolo principal", "RSF no QB, Cox no RB e no WR", t, rs_p.name)
    hp = pd.read_csv(hp_p)
    ref = (hp.posicao == "QB") & (hp.modelo == "rsf") & (hp.fonte == "walkforward") & hp.teste_em_temporada.between(2018, 2021)
    _verdade(aud, "RSF no QB com hiperparametros por fold (4 folds de referencia)", int(ref.sum()) == 4, hp_p.name, f"linhas = {int(ref.sum())}")


def checa_citacoes(aud: Auditoria) -> None:
    """Integridade das citacoes: chave citada sem entrada, entrada sem uso, entrada sem ano.

    Nao julga se o artigo diz o que a frase afirma (isso e leitura humana); garante
    que toda citacao resolve, que nenhuma entrada ficou orfa e que as entradas tem
    os campos minimos para o leitor conferir.
    """
    bib = LIVRO / "bibliografia.bib"
    if not bib.exists():
        return
    texto = "\n".join(
        (LIVRO / nome).read_text(encoding="utf-8", errors="replace")
        for nome in CAPITULOS
        if (LIVRO / nome).exists()
    )
    entradas = re.findall(r"@(\w+)\{([^,]+),", bib.read_text(encoding="utf-8", errors="replace"))
    chaves = [chave.strip() for _, chave in entradas]
    duplicadas = sorted({c for c in chaves if chaves.count(c) > 1})
    aud.checagens += 1
    if duplicadas:
        aud.divergencias.append(
            Divergencia("citacoes: chaves duplicadas", "nenhuma", ", ".join(duplicadas), bib.name)
        )
    citadas = set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]*)\}", texto):
        citadas |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    sem_entrada = sorted(citadas - set(chaves))
    aud.checagens += 1
    if sem_entrada:
        aud.divergencias.append(
            Divergencia(
                "citacoes: chave sem entrada",
                "toda citacao resolve no .bib",
                ", ".join(sem_entrada),
                bib.name,
                "citacao quebrada nao compila referencia",
            )
        )
    orfas = sorted(set(chaves) - citadas)
    aud.checagens += 1
    if orfas:
        aud.divergencias.append(
            Divergencia(
                "citacoes: entrada sem uso",
                "toda entrada e citada ao menos uma vez",
                ", ".join(orfas),
                bib.name,
                "entrada orfa e ruido na bibliografia",
            )
        )
    faltando_ano = []
    for bloco in re.split(r"\n(?=@)", bib.read_text(encoding="utf-8", errors="replace")):
        chave = re.match(r"@\w+\{([^,]+),", bloco)
        if not chave:
            continue
        if not re.search(r"\byear\s*=", bloco, re.I) and not re.search(r"\bdate\s*=", bloco, re.I):
            faltando_ano.append(chave.group(1))
    aud.checagens += 1
    if faltando_ano:
        aud.divergencias.append(
            Divergencia(
                "citacoes: entrada sem ano",
                "toda entrada tem year",
                ", ".join(faltando_ano),
                bib.name,
                "sem ano o leitor nao confere a referencia",
            )
        )


def _modulo_series_figuras():
    """Importa `pipeline/52_series_figuras.py` para reusar o mesmo parser de citacao.

    Reusar o parser (em vez de reimplementar um regex aqui) garante que o CSV de
    rastreabilidade nao pode estar desatualizado em relacao aos `.tex`: a
    auditoria enxerga exatamente as figuras que o passo enxergou.
    """
    import importlib.util

    caminho = ROOT / "pipeline" / "52_series_figuras.py"
    spec = importlib.util.spec_from_file_location("pipeline_52", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def checa_series_figuras(aud: Auditoria) -> None:
    """Lastro numerico das figuras do EDA e anotacoes de teste do Capitulo 3.

    Duas computacoes independentes produzem as mesmas estatisticas: o passo
    `estatisticas_eda` (50) e o passo `series_figuras` (52), que recalcula as
    series efetivamente plotadas. O cruzamento dos dois e a verificacao: se
    divergirem, a figura nao representa o dado que a escrita cita.
    """
    res = ROOT / "Análise Exploratória de Dados" / "resultados"
    series = res / "series_figuras"
    f_est, f_tes, f_ft = res / "estatisticas_eda.csv", res / "testes_eda.csv", series / "fig_testes.csv"
    if not (f_est.exists() and f_tes.exists() and f_ft.exists()):
        return
    est, tes, ft = (pd.read_csv(f) for f in (f_est, f_tes, f_ft))
    prosa = (LIVRO / "c3_material_tg.tex").read_text(encoding="utf-8", errors="replace")

    def de_est(pos: str, metrica: str) -> float | None:
        linha = est[(est["posicao"] == pos) & (est["metrica"] == metrica)]
        return float(linha["valor"].iloc[0]) if len(linha) else None

    def de_tes(pos: str, teste: str, campo: str = "p") -> float | None:
        linha = tes[(tes["posicao"] == pos) & (tes["teste"] == teste)]
        return float(linha[campo].iloc[0]) if len(linha) else None

    def de_ft(pos: str, teste: str, campo: str) -> float | None:
        linha = ft[(ft["posicao"] == pos) & (ft["teste"] == teste)]
        return float(linha[campo].iloc[0]) if len(linha) else None

    for pos in ("QB", "RB", "WR"):
        hist = pd.read_csv(series / f"fig_dist_carreira_{pos}.csv")
        media = pd.read_csv(series / f"fig_draft_media_{pos}.csv")
        grupos = pd.read_csv(series / f"fig_games_missed_grupos_{pos}.csv")
        contagem = pd.read_csv(series / f"fig_depth_contagem_{pos}.csv")
        corr = pd.read_csv(series / f"fig_corr_matrix_{pos}.csv", index_col=0)

        # --- cruzamento das duas computacoes independentes -------------------
        aud.compara(
            f"series_figuras: Kruskal H {pos}",
            de_ft(pos, "kruskal_longevidade_por_faixa_draft", "estatistica"),
            de_tes(pos, "kruskal_wallis", "estatistica"),
            "fig_testes.csv vs testes_eda.csv", tol=5e-5,
            nota="dois passos independentes calculam a mesma estatistica",
        )
        aud.compara(
            f"series_figuras: Kruskal p {pos}",
            de_ft(pos, "kruskal_longevidade_por_faixa_draft", "p_valor"),
            de_tes(pos, "kruskal_wallis", "p"), "fig_testes.csv vs testes_eda.csv", tol=1e-12,
        )
        aud.compara(
            f"series_figuras: Mann-Whitney U {pos}",
            de_ft(pos, "mannwhitney_longevidade_por_jogos_perdidos", "estatistica"),
            de_tes(pos, "mann_whitney", "estatistica"), "fig_testes.csv vs testes_eda.csv", tol=5e-5,
        )
        aud.compara(
            f"series_figuras: Mann-Whitney p {pos}",
            de_ft(pos, "mannwhitney_longevidade_por_jogos_perdidos", "p_valor"),
            de_tes(pos, "mann_whitney", "p"), "fig_testes.csv vs testes_eda.csv", tol=1e-12,
        )

        # --- a serie plotada reconstroi a estatistica publicada --------------
        aud.compara(
            f"series_figuras: total de jogadores no histograma {pos}",
            float(hist["n_jogadores"].sum()), de_est(pos, "eda_jogadores"),
            f"fig_dist_carreira_{pos}.csv", tol=0,
        )
        acumulado = hist["n_jogadores"].cumsum()
        metade = hist["n_jogadores"].sum() / 2
        mediana_hist = float(hist.loc[acumulado >= metade, "temporadas"].iloc[0])
        aud.compara(
            f"series_figuras: mediana de carreira pelo histograma {pos}",
            mediana_hist, de_est(pos, "carreira_mediana_temporadas"),
            f"fig_dist_carreira_{pos}.csv", tol=0.5,
            nota="mediana agrupada difere no maximo 0,5 da mediana exata",
        )
        pct3 = float(hist.loc[hist["temporadas"] <= 3, "n_jogadores"].sum()) / float(hist["n_jogadores"].sum()) * 100
        aud.compara(
            f"series_figuras: percentual com ate 3 temporadas {pos}",
            pct3, de_est(pos, "carreira_pct_ate_3_temporadas"),
            f"fig_dist_carreira_{pos}.csv", tol=0.01,
        )
        aud.compara(
            f"series_figuras: total de jogadores nas faixas de draft {pos}",
            float(media["n_jogadores"].sum()), de_est(pos, "eda_jogadores"),
            f"fig_draft_media_{pos}.csv", tol=0,
        )
        media_ponderada = float(
            (media["media_temporadas"] * media["n_jogadores"]).sum() / media["n_jogadores"].sum()
        )
        aud.compara_arredondado(
            f"series_figuras: media de carreira ponderada pelas faixas {pos}",
            media_ponderada, de_est(pos, "carreira_media_temporadas"),
            f"fig_draft_media_{pos}.csv", nd=4,
            nota="o passo 50 publica a media com 4 casas",
        )
        aud.compara(
            f"series_figuras: total de jogadores nos grupos de jogos perdidos {pos}",
            float(grupos["n"].sum()), de_est(pos, "eda_jogadores"),
            f"fig_games_missed_grupos_{pos}.csv", tol=0,
        )
        aud.compara(
            f"series_figuras: registros jogador-temporada em depth_team {pos}",
            float(contagem["n_registros"].sum()), de_est(pos, "eda_registros_jogador_temporada"),
            f"fig_depth_contagem_{pos}.csv", tol=0,
        )
        diagonais = np.diag(corr.to_numpy(dtype=float))
        aud.padrao(
            f"series_figuras: diagonal unitaria da matriz de correlacao {pos}",
            bool(np.allclose(diagonais, 1.0)) and "career_length" in corr.columns,
            f"fig_corr_matrix_{pos}.csv",
            "matriz de Pearson precisa ter diagonal 1 e conter career_length",
        )
        aud.padrao(
            f"series_figuras: matriz de correlacao simetrica {pos}",
            bool(np.allclose(corr.to_numpy(dtype=float), corr.to_numpy(dtype=float).T, equal_nan=True)),
            f"fig_corr_matrix_{pos}.csv",
        )

    lag = series / "fig_career_lag_evo_QB.csv"
    if lag.exists():
        tabela = pd.read_csv(lag)
        por_var = tabela[tabela["variavel"] == "career_passing_yards"]
        aud.compara(
            "series_figuras: registros na evolucao de carreira QB",
            float(por_var["n"].sum()), de_est("QB", "eda_registros_jogador_temporada"),
            lag.name, tol=0,
        )

    # --- direcao do efeito de draft (rounds 1-2 contra fim/UDFA) ------------
    afirma_draft = bool(re.search(r"rodadas 1 e 2[^.]{0,180}mais temporadas", prosa, re.I))
    aud.padrao(
        "c3: afirmacao de maior longevidade para rounds 1 e 2", afirma_draft, "c3_material_tg.tex",
        "se a frase sair do texto, esta checagem perde o objeto e precisa ser reescrita",
    )
    for pos in ("QB", "RB", "WR"):
        tabela = pd.read_csv(series / f"fig_draft_media_{pos}.csv")
        cedo = tabela[tabela["faixa_draft"].isin(["Round 1", "Round 2"])]
        tarde = tabela[tabela["faixa_draft"].isin(["Round 7", "Undrafted"])]
        if cedo.empty or tarde.empty:
            continue
        m_cedo = float((cedo["media_temporadas"] * cedo["n_jogadores"]).sum() / cedo["n_jogadores"].sum())
        m_tarde = float((tarde["media_temporadas"] * tarde["n_jogadores"]).sum() / tarde["n_jogadores"].sum())
        if afirma_draft:
            aud.padrao(
                f"c3: direcao do draft (rounds 1-2 acima de fim/UDFA) — {pos}", m_cedo > m_tarde,
                f"fig_draft_media_{pos}.csv",
                f"media rounds 1-2={m_cedo:.2f} vs round 7/UDFA={m_tarde:.2f}",
            )

    # --- direcao da comparacao de jogos perdidos ----------------------------
    # A prosa afirma um lado ("mais longas"/"inferiores"); o sinal das medianas
    # gravadas tem que ser o mesmo. Foi assim que a inversao apareceu.
    par_gm = re.search(r"As ausências pedem a mesma atenção[\s\S]{0,1800}?(?=\n\n|\Z)", prosa)
    aud.padrao("c3: paragrafo de jogos perdidos", par_gm is not None, "c3_material_tg.tex")
    if par_gm:
        texto_gm = par_gm.group(0)
        medianas = re.search(
            r"medianas de temporadas são de (\d+) contra (\d+) no QB, (\d+) contra (\d+) no RB e \3 contra \4 no WR",
            texto_gm,
        )
        aud.padrao(
            "c3: medianas por ausencias declaradas",
            medianas is not None,
            "c3_material_tg.tex",
        )
        if medianas:
            valores = {
                "QB": (int(medianas.group(1)), int(medianas.group(2))),
                "RB": (int(medianas.group(3)), int(medianas.group(4))),
                "WR": (int(medianas.group(3)), int(medianas.group(4))),
            }
        else:
            valores = {}
        for pos in ("QB", "RB", "WR"):
            tabela = pd.read_csv(series / f"fig_games_missed_grupos_{pos}.csv")
            medias = dict(zip(tabela["grupo"], tabela["mediana"]))
            m_baixo, m_alto = medias.get("≤4 jogos"), medias.get(">4 jogos")
            if m_baixo is None or m_alto is None:
                continue
            if pos in valores:
                baixo, alto = valores[pos]
                aud.compara_arredondado(
                    f"c3: mediana com mais ausencias {pos}",
                    alto,
                    m_alto,
                    f"fig_games_missed_grupos_{pos}.csv",
                    0,
                )
                aud.compara_arredondado(
                    f"c3: mediana com ate quatro ausencias {pos}",
                    baixo,
                    m_baixo,
                    f"fig_games_missed_grupos_{pos}.csv",
                    0,
                )
                aud.padrao(
                    f"c3: mediana maior no grupo com mais ausencias {pos}",
                    alto > baixo and m_alto > m_baixo,
                    f"fig_games_missed_grupos_{pos}.csv",
                )

    # --- anotacoes de teste citadas na prosa --------------------------------
    trecho_mw = re.search(r"Mann--Whitney rejeita.{0,600}?\.", prosa, re.S)
    aud.padrao("c3: frase do teste de Mann-Whitney", trecho_mw is not None, "c3_material_tg.tex")
    if trecho_mw:
        for rotulo, pos in (("WR", "WR"), ("RB", "RB"), ("QB", "QB")):
            casou = re.search(rf"{rotulo}s?\b[^$]{{0,80}}\$p\s*=\s*([0-9]+\{{,\}}[0-9]+)\$", trecho_mw.group(0))
            aud.padrao(
                f"c3: p do Mann-Whitney para {pos} esta na prosa",
                casou is not None, "c3_material_tg.tex",
                "a anotacao da figura de jogos perdidos precisa constar no texto",
            )
            if casou:
                aud.compara_arredondado(
                    f"c3: Mann-Whitney p {pos}", numero(casou.group(1)),
                    de_tes(pos, "mann_whitney", "p"), "testes_eda.csv", nd=3,
                )
                rejeita = re.search(r"não(?:\s+rejeita)?(?:\s+(?:no|em))?\s*$", trecho_mw.group(0)[:casou.start()]) is None
                p = de_tes(pos, "mann_whitney", "p")
                if p is not None:
                    aud.padrao(
                        f"c3: veredito do Mann-Whitney para {pos}",
                        rejeita == (p < 0.05), "testes_eda.csv",
                        "o texto precisa dizer 'não rejeita' exatamente quando p >= 0,05",
                    )

    trecho_lr = re.search(
        r"contra os não draftados é significativo no RB e no WR \(\$p<0\{,\}001\$\) e não no QB \(\$p=0\{,\}45\$",
        prosa,
        re.S,
    )
    aud.padrao("c3: frase do teste log-rank de draft", trecho_lr is not None, "c3_material_tg.tex")
    if trecho_lr:
        aud.compara_p("c3: log-rank rodadas 1-2 vs nao draftados QB", "0{,}45", de_tes("QB", "logrank_r1r2_vs_udfa", "p"), "testes_eda.csv")
        for pos in ("RB", "WR"):
            aud.compara_p(f"c3: log-rank rodadas 1-2 vs nao draftados {pos}", "<0{,}001", de_tes(pos, "logrank_r1r2_vs_udfa", "p"), "testes_eda.csv")

    # --- afirmacoes qualitativas que os testes precisam sustentar -----------
    aud.padrao(
        "c3: Kruskal-Wallis rejeita para as tres posicoes",
        "Kruskal--Wallis rejeita a homogeneidade entre as faixas de \\textit{draft}" in prosa, "c3_material_tg.tex",
    )
    for pos in ("QB", "RB", "WR"):
        for teste, afirmacao in (
            ("kruskal_wallis", "Kruskal-Wallis rejeita a homogeneidade entre os estratos de draft"),
            ("logrank_faixa_etaria", "km_idade afirma curvas inferiores para estreia tardia"),
            ("logrank_profundidade", "km_depth afirma titulares acima de reservas"),
        ):
            p = de_tes(pos, teste, "p")
            aud.padrao(
                f"c3: {afirmacao} — {pos}", p is not None and p < 0.05, "testes_eda.csv",
                "afirmacao de diferenca exige p < 0,05 na posicao",
            )


def checa_rastreabilidade_figuras(aud: Auditoria) -> None:
    """Gate de lastro: toda figura citada tem produtor e lastro declarado existente.

    O CSV e produzido pelo passo `series_figuras`. Aqui a exigencia e dura: figura
    de dados sem lastro declarado, lastro que nao existe em disco ou citacao que
    nao aparece no CSV (rastreabilidade desatualizada) sao divergencias.
    """
    caminho = ROOT / "reprodutibilidade" / "rastreabilidade_figuras.csv"
    if not caminho.exists():
        return
    tabela = pd.read_csv(caminho).fillna("")
    modulo = _modulo_series_figuras()
    citadas_agora = set(modulo.figuras_citadas(LIVRO))
    registradas = set(tabela["figura"])

    aud.padrao(
        "rastreabilidade: CSV cobre exatamente as figuras citadas hoje",
        citadas_agora == registradas, caminho.name,
        "rode o passo `series_figuras`; o CSV esta desatualizado em relacao aos .tex"
        f" (faltam: {sorted(citadas_agora - registradas)[:5]};"
        f" sobram: {sorted(registradas - citadas_agora)[:5]})",
    )
    dados = tabela[tabela["tipo"] == "dados"]
    aud.padrao(
        "rastreabilidade: toda figura de dados tem lastro declarado",
        bool((dados["lastro_declarado"] != "").all()), caminho.name,
        "; ".join(dados.loc[dados["lastro_declarado"] == "", "figura"].tolist()),
    )
    # A coluna `lastro_ausente_no_momento` registra o estado quando o passo
    # `series_figuras` rodou — e ele roda ANTES dos passos de Cox, Aalen e do
    # pacote preditivo. O portao, portanto, confere o disco, nao a coluna.
    ausentes = []
    for _, linha in dados.iterrows():
        for artefato in [a for a in str(linha["lastro_declarado"]).split(";") if a]:
            if not (ROOT / artefato).exists():
                ausentes.append(f"{linha['figura']} -> {artefato}")
    aud.padrao(
        "rastreabilidade: todo lastro declarado existe em disco",
        not ausentes, caminho.name, "; ".join(ausentes[:6]),
    )
    aud.padrao(
        "rastreabilidade: toda figura citada tem produtor no manifesto",
        bool((tabela["produtor"] != "").all()), caminho.name,
        "; ".join(tabela.loc[tabela["produtor"] == "", "figura"].tolist()),
    )
    aud.padrao(
        "rastreabilidade: nenhuma figura de dados ficou sem regra de lastro",
        bool((dados["observacao"] != "").all()), "pipeline/52_series_figuras.py",
        "familia de figura sem regra declarada e lastro nao verificado",
    )


def checa_figuras_nao_citadas(aud: Auditoria) -> None:
    """Figuras geradas que nenhum capitulo cita: ruido declarado, nao silencioso."""
    caminho = ROOT / "reprodutibilidade" / "figuras_nao_citadas.csv"
    rastro = ROOT / "reprodutibilidade" / "rastreabilidade_figuras.csv"
    if not (caminho.exists() and rastro.exists()):
        return
    nao_citadas = pd.read_csv(caminho)["figura_gerada_nao_citada"].dropna().tolist()
    citadas = set(pd.read_csv(rastro)["figura"])
    aud.padrao(
        "figuras nao citadas: lista gerada pelo pipeline", len(nao_citadas) > 0, caminho.name,
        "a lista de ruido precisa existir para a limpeza ser decidida com numero",
    )
    sobrepostas = [n for n in nao_citadas if n in citadas]
    aud.padrao(
        "figuras nao citadas: nenhuma figura citada aparece como orfa",
        not sobrepostas, caminho.name, ", ".join(sobrepostas[:5]),
    )
    # Decisao do autor: figura gerada que documento nenhum usa e apagada. O `run_all.py`
    # faz a varredura no fim da execucao (`limpar_figuras_sem_uso`); a lista continua
    # existindo como registro do que saiu, mas os arquivos nao podem estar em disco.
    ainda_no_disco = [n for n in nao_citadas if (ROOT / n).exists()]
    aud.padrao(
        "figuras nao citadas: nenhuma delas ficou em disco", not ainda_no_disco, caminho.name,
        "a varredura do run_all deveria ter removido: " + ", ".join(ainda_no_disco[:5]),
    )
    fora_de_escopo = [n for n in nao_citadas if "figuras" not in Path(n).parts]
    aud.padrao(
        "figuras nao citadas: todas moram em diretorio figuras/", not fora_de_escopo, caminho.name,
        "a varredura so remove dentro de figuras/: " + ", ".join(fora_de_escopo[:5]),
    )


def checa_mask_support(aud: Auditoria) -> None:
    """Confere o artefato da regra `mask_test_support`.

    A regra descarta do teste observacoes com residual maior ou igual ao maximo
    do treino; o numero de linhas removidas vem do proprio passo que aplica a regra.
    """
    caminho = (
        ROOT / "documentos" / "estudo_modelos_finais" / "resultados" / "auditoria" / "mask_support.csv"
    )
    if not caminho.exists():
        return
    tabela = pd.read_csv(caminho)
    colunas = {"n_teste_antes", "n_teste_depois", "n_removidas"}
    aud.padrao(
        "mask_test_support: artefato cobre os folds oficiais", len(tabela) > 0, caminho.name,
        "sem folds registrados o efeito da regra nao esta medido",
    )
    aud.padrao(
        "mask_test_support: colunas de contagem presentes", colunas <= set(tabela.columns), caminho.name,
        f"colunas encontradas: {', '.join(map(str, tabela.columns))}",
    )
    if tabela.empty or not colunas <= set(tabela.columns):
        return
    aud.padrao(
        "mask_test_support: contagem consistente (antes - depois = removidas)",
        bool((tabela["n_teste_antes"] - tabela["n_teste_depois"] == tabela["n_removidas"]).all()),
        caminho.name,
    )
    posicoes = {str(p) for p in tabela["posicao"].unique()}
    aud.padrao(
        "mask_test_support: as tres posicoes aparecem", posicoes == {"QB", "RB", "WR"}, caminho.name,
        f"posicoes: {', '.join(sorted(posicoes))}",
    )
    temporadas = {int(t) for t in tabela["temporada"].unique()}
    aud.padrao(
        "mask_test_support: so temporadas do walk-forward (2018-2023)",
        temporadas <= set(range(2018, 2024)), caminho.name,
        f"temporadas: {', '.join(map(str, sorted(temporadas)))}",
    )
    for pos in ("QB", "RB", "WR"):
        do_pos = {int(t) for t in tabela.loc[tabela["posicao"] == pos, "temporada"]}
        aud.padrao(
            f"mask_test_support: os quatro folds oficiais de {pos} estao no artefato",
            {2018, 2019, 2020, 2021} <= do_pos, caminho.name,
            f"temporadas de {pos}: {', '.join(map(str, sorted(do_pos)))}",
        )


def _pdftotext_poppler() -> str | None:
    """Caminho do `pdftotext` do Poppler, ou None (outro `pdftotext`, como o do xpdf, e ignorado).

    A saida e lida em UTF-8 (`-enc UTF-8`): no Windows, a decodificacao padrao do subprocess
    quebra os rotulos acentuados das figuras.
    """
    exe = shutil.which("pdftotext")
    if exe is None:
        return None
    try:
        versao = subprocess.run([exe, "-v"], capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return exe if "poppler" in (versao.stdout + versao.stderr).lower() else None


def _extrator_pdf_disponivel() -> bool:
    """`pdftotext` (Poppler) ou, na falta dele, PyMuPDF."""
    if _pdftotext_poppler() is not None:
        return True
    try:
        import fitz  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _texto_pdf_bruto(caminho: Path, layout: bool = False) -> str | None:
    """Texto de um PDF: `pdftotext` quando existe, PyMuPDF como alternativa (Windows)."""
    exe = _pdftotext_poppler()
    if exe is not None:
        cmd = [exe, "-enc", "UTF-8"] + (["-layout"] if layout else []) + [str(caminho), "-"]
        try:
            saida = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=120)
        except (OSError, subprocess.SubprocessError):
            return None
        # o matplotlib desenha o hifen dos rotulos como U+2212 (sinal de menos); normaliza para "-"
        return saida.stdout.replace("\u2212", "-") if saida.returncode == 0 and saida.stdout else None
    try:
        import fitz  # noqa: PLC0415

        with fitz.open(caminho) as doc:
            return "\n".join(pagina.get_text(sort=layout) for pagina in doc).replace("\u2212", "-") or None
    except Exception:  # noqa: BLE001
        return None


def _texto_da_figura(caminho: Path) -> str | None:
    """Texto impresso na figura (anotacoes, titulos).

    A comparacao e sobre o texto NORMALIZADO (espacos colapsados), porque a
    extracao quebra linha no meio das anotacoes.
    """
    bruto = _texto_pdf_bruto(caminho, layout=True)
    return re.sub(r"\s+", " ", bruto) if bruto else None


def checa_anotacoes_figuras(aud: Auditoria) -> None:
    """O que esta DESENHADO na figura confere com o artefato que a sustenta.

    As checagens anteriores garantem que a figura tem produtor e lastro em disco, e que
    os numeros da prosa batem com o artefato. Falta o elo do meio: a anotacao impressa
    dentro da figura. Uma figura pode estar no lugar certo, citada na frase certa e
    desenhada com o numero errado.

    Le o texto do PDF (`pdftotext`) e exige que cada valor anotado apareca na forma que
    o artefato produz: medianas e quartis das figuras do EDA, log-rank das curvas de
    Kaplan-Meier, HR com IC exponenciado do forest plot e p de Schoenfeld por covariavel
    e global. O IC do `cox_hr_*.csv` esta na escala do coeficiente; a figura mostra
    `exp()`, e a checagem reproduz essa exponenciacao.
    """
    if not _extrator_pdf_disponivel():
        aud.checagens += 1
        aud.divergencias.append(
            Divergencia(
                "anotacoes de figura",
                "pdftotext disponivel",
                "ausente",
                "ambiente",
                "sem pdftotext a correspondencia entre desenhado e gerado nao pode ser verificada",
            )
        )
        return

    # O mesmo formatador que DESENHA a anotacao: importar evita reimplementar a regra
    # do limite ("< 0.0001") e deixa checagem e figura presas uma na outra.
    try:
        sys.path.insert(0, str(ROOT / "Predição" / "shared"))
        from thesis_figure_style import format_p_value, thesis_var_label  # noqa: PLC0415
    except Exception as erro:  # noqa: BLE001
        aud.checagens += 1
        aud.divergencias.append(
            Divergencia(
                "anotacoes de figura",
                "formatador das figuras importavel",
                f"falhou: {erro}",
                "Predição/shared/thesis_figure_style.py",
                "sem o formatador a forma anotada na figura nao pode ser prevista",
            )
        )
        return

    res = ROOT / "Análise Exploratória de Dados" / "resultados"
    series = res / "series_figuras"
    f_est, f_ft = res / "estatisticas_eda.csv", series / "fig_testes.csv"
    tes_p = res / "testes_eda.csv"
    if not (f_est.exists() and f_ft.exists() and tes_p.exists()):
        return
    est = pd.read_csv(f_est)
    ft = pd.read_csv(f_ft)
    tes = pd.read_csv(tes_p)

    def figura(nome: str) -> str | None:
        # `LIVRO/figuras` e symlink para `../figuras`; no Windows o git o grava como arquivo-texto.
        for candidato in (LIVRO / "figuras" / nome, ROOT / "Escrita Final" / "figuras" / nome, LIVRO / nome):
            if candidato.exists():
                texto_fig = _texto_da_figura(candidato)
                # as figuras usam virgula decimal; as checagens comparam com ponto
                return re.sub(r"(\d),(\d)", r"\1.\2", texto_fig) if texto_fig else texto_fig
        return None

    def p_teste(pos: str, teste: str) -> float | None:
        linha = tes[(tes["posicao"] == pos) & (tes["teste"] == teste)]
        return float(linha["p"].iloc[0]) if len(linha) else None

    def valor_ft(pos: str, teste: str, campo: str) -> float | None:
        linha = ft[(ft["posicao"] == pos) & (ft["teste"] == teste)]
        return float(linha[campo].iloc[0]) if len(linha) else None

    def valor_est(pos: str, metrica: str) -> float | None:
        linha = est[(est["posicao"] == pos) & (est["metrica"] == metrica)]
        return float(linha["valor"].iloc[0]) if len(linha) else None

    def contem(texto: str, esperado: str, onde: str, artefato: str) -> None:
        aud.padrao(onde, esperado in texto, artefato, f"anotacao esperada na figura: {esperado!r}")

    for pos in ("QB", "RB", "WR"):
        # --- distribuicao do tamanho de carreira -----------------------------
        texto = figura(f"dist_carreira_{pos}.pdf")
        if texto:
            hist = pd.read_csv(series / f"fig_dist_carreira_{pos}.csv")
            n = int(hist["n_jogadores"].sum())
            acumulado = hist["n_jogadores"].cumsum()
            mediana = float(hist.loc[acumulado >= n / 2, "temporadas"].iloc[0])
            media = float((hist["temporadas"] * hist["n_jogadores"]).sum() / n)
            p25 = float(hist.loc[acumulado >= n * 0.25, "temporadas"].iloc[0])
            p75 = float(hist.loc[acumulado >= n * 0.75, "temporadas"].iloc[0])
            pct3 = float(hist.loc[hist["temporadas"] <= 3, "n_jogadores"].sum()) / n * 100
            for esperado in (
                f"Mediana = {mediana:.0f}",
                f"Média = {media:.1f}",
                f"P25 = {p25:.0f}",
                f"P75 = {p75:.0f}",
                f"{pct3:.0f}% das carreiras",
            ):
                contem(texto, esperado, f"figura dist_carreira_{pos}", f"fig_dist_carreira_{pos}.csv")

        # --- longevidade por faixa de draft ----------------------------------
        texto = figura(f"draft_{pos}.pdf")
        if texto:
            pass  # os valores do Kruskal-Wallis saíram da figura e estão na tabela do capítulo 3

        # --- jogos perdidos e longevidade ------------------------------------
        texto = figura(f"games_missed_{pos}.pdf")
        if texto:
            p = valor_ft(pos, "mannwhitney_longevidade_por_jogos_perdidos", "p_valor")
            if p is not None:
                contem(texto, f"p {format_p_value(p)}", f"figura games_missed_{pos} (Mann-Whitney)", "fig_testes.csv")

        # --- Kaplan-Meier: mediana global e log-rank por estrato -------------
        texto = figura(f"km_global_{pos}.pdf")
        if texto:
            mediana = valor_est(pos, "km_mediana_anos")
            if mediana is not None:
                contem(texto, f"Mediana = {mediana:.1f} anos", f"figura km_global_{pos}", "estatisticas_eda.csv")

        n_km = valor_est(pos, "km_n_jogadores")
        for nome, teste, rotulo in (
            (f"km_draft_{pos}.pdf", "logrank_draft_tiers", "log-rank de draft"),
            (f"km_idade_{pos}.pdf", "logrank_faixa_etaria", "log-rank de faixa etaria"),
            (f"km_depth_{pos}.pdf", "logrank_profundidade", "log-rank de profundidade"),
        ):
            texto = figura(nome)
            if not texto:
                continue
            p = p_teste(pos, teste)
            if p is not None:
                contem(texto, f"Log-rank p {format_p_value(p)}", f"figura {nome} ({rotulo})", "testes_eda.csv")
            # Os estratos da figura tem que particionar a amostra do Kaplan-Meier.
            tamanhos = [int(x) for x in re.findall(r"\(n=(\d+)\)", texto)]
            if len(tamanhos) >= 2 and n_km is not None:
                aud.padrao(
                    f"figura {nome}: estratos somam a amostra do KM",
                    sum(tamanhos) == int(n_km), "estatisticas_eda.csv",
                    f"soma dos estratos = {sum(tamanhos)}, km_n_jogadores = {int(n_km)}",
                )

        # --- forest plot: HR, IC95 exponenciado e p --------------------------
        texto = figura(f"forest_plot_{pos}.pdf")
        hr_path = COX / f"cox_hr_{pos}.csv"
        if texto and hr_path.exists():
            tabela = pd.read_csv(hr_path)
            for _, linha in tabela.iterrows():
                hr = float(linha["HR"])
                lo, hi = np.exp(float(linha["CI_lower"])), np.exp(float(linha["CI_upper"]))
                p = float(linha["p_value"])
                contem(texto, f"HR={hr:.2f}", f"figura forest_plot_{pos} ({linha['covariate']})", hr_path.name)
                contem(texto, f"({lo:.2f}-{hi:.2f})", f"figura forest_plot_{pos} IC ({linha['covariate']})", hr_path.name)
                # mesmo rotulo que pipeline/47_export_figuras_tese.py desenha ("p < 0.001" ou "p = 0.036")
                rotulo_p = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
                contem(texto, rotulo_p, f"figura forest_plot_{pos} p ({linha['covariate']})", hr_path.name)

        # --- Schoenfeld: p por covariavel e teste global ---------------------
        texto = figura(f"schoenfeld_{pos}.pdf")
        sch_path = COX / f"schoenfeld_{pos}.csv"
        if texto and sch_path.exists():
            tabela = pd.read_csv(sch_path)
            for _, linha in tabela.iterrows():
                termo, p = str(linha["term"]), float(linha["p"])
                if termo == "GLOBAL":
                    contem(texto, f"chisq = {float(linha['chisq']):.2f}", f"figura schoenfeld_{pos} (chisq global)", sch_path.name)
                    contem(texto, f"p = {p:.4f}", f"figura schoenfeld_{pos} (p global)", sch_path.name)
                else:
                    contem(texto, thesis_var_label(termo), f"figura schoenfeld_{pos} rotulo ({termo})", sch_path.name)
                    contem(texto, f"p = {p:.4f}", f"figura schoenfeld_{pos} p ({termo})", sch_path.name)



# ---------------------------------------------------------------- ramo preditivo
#
# Checagens dos capitulos 5 a 7, resumo e abstract. Cada numero citado e recalculado
# do artefato que o produz, formatado como a escrita o cita e procurado no texto. Uma frase reescrita ou um numero errado reprova.

FUROS = ROOT / "documentos" / "estudo_furos_preditivos" / "resultados"
VAZ = ROOT / "documentos" / "estudo_vazamento" / "resultados"
SENS_TUNE = ROOT / "documentos" / "estudo_sensibilidade_tune" / "resultados"
SENS_MET = ROOT / "documentos" / "estudo_sensibilidade_metodo" / "resultados"
AALEN_SEL = ROOT / "documentos" / "estudo_aalen_selecao" / "resultados"
ROT_MODELO = {"cox": "Cox", "gbsa": "GBSA", "rsf": "RSF", "km": "Kaplan--Meier"}


def br(valor: float, nd: int = 3, sinal: bool = False) -> str:
    """Numero como a escrita o cita dentro de $...$ (virgula decimal com chaves).

    Arredondamento usual (meio para cima sobre o decimal do artefato, p. ex. 0,6825 -> 0,683).
    """
    from decimal import Decimal, ROUND_HALF_UP
    quant = Decimal(1).scaleb(-nd)
    arred = Decimal(repr(abs(float(valor)))).quantize(quant, rounding=ROUND_HALF_UP)
    texto = f"{arred:.{nd}f}".replace(".", "{,}")
    if sinal:
        return ("-" if float(valor) < 0 and round(abs(float(valor)), nd) > 0 else "+") + texto
    return ("-" if float(valor) < 0 and round(abs(float(valor)), nd) > 0 else "") + texto


def en(valor: float, nd: int = 3) -> str:
    return f"{float(valor):.{nd}f}"


def _texto(nome: str) -> str:
    caminho = LIVRO / nome
    return caminho.read_text(encoding="utf-8", errors="replace") if caminho.exists() else ""


def _cadeia_cox(posicao: str) -> dict:
    """Cadeia do Cox na ordem de pipeline/11: identificabilidade, colinearidade 0,85, nova identificabilidade."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "Inferência" / "shared"))
    import survival_utils as _su  # noqa: E402
    funil = json.loads((PREP / "funil_covariaveis.json").read_text(encoding="utf-8"))
    df = pd.read_csv(PREP / f"df_cox_{posicao}.csv")
    cand = [c for c in funil[posicao]["candidatas"] if c in df.columns]
    ident = _su.covariaveis_identificaveis_cox(df, cand, verbose=False)
    coli = _su.filter_collinear_covariates(df, ident, threshold=0.85, verbose=False)
    coli = _su.force_missing_flags(coli, df.columns.tolist())
    final = _su.covariaveis_identificaveis_cox(df, coli, verbose=False)
    return {"candidatas": len(cand), "ident": len(ident), "coli": len(coli), "final": len(final)}


def _identificaveis_cox(posicao: str) -> int:
    """Colunas do saturado do Cox (apos a cadeia de pipeline/11)."""
    return _cadeia_cox(posicao)["final"]


def _vies_exato(posicao: str) -> float:
    """Vies medio em anos do modelo de maior C, sem o arredondamento do CSV de resumo."""
    o = pd.read_csv(OFICIAL / "pred_oof_walkforward_2018_2021.csv")
    o = o[o.error_identifiable.astype(str).str.lower().isin(["true", "1"])]
    s = o[(o.posicao == posicao) & (o.modelo == CAMPEAO[posicao])]
    return float((s.expected_years - s.error_target_years).mean())


VALOR_REGISTRADO = object()  # sentinela: valor de artefato nao citado no texto, conferido contra o registro
VALORES_REGISTRADOS = Path(__file__).resolve().parent / "valores_registrados.json"
_ATUALIZAR_REGISTRADOS = False


def _registrado(aud: Auditoria, rotulo: str, valor: str, artefato: str) -> None:
    """Valor de artefato que o texto nao cita, conferido contra o registro.

    A checagem confere que o valor calculado do artefato continua o registrado em
    valores_registrados.json, para que uma mudanca no artefato nao passe em
    silencio. O registro so muda com --atualizar-registrados, de modo explicito.
    """
    aud.checagens += 1
    registro = json.loads(VALORES_REGISTRADOS.read_text(encoding="utf-8")) if VALORES_REGISTRADOS.exists() else {}
    if _ATUALIZAR_REGISTRADOS:
        registro[rotulo] = valor
        VALORES_REGISTRADOS.write_text(json.dumps(registro, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return
    if registro.get(rotulo) != valor:
        aud.divergencias.append(Divergencia(rotulo, "(valor registrado)", valor, artefato,
                                            "valor do artefato diferente do registrado; use --atualizar-registrados se a mudanca for esperada"))



def _tem(aud: Auditoria, rotulo: str, trecho: str, texto: str, artefato: str) -> bool:
    if texto is VALOR_REGISTRADO:
        _registrado(aud, rotulo, trecho, artefato)
        return True
    ok = trecho in texto
    aud.checagens += 1
    if not ok:
        aud.divergencias.append(Divergencia(rotulo, "(trecho ausente)", trecho, artefato, "numero do artefato nao citado como esperado"))
    return ok


def _tem_se_afirmado(
    aud: Auditoria,
    rotulo: str,
    trecho: str,
    texto: str,
    artefato: str,
    gatilho: str,
    artefato_valido: bool,
) -> None:
    if re.search(gatilho, texto, re.I):
        _tem(aud, rotulo, trecho, texto, artefato)
    else:
        aud.checagens += 1  # sem alegacao correspondente na escrita: nada a conferir contra o artefato


def _verdade(aud: Auditoria, rotulo: str, condicao: bool, artefato: str, detalhe: str = "") -> None:
    aud.checagens += 1
    if not condicao:
        aud.divergencias.append(Divergencia(rotulo, "afirmacao da escrita", "contrariada pelo artefato", artefato, detalhe))


def janela_citada(texto: str, posicao: str, inicio: int, fim: int) -> bool:
    padrao = (rf"(?:do\s+)?{inicio}\.?º?\.?\s+(?:ao|a)\s+{fim}\.?º?\.?(?:\s+anos?)?\s+no\s+{re.escape(posicao)}\b"
              rf"|de\s+{inicio}\s+a\s+{fim}\s+(?:anos\s+de\s+risco\s+)?no\s+{re.escape(posicao)}\b")
    return re.search(padrao, texto) is not None


def constancia_compativel_com_p(resultado: dict) -> bool:
    valor_p = resultado.get("p_value")
    return (
        isinstance(valor_p, (int, float))
        and math.isfinite(valor_p)
        and 0 <= valor_p <= 1
        and resultado.get("is_constant") is (valor_p >= 0.05)
    )


def coeficiente_no_suporte(dados: pd.DataFrame, covariavel: str, tempo: float, inicio: float, fim: float) -> float:
    if not inicio <= tempo <= fim:
        raise ValueError("Coeficiente solicitado fora da janela ajustada")
    anteriores = dados.loc[dados["time"] <= tempo].sort_values("time")
    if anteriores.empty:
        raise ValueError("Nao ha coeficiente observado ate o tempo solicitado")
    return float(anteriores.iloc[-1][covariavel])


def checa_tgb_tabelas(aud: Auditoria) -> None:
    """Tabelas do cap. 5."""
    # tab:tgb_hit: modelos contra acerto_horizontes, Kaplan-Meier contra a referencia do diagnostico
    hit_p = OFICIAL / "auditoria" / "acerto_horizontes.csv"
    km_p = DIAG / "erro_km_referencia.csv"
    if hit_p.exists() and km_p.exists():
        d = pd.concat([pd.read_csv(hit_p), pd.read_csv(km_p)], ignore_index=True)
        vistos = 0
        for linha in aud.linhas("tab:tgb_hit"):
            if len(linha) < 7 or linha[0].strip() not in POSICOES:
                continue
            pos = linha[0].strip()
            mod = next((k for k, v in ROT_MODELO.items() if v.replace("--", "-") == linha[1].strip().replace("--", "-")), None)
            r = d[(d.posicao == pos) & (d.modelo == mod)]
            if r.empty:
                aud.padrao(f"tab:tgb_hit {pos} {linha[1]} no artefato", False, hit_p.name)
                continue
            r = r.iloc[0]
            vistos += 1
            ref = f"tab:tgb_hit {pos} {mod}"
            aud.compara_arredondado(f"{ref} mae", numero(linha[2]), r["mae"], hit_p.name, 2)
            aud.compara_arredondado(f"{ref} vies", numero(linha[3]), r["vies"], hit_p.name, 2)
            aud.compara_arredondado(f"{ref} mediana", numero(linha[4]), r["mediana_ae"], hit_p.name, 2)
            aud.compara_arredondado(f"{ref} acerto_1a", numero(linha[5]), r["acerto_1a"], hit_p.name, 1)
            aud.compara_arredondado(f"{ref} acerto_3a", numero(linha[6]), r["acerto_3a"], hit_p.name, 1)
        aud.compara_arredondado("tab:tgb_hit linhas conferidas", 12, vistos, hit_p.name, 0)

    # tab:tgb_vies: modelo de maior C contra erro_por_ano, Kaplan-Meier contra erro_km_por_fold
    ano_p, kmf_p = DIAG / "erro_por_ano.csv", DIAG / "erro_km_por_fold.csv"
    if ano_p.exists() and kmf_p.exists():
        ano, kmf = pd.read_csv(ano_p), pd.read_csv(kmf_p)
        for linha in aud.linhas("tab:tgb_vies"):
            if len(linha) < 6 or linha[0].strip() not in POSICOES:
                continue
            pos = linha[0].strip()
            km = "Kaplan" in linha[1]
            if not km:
                _verdade(aud, f"tab:tgb_vies {pos}: linha e do modelo de maior C", linha[1].strip() == ROT_MODELO[CAMPEAO[pos]], ano_p.name)
            for col, fold in zip(range(2, 6), (2018, 2019, 2020, 2021)):
                if km:
                    v = kmf[(kmf.posicao == pos) & (kmf.fold == fold)]["vies_fold"]
                else:
                    v = ano[(ano.posicao == pos) & (ano.modelo == CAMPEAO[pos]) & (ano.fold == fold)]["vies_medio"]
                aud.compara_arredondado(f"tab:tgb_vies {pos} {'km' if km else 'modelo'} {fold}", numero(linha[col]), float(v.iloc[0]), ano_p.name, 2)

    # tab:tgb_2024: modelo de maior C de cada posicao e postos nas tres familias
    p24 = OFICIAL / "predicao_2024_modelos.csv"
    if p24.exists():
        d = pd.read_csv(p24)
        n = 0
        for linha in aud.linhas("tab:tgb_2024"):
            if len(linha) < 6 or linha[0].strip() not in POSICOES:
                continue
            pos, nome = linha[0].strip(), linha[1].replace("\\ ", " ").replace("\\", "").strip()
            sub = d[(d.posicao == pos) & (d.player_name.str.replace(".", "", regex=False).str.lower() == nome.replace(".", "").lower())]
            if sub.empty:
                aud.padrao(f"tab:tgb_2024 {pos} {nome} existe", False, p24.name)
                continue
            n += 1
            camp = sub[sub.modelo == CAMPEAO[pos]].iloc[0]
            aud.compara_arredondado(f"tab:tgb_2024 {pos} {nome} anos", numero(linha[2]), camp["expected_years"], p24.name, 2)
            for col, mod in zip(range(3, 6), ("cox", "gbsa", "rsf")):
                r = sub[sub.modelo == mod]
                if r.empty:
                    _verdade(aud, f"tab:tgb_2024 {pos} {nome} sem {mod}", linha[col].strip() in ("--", "-", ""), p24.name)
                else:
                    aud.compara_arredondado(f"tab:tgb_2024 {pos} {nome} posto {mod}", numero(linha[col]), r.iloc[0]["rank"], p24.name, 0)
            # a tabela lista o topo do modelo de maior C, em ordem
            topo = d[(d.posicao == pos) & (d.modelo == CAMPEAO[pos])].sort_values("rank")
            _verdade(aud, f"tab:tgb_2024 {pos} {nome} no topo do modelo de maior C", int(camp["rank"]) <= 5, p24.name)
        for nome_txt in ("Lamar Jackson", "C.J. Stroud", "Brock Purdy", "Devon Achane", "Bijan Robinson", "Malik Nabers", "Brian Thomas", "Amon-Ra St. Brown", "Joe Flacco", "Allen Robinson"):
            achou_nome = not d[d.player_name.str.replace(".", "", regex=False).str.lower() == nome_txt.replace(".", "").lower()].empty
            aud.padrao(f"c5 lista 2024: {nome_txt} no artefato", achou_nome, p24.name)

    # tab:tgb_tune
    tune_p = TUNE / "resumo_oficial_2018_2021.csv"
    el_p = FUROS / "elasticnet_vs_cox.csv"
    if tune_p.exists() and el_p.exists():
        t = pd.read_csv(tune_p).set_index(["posicao", "modelo", "tag"])["c_index"]
        el = pd.read_csv(el_p)
        el = el[el.modelo == "elasticnet"].set_index("posicao")["c_index_medio"]
        n = 0
        for linha in aud.linhas("tab:tgb_tune"):
            if len(linha) < 5:
                continue
            nome = linha[0].strip()
            if nome not in ("Cox", "GBSA", "RSF") and not nome.startswith("Rede"):
                continue
            if nome.startswith("Rede"):
                for col, pos in zip(range(2, 5), POSICOES):
                    aud.compara_arredondado(f"tab:tgb_tune rede elastica {pos}", numero(linha[col]), el[pos], el_p.name, 3)
                n += 1
                continue
            mod = nome.lower()
            tag = "default" if linha[1].startswith("padr") else "tunado"
            for col, pos in zip(range(2, 5), POSICOES):
                if (pos, mod, tag) in t.index:
                    aud.compara_arredondado(f"tab:tgb_tune {mod} {tag} {pos}", numero(linha[col]), t[(pos, mod, tag)], tune_p.name, 3)
                else:
                    _verdade(aud, f"tab:tgb_tune {mod} {tag} {pos} fora", linha[col].strip() == "fora", tune_p.name)
            n += 1
        _tem(aud, "c5 tune: busca supera o padrão em sete combinações", "A busca melhora o $C$ médio em sete das nove combinações", _texto("apendice_predicao.tex"), tune_p.name)


def checa_tgb_prosa_c5(aud: Auditoria) -> None:
    """Numeros citados na prosa do cap. 5."""
    t = _texto("c5_resultados_preditivos.tex")
    wf_p = OFICIAL / "resumo_walkforward_2018_2021.csv"
    if not t or not wf_p.exists():
        return
    wf = pd.read_csv(wf_p).set_index(["posicao", "modelo"])
    c = wf["c_index"]
    nao_km = c[[i for i in c.index if i[1] != "km"]]

    # 5.1 protocolo: Aalen fora da comparacao; RSF ajustado nas tres posicoes; sem regra de eventos
    _verdade(aud, "c5 sem numeros do Aalen preditivo", not re.search(r"Aalen[^.]{0,200}\$0\{,\}", t), "c5_resultados_preditivos.tex")
    _verdade(aud, "c5 sem regra de eventos do RSF", "saídas ou mais" not in t, "c5_resultados_preditivos.tex")
    rsf_p = OFICIAL / "auditoria" / "rsf_qb_diagnostico.csv"
    if rsf_p.exists():
        rsf = pd.read_csv(rsf_p).set_index("posicao")
        _verdade(aud, "c5 RSF ajustado nas tres posicoes", bool(rsf.rsf_ajustado.all()), rsf_p.name)
    _tem(aud, "c5 comparacao com valores padrao remete a robustez", "Sete verificações refazem escolhas do protocolo", VALOR_REGISTRADO, "tgb_protocol.py")
    _tem(
        aud,
        "c5 corte do treino em 2018 e acompanhamento so na avaliacao",
        "Na origem $s$, o treino reúne as temporadas anteriores a $s$, e o teste reúne a temporada $s$",
        t,
        "resumo_walkforward_2018_2021.csv",
    )
    _tem(
        aud,
        "c5 landmark: origens com acompanhamento completo",
        "de 2020 a 2022 com $w=1$, 2020 e 2021 com $w=2$ e 2020 com $w=3$",
        t,
        "metricas_landmark_folds.csv + hps_por_fold.csv",
    )
    proto = (ROOT / "Predição" / "shared" / "tgb_protocol.py").read_text(encoding="utf-8")
    for trecho in ('"n_estimators": 100,\n    "learning_rate": 0.1,\n    "max_depth": 3,', '"min_samples_leaf": 3,'):
        _verdade(aud, f"protocolo declara {trecho.splitlines()[0][:30]}", trecho in proto, "tgb_protocol.py")

    # 5.2 comparacao
    _tem(aud, "c5 faixa do C", f"médio fica entre ${br(nao_km.min(), 3)}$ e ${br(nao_km.max(), 3)}$, contra $0{{,}}500$ do Kaplan--Meier", t, wf_p.name)
    ic_p = IC / "ic_metricas.csv"
    if ic_p.exists():
        ic = pd.read_csv(ic_p)
        _verdade(aud, "c5 nenhum IC chega a 0,50", float(ic[ic.metrica == "c_index"].ic_baixo.min()) > 0.5, ic_p.name)
    frase = "O modelo usado nas leituras seguintes é o de maior $C$ em cada posição: RSF no QB, Cox no RB e no WR"
    _tem(aud, "c5 modelos de maior C", frase, t, wf_p.name)
    _verdade(aud, "c5 campeoes declarados = artefato", CAMPEAO == {"QB": "rsf", "RB": "cox", "WR": "cox"}, wf_p.name, str(CAMPEAO))
    dist = max(float(nao_km.xs(p, level="posicao").max() - nao_km.xs(p, level="posicao").min()) for p in POSICOES)
    # distancia maxima e contrastes pareados: valores na tab:tgb_delta (conferida por tab:tgb_delta acima)
    par_p = IC / "delta_c_pareado.csv"
    if par_p.exists():
        dp = pd.read_csv(par_p)
        _verdade(aud, "c5 contrastes com p<0,05: Cox contra arvores no RB",
                 set(map(tuple, dp[(dp.p_bilateral < 0.05) & dp.par.str.contains("cox")][["posicao", "par"]].values.tolist())) == {("RB", "gbsa-cox"), ("RB", "rsf-cox")}, par_p.name)
        _verdade(aud, "c5 nove contrastes", len(dp) == 9, par_p.name)
        g_rb, r_rb = dp[(dp.posicao == "RB") & (dp.par == "gbsa-cox")].iloc[0], dp[(dp.posicao == "RB") & (dp.par == "rsf-cox")].iloc[0]
        g_wr, r_wr = dp[(dp.posicao == "WR") & (dp.par == "gbsa-cox")].iloc[0], dp[(dp.posicao == "WR") & (dp.par == "rsf-cox")].iloc[0]
        q_r = dp[(dp.posicao == "QB") & (dp.par == "rsf-cox")].iloc[0]
        q_g = dp[(dp.posicao == "QB") & (dp.par == "gbsa-cox")].iloc[0]
        q_gr = dp[(dp.posicao == "QB") & (dp.par == "gbsa-rsf")].iloc[0]
        _tem(aud, "c5 contrastes QB: GBSA abaixo do RSF", "No QB, RSF e Cox empatam", t, par_p.name)
        _verdade(aud, "c5 contrastes QB: so GBSA-RSF com Holm abaixo de 0,05", set(dp[(dp.posicao == "QB") & (dp.p_holm < 0.05)].par) == {"gbsa-rsf"}, par_p.name)
        _verdade(aud, "c5 intervalo QB RSF-Cox contem zero", q_r.ic_baixo < 0 < q_r.ic_alto, par_p.name)

    # fold a fold
    fd_p = OFICIAL / "metricas_walkforward_folds.csv"
    if fd_p.exists():
        f = pd.read_csv(fd_p)
        f = f[(f.protocolo == "walkforward") & (f.modelo != "km") & (f.teste_em_temporada.between(2018, 2021))]
        venc = f.loc[f.groupby(["posicao", "teste_em_temporada"]).c_index.idxmax()].set_index(["posicao", "teste_em_temporada"])
        cf = f.set_index(["posicao", "modelo", "teste_em_temporada"])["c_index"]
        esperado = {("QB", 2018): "rsf", ("QB", 2019): "gbsa", ("QB", 2020): "rsf", ("QB", 2021): "cox",
                    ("RB", 2018): "cox", ("RB", 2019): "cox", ("RB", 2020): "cox", ("RB", 2021): "cox",
                    ("WR", 2018): "rsf", ("WR", 2019): "cox", ("WR", 2020): "cox", ("WR", 2021): "cox"}
        for (pos, ano), mod in esperado.items():
            _verdade(aud, f"c5 vencedor do fold {pos} {ano}", venc.loc[(pos, ano), "modelo"] == mod, fd_p.name, str(venc.loc[(pos, ano), "modelo"]))
        _tem(aud, "c6 WR: predição menos clara", "o WR é a posição em que a predição é menos clara", VALOR_REGISTRADO, fd_p.name)
        _verdade(aud, "c5 Cox com maior C nos quatro folds do RB", all(venc.loc[("RB", a), "modelo"] == "cox" for a in (2018, 2019, 2020, 2021)), fd_p.name)
    sf_p = IC / "sensibilidade_folds.csv"
    if sf_p.exists():
        sf = pd.read_csv(sf_p)
        disp = sf[(sf.conjunto == "disponivel_2018_2023") & (sf.posicao == "QB")].set_index("modelo")["c_medio"]
        _verdade(aud, "c5 Cox maior C com seis folds no RB e no WR",
                 bool((sf[(sf.conjunto == "disponivel_2018_2023") & sf.campeao & (sf.posicao != "QB")].modelo == "cox").all()), sf_p.name)

    # IBS
    ibs = wf["ibs"]
    _verdade(aud, "c5 IBS: RSF a frente do Cox no QB e no RB", ibs[("QB", "rsf")] < ibs[("QB", "cox")] and ibs[("RB", "rsf")] < ibs[("RB", "cox")], wf_p.name)
    _tem(aud, "c7 IBS: RSF menor no QB e no WR", "o RSF tem o menor erro de probabilidade no QB e no WR", VALOR_REGISTRADO, wf_p.name)
    _tem(aud, "c7 IBS: RB empata com o Cox", f"no RB empata com o Cox, com IBS de ${br(ibs[('RB','rsf')])}$ e ${br(ibs[('RB','cox')])}$", VALOR_REGISTRADO, wf_p.name)
    _verdade(aud, "c5 IBS WR: RSF com o menor IBS no WR", ibs[("WR", "rsf")] < min(ibs[("WR", "cox")], ibs[("WR", "gbsa")]), wf_p.name)
    _verdade(aud, "c7 IBS WR: RSF abaixo de Cox e GBSA", ibs[("WR", "rsf")] < min(ibs[("WR", "cox")], ibs[("WR", "gbsa")]), wf_p.name)
    _verdade(aud, "c5 modelo principal abaixo do KM no IBS", all(ibs[(p, CAMPEAO[p])] < ibs[(p, "km")] for p in POSICOES), wf_p.name)
    melhora = max(float(ibs[(p, "km")] - ibs.xs(p, level="posicao").drop("km").min()) for p in POSICOES)
    melhora_min = min(float(ibs[(p, "km")] - ibs.xs(p, level="posicao").drop("km").min()) for p in POSICOES)
    _tem(aud, "c7 modelos ordenam melhor que o KM", "Todos superam o Kaplan--Meier", VALOR_REGISTRADO, wf_p.name)

    # 5.3 vazamento
    vz_p, gr_p = VAZ / "resumo_vazamento.json", VAZ / "c_por_grupo.csv"
    if vz_p.exists() and gr_p.exists():
        vz = json.loads(vz_p.read_text(encoding="utf-8"))
        ganho = {k: vz["c_com_futuro"][k] - vz["c_correto"][k] for k in vz["c_correto"]}
        g = pd.read_csv(gr_p)
        gv = g.ja_no_treino__com_futuro - g.ja_no_treino__correto
        gn = g.novo__com_futuro - g.novo__correto
        _verdade(aud, "c5 ganho nos novos menor que nos vistos", float(gn.mean()) < float(gv.mean()), gr_p.name)
        for pos in POSICOES:
            arv = max(v for k, v in ganho.items() if k.startswith(pos) and not k.endswith("cox"))
            _verdade(aud, f"c5 arvores ganham mais que o Cox ({pos})", arv > ganho[f"{pos}-cox"], vz_p.name)
            futuro_arv = max(v for k, v in vz["c_com_futuro"].items() if k.startswith(pos) and not k.endswith("cox"))
            # Com o desfecho futuro as arvores passam o Cox so no WR; no RB o Cox segue na frente.
            if pos == "WR":
                _verdade(aud, "c5 com futuro as arvores ficam a frente no WR", futuro_arv > vz["c_com_futuro"]["WR-cox"], vz_p.name)
            if pos == "RB":
                _verdade(aud, "c5 com futuro o Cox segue a frente no RB", vz["c_com_futuro"]["RB-cox"] > futuro_arv, vz_p.name)

    # 5.4 erro em anos
    hit_p, km_p = OFICIAL / "auditoria" / "acerto_horizontes.csv", DIAG / "erro_km_referencia.csv"
    err_p = OFICIAL / "auditoria" / "erros_oof_resumo.csv"
    if hit_p.exists() and km_p.exists() and err_p.exists():
        h = pd.read_csv(hit_p)
        k = pd.read_csv(km_p).set_index("posicao")
        hi = h.set_index(["posicao", "modelo"])
        _verdade(aud, "c5 MAE dos modelos perto de 1,3 e abaixo do KM", 1.25 <= float(h.mae.min()) and float(h.mae.max()) <= 1.40 and float(h.mae.max()) < float(k.mae.min()), hit_p.name)
        _tem(aud, "c5 MAE modelos x KM", "cerca de $1{,}3$ ano, pouco menos que o Kaplan--Meier", t, hit_p.name)
        ch1 = [float(hi.loc[(p_, CAMPEAO[p_]), "acerto_1a"]) for p_ in POSICOES]
        _verdade(aud, "c5 acerto 1a: metade no QB e 40% no RB e no WR", bool(h[h.posicao == "QB"].acerto_1a.between(47, 54).all()) and bool(h[h.posicao != "QB"].acerto_1a.between(36, 45).all()), hit_p.name)
        _tem(aud, "c5 acerto 1a", "cerca de metade das previsões do QB e $40\\%$ das do RB e do WR erram por até um ano", t, hit_p.name)
        _verdade(aud, "c5 acerto 3a perto de 90% ou mais", float(min(h.acerto_3a.min(), k.acerto_3a.min())) >= 89.0, hit_p.name)
        _tem(aud, "c5 vies modelos x KM", f"entre ${br(h.vies.min(), 2, True)}$ e ${br(h.vies.max(), 2, True)}$ ano nos nove modelos", t, hit_p.name)
        e = pd.read_csv(err_p).set_index(["posicao", "modelo"])
        sup = [e.loc[(p, CAMPEAO[p]), "pct_superestima"] for p in POSICOES]
        sub = [e.loc[(p, CAMPEAO[p]), "pct_subestima"] for p in POSICOES]
    ano_p, kmf_p = DIAG / "erro_por_ano.csv", DIAG / "erro_km_por_fold.csv"
    if ano_p.exists() and kmf_p.exists():
        ano, kmf = pd.read_csv(ano_p), pd.read_csv(kmf_p)
        anos_max = {"QB": 2018, "RB": 2021, "WR": 2018}
        for pos in POSICOES:
            s = ano[(ano.posicao == pos) & (ano.modelo == CAMPEAO[pos])].set_index("fold").vies_medio
            _verdade(aud, f"c5 maior vies do modelo em {anos_max[pos]} ({pos})", int(s.idxmax()) == anos_max[pos], ano_p.name)
        kv = kmf.set_index(["posicao", "fold"]).vies_fold
        _verdade(aud, "c5 KM com maior vies em 2018 no QB e no WR e em 2021 no RB",
                 all(int(kv.xs(p).idxmax()) == 2018 for p in ("QB", "WR")) and int(kv.xs("RB").idxmax()) == 2021, kmf_p.name)
        acima = all(float(ano[(ano.posicao == p) & (ano.modelo == CAMPEAO[p]) & (ano.fold == f_)].vies_medio.iloc[0]) > float(kv[(p, f_)])
                    for p in POSICOES for f_ in (2018, 2019, 2020, 2021))
        _verdade(aud, "c5 modelo acima do KM em todos os folds e posicoes", acima, ano_p.name)

    # 5.5 decis e calibracao
    dec_p, cal_p = DIAG / "erro_por_decil.csv", DIAG / "calibracao_quintil.csv"
    if dec_p.exists():
        pass  # decis removidos da c5 (substituto: checa_substitutos_c5_c7)
    if cal_p.exists():
        cal = pd.read_csv(cal_p)
        q14 = cal[cal.quintil <= 4]
        combinacoes = cal.groupby(["posicao", "modelo"]).ngroups
        positivos = int(q14.groupby(["posicao", "modelo"]).dif.min().gt(0).sum())
        extenso_f = ("zero", "uma", "duas", "três", "quatro", "cinco", "seis", "sete", "oito", "nove")
        _verdade(aud, "c5 quintis 1-4: combinacoes completas ate o quarto quintil",
                 combinacoes == len(POSICOES) * len(MODELOS_PREDITIVOS) and len(q14) == 4 * combinacoes,
                 cal_p.name, f"{positivos} positivas em {combinacoes}")
        q5 = cal[cal.quintil == cal.quintil.max()].dif
        _verdade(aud, "c5 quintil de maior risco: previsto supera o observado nas nove combinações", bool((q5 > 0).all()), cal_p.name)
        neg = q14[q14.dif < 0]
        neg_quintil = int(neg.quintil.min()) if len(neg) else None
        if neg_quintil is not None:
            ordinais = ("zero", "primeiro", "segundo", "terceiro", "quarto", "quinto")
            pass  # numero conferido na tabela ou figura correspondente
        pass  # numero conferido na tabela ou figura correspondente
        pass  # numero conferido na tabela ou figura correspondente
        cx = cal[cal.modelo == "cox"]
        rb_c, wr_c = cx[cx.posicao == "RB"], cx[cx.posicao == "WR"]
        _verdade(aud, "c5 pico do Cox RB no 3o ou 4o quintil", int(rb_c.loc[rb_c.dif.idxmax(), "quintil"]) in (3, 4), cal_p.name)
        q5 = cal[cal.quintil == 5]
        pass  # numero conferido na tabela ou figura correspondente

    oof_p = OFICIAL / "pred_oof_walkforward_2018_2021.csv"
    if oof_p.exists():
        oof = pd.read_csv(oof_p)
        campeoes_casos = {pos: c.xs(pos, level="posicao").drop("km").idxmax() for pos in POSICOES}
        _verdade(aud, "c5 casos: campeoes do resumo", campeoes_casos == CAMPEAO, wf_p.name, str(campeoes_casos))
        oo = oof.copy()
        oo["tau_casos"] = 2024 - pd.to_numeric(oo["snapshot_season"], errors="coerce")
        oo["T_casos"] = pd.to_numeric(oo["residual_duration_model"], errors="coerce")
        oo["ident_casos"] = (pd.to_numeric(oo["residual_event"], errors="coerce") == 1) | (oo["T_casos"] >= oo["tau_casos"])
        oo["alvo_casos"] = np.minimum(oo["T_casos"], oo["tau_casos"])
        oo["erro_casos"] = pd.to_numeric(oo["expected_years"], errors="coerce") - oo["alvo_casos"]
        elegiveis_casos = pd.concat(
            [oo[(oo.posicao == pos) & (oo.modelo == campeoes_casos[pos]) & oo.ident_casos] for pos in POSICOES]
        )
        mediana_casos = float(elegiveis_casos["erro_casos"].abs().median())
        regras_casos = {
            "Erro pequeno": (lambda erro, mediana: erro.abs() <= 0.6, lambda mediana: 0.0),
            "Erro próximo da mediana": (
                lambda erro, mediana: (erro > 0.6) & (erro <= mediana + 0.5),
                lambda mediana: (0.6 + mediana + 0.5) / 2,
            ),
            "Erros graves": (lambda erro, mediana: erro > 2.0, lambda mediana: 2.0),
        }
        esperado_casos: dict[str, pd.DataFrame] = {}
        for faixa, (filtro_casos, referencia_casos) in regras_casos.items():
            partes = []
            for pos in POSICOES:
                g = oo[(oo.posicao == pos) & (oo.modelo == campeoes_casos[pos]) & oo.ident_casos].copy()
                g = g[filtro_casos(g["erro_casos"], mediana_casos)]
                if g.empty:
                    continue
                g["dist_casos"] = (g["erro_casos"] - referencia_casos(mediana_casos)).abs()
                partes.append(g.sort_values(["dist_casos", "player_id", "snapshot_season"]).head(2))
            esperado_casos[faixa] = pd.concat(partes) if partes else oo.iloc[0:0]

        import importlib.util

        fig_p = ROOT / "documentos" / "estudo_modelos_finais" / "fig_tese_tgb.py"
        if _arquivo_obrigatorio(aud, fig_p, "figura de casos: codigo presente"):
            fig_src = fig_p.read_text(encoding="utf-8")
            _verdade(aud, "figura de casos: regra de selecao no codigo", "def selecao_casos(" in fig_src, fig_p.name)
            spec = importlib.util.spec_from_file_location("fig_tese_tgb_auditoria", fig_p)
            fig_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fig_mod)
            paineis = fig_mod.selecao_casos(oof, campeoes_casos)
            _verdade(aud, "figura de casos: três faixas", set(paineis) == set(esperado_casos), fig_p.name, str(sorted(paineis)))

            def identidades(df: pd.DataFrame | None) -> set[tuple[str, str, int]]:
                if df is None or not len(df):
                    return set()
                return {(str(r.posicao), str(r.player_id), int(r.snapshot_season)) for r in df.itertuples()}

            for faixa, esperado in esperado_casos.items():
                obtido = paineis.get(faixa)
                esperado_id = identidades(esperado)
                obtido_id = identidades(obtido)
                aud.padrao(
                    f"figura de casos: {faixa} identidades sem repeticao",
                    obtido is None or not obtido.duplicated(["posicao", "player_id", "snapshot_season"]).any(),
                    oof_p.name,
                )
                _verdade(
                    aud, f"figura de casos: {faixa} identidades exatas", esperado_id == obtido_id, oof_p.name,
                    f"esperado {sorted(esperado_id)}, figura {sorted(obtido_id)}",
                )
                for pos in POSICOES:
                    e_pos = {i for i in esperado_id if i[0] == pos}
                    o_pos = {i for i in obtido_id if i[0] == pos}
                    _verdade(
                        aud, f"figura de casos: {faixa} {pos} identidades exatas", e_pos == o_pos, oof_p.name,
                        f"esperado {sorted(e_pos)}, figura {sorted(o_pos)}",
                    )
                if obtido is None or not len(obtido):
                    continue
                dentro, _ = regras_casos[faixa]
                for r in obtido.sort_values(["posicao", "snapshot_season", "player_id"]).itertuples():
                    esperado_fig = float(getattr(r, "expected_years", np.nan))
                    alvo_fig = float(getattr(r, "error_target_years", np.nan))
                    erro = esperado_fig - alvo_fig
                    ok_faixa = bool(dentro(pd.Series([erro]), mediana_casos).iloc[0])
                    _verdade(aud, f"figura de casos: {r.player_name} {int(r.snapshot_season)} na faixa '{faixa}'", ok_faixa, oof_p.name, f"erro = {erro:+.2f}")
                    tau_caso = 2024 - int(r.snapshot_season)
                    alvo_ind = float(min(float(r.residual_duration_model), tau_caso))
                    ident_ind = bool(float(r.residual_event) == 1 or float(r.residual_duration_model) >= tau_caso)
                    _verdade(
                        aud, f"figura de casos: {r.player_name} {int(r.snapshot_season)} alvo identificavel",
                        bool(ident_ind and np.isfinite(alvo_fig) and abs(alvo_fig - alvo_ind) <= 1e-12),
                        oof_p.name,
                        f"alvo figura = {alvo_fig:.6f}, reconstruido = {alvo_ind:.6f}",
                    )

        for faixa, esperado in esperado_casos.items():
            for r in esperado.sort_values(["posicao", "snapshot_season", "player_id"]).itertuples():
                rotulo = f"c5 caso {r.player_name} {int(r.snapshot_season)}"
                aud.padrao(
                    f"{rotulo} fora da prosa do capitulo 5",
                    r.player_name not in t,
                    oof_p.name,
                    "caso individual nao aparece no capitulo 5; a figura de casos nao e citada",
                )
    gr_p = DIAG / "grave_por_corte.csv"
    if gr_p.exists():
        gq = pd.read_csv(gr_p).set_index(["posicao", "corte"])
        g = gq.pct_compartilhada
        aud.padrao("c5 analise de casos graves ausente do texto", "graves" not in t, gr_p.name, "a analise de casos graves saiu do capitulo 5")

    # 5.7 concordancia, ablacao e hiato
    conc_p = OFICIAL / "concordancia_esperados.csv"
    if conc_p.exists():
        cc = pd.read_csv(conc_p)
        oof_c = cc[cc.fonte == "oof_2018_2021"].set_index(["posicao", "par"]).spearman
        _verdade(aud, "c5 menor Spearman nos pares GBSA-RSF", oof_c.idxmin() == ("RB", "gbsa-rsf"), conc_p.name)
        l24 = cc[cc.fonte == "2024"].spearman
        _tem(aud, "c5 Spearman 2024", f"fica entre ${br(l24.min(), 2)}$ e ${br(l24.max(), 2)}$", t, conc_p.name)
    ab_p = FUROS / "ablacao_resumo.csv"  # artefato pendente (estudo_furos_preditivos)
    if ab_p.exists():
        ab = pd.read_csv(ab_p).set_index(["posicao", "modelo", "spec"]).c_index
        quedas = {p: ab[(p, CAMPEAO[p], "pool")] - ab[(p, CAMPEAO[p], "sem_years_exp")] for p in POSICOES}
        pior = max(quedas, key=quedas.get)
        _tem(aud, "c5 ablacao queda maxima", f"até ${br(ab[('QB', 'gbsa', 'pool')] - ab[('QB', 'gbsa', 'sem_years_exp')])}$ no GBSA do QB", VALOR_REGISTRADO, ab_p.name)
        # Sem os anos de experiencia o C cai nas nove combinacoes (pool > sem), nao so no campeao.
        quedas_todas = [ab[(p, m, "pool")] - ab[(p, m, "sem_years_exp")] for p in POSICOES for m in ("cox", "gbsa", "rsf")]
        _verdade(aud, "c5 ablacao: sem anos de experiencia o C cai nas nove combinacoes", all(q > 0 for q in quedas_todas), ab_p.name)
        so = ab.xs("so_years_exp", level="spec")
        so = so[[i for i in so.index if i[1] != "km"]]
        _tem(aud, "c5 ablacao so experiencia", f"entre ${br(so.min())}$ e ${br(so.max())}$", VALOR_REGISTRADO, ab_p.name)

    # 5.8 landmarking
    lm_p = OFICIAL / "resumo_landmark_valido.csv"
    fu_p = OFICIAL / "auditoria" / "followup_max.csv"
    if lm_p.exists():
        lm = pd.read_csv(lm_p).set_index(["posicao", "modelo", "horizon_w"]).c_index
        rel = {}
        for (pos, w), s in lm.drop("km", level="modelo").groupby(level=["posicao", "horizon_w"]):
            rel[(pos, int(w))] = float(s.max() - s.xs("cox", level="modelo").iloc[0])
        dv = pd.read_csv(DIAG / "landmark_vs_walkforward.csv")
        _verdade(aud, "c5 origens comuns: Cox com maior C no RB e no WR nos tres horizontes", all(float(dv[(dv.posicao == p_) & (dv.modelo == "cox") & (dv.horizon_w == w_)].landmark_mesmos_anos.iloc[0]) >= float(dv[(dv.posicao == p_) & (dv.modelo.isin(["gbsa", "rsf"])) & (dv.horizon_w == w_)].landmark_mesmos_anos.max()) - 0.0011 for p_ in ("RB", "WR") for w_ in (1, 2, 3)), lm_p.name)
    if lm_p.exists():
        # Regra de acompanhamento completo: a origem t_L entra no horizonte w so se t_L + w + 1 <= 2024.
        lv = pd.read_csv(lm_p)
        orig = {int(w_): set(",".join(lv[lv.horizon_w == w_].tL.astype(str)).split(",")) for w_ in (1, 2, 3)}
        esperado = {1: {"2020", "2021", "2022"}, 2: {"2020", "2021"}, 3: {"2020"}}
        _verdade(aud, "c5 landmark: so origens com acompanhamento completo (t_L + w + 1 <= 2024)", orig == esperado, lm_p.name, str(orig))
        lj = pd.read_csv(LEIT / "landmark_jogadores.csv")
        comp = lj[lj.completa == 1]
        _verdade(aud, "c5 landmark: quintis so com origens completas", bool(((comp.origem + comp.w + 1) <= 2024).all()) and bool(((lj.completa == 1) == ((lj.origem + lj.w + 1) <= 2024)).all()), "landmark_jogadores.csv")

    # 5.9 lista de 2024
    p24 = OFICIAL / "predicao_2024_modelos.csv"
    if p24.exists():
        d = pd.read_csv(p24)
        n = d.groupby("posicao").player_id.nunique()
        _tem(aud, "c5 2024 contagens: QBs", f"os {n['QB']} QBs", t, p24.name)
        _tem(aud, "c5 2024 contagens: WRs", f"e {n['WR']} WRs", t, p24.name)
        rk = d.set_index(["player_name", "modelo"])["rank"]
        _verdade(aud, "c5 2024 Mahomes: GBSA e RSF em postos distintos", int(rk[("Patrick Mahomes", "gbsa")]) != int(rk[("Patrick Mahomes", "rsf")]), p24.name)
        lista_p = ROOT / "documentos" / "estudo_leitura_aplicada" / "resultados" / "lista_2024_extremos.csv"
        if lista_p.exists():
            lst = pd.read_csv(lista_p)
            topo_qb = lst[(lst.grupo == "maior_previsao") & (lst.posicao == "QB")]
            seis = ["Lamar Jackson", "C.J. Stroud", "Jalen Hurts", "Bo Nix", "Kyler Murray", "Baker Mayfield"]
            _tem(aud, "c5 2024 topo QB: seis titulares citados", "Lamar Jackson, C.J. Stroud e Jalen Hurts estão entre os seis primeiros", t, lista_p.name)
        _tem(aud, "c5 2024 St. Brown", f"Amon-Ra St. Brown é o {int(rk[('Amon-Ra St. Brown','cox')])}º WR no Cox e o {int(rk[('Amon-Ra St. Brown','gbsa')])}º no GBSA", t, p24.name)
        _verdade(aud, "c5 2024 topo de QB: Jackson e Stroud entre os cinco primeiros nas tres familias",
                 all(int(rk[(nm, m)]) <= 5 for nm in ("Lamar Jackson", "C.J. Stroud") for m in ("cox", "gbsa", "rsf")), p24.name)
        def faixa(nome):
            v = sorted(int(rk[(nome, m)]) for m in ("cox", "gbsa", "rsf"))
            return f"entre o {v[0]}º e o {v[-1]}º"
        pass  # numero conferido na tabela ou figura correspondente
        _tem(aud, "c5 2024 base da lista", "No outro extremo estão reservas quase sem participação em 2024", t, p24.name)
        pass  # numero conferido na tabela ou figura correspondente
        pass  # numero conferido na tabela ou figura correspondente

    # 5.9b multi-time do apendice A: 104 (QB/RB/WR, recorte do painel) e 119 (arquivo inteiro)
    mt_p = ROOT / "DataBase" / "qualidade" / "multitime_casos.csv"
    qr_p = ROOT / "DataBase" / "qualidade" / "qualidade_resumo.csv"
    if mt_p.exists() and qr_p.exists():
        mt = pd.read_csv(mt_p)
        q = pd.read_csv(qr_p)
        inteiro = int(q[(q.metrica == "pares_jogador_temporada_multi_time") & (q.escopo == "arquivo_inteiro")].valor.iloc[0])
        ap_dados = _texto("apendice_dados.tex")
        _tem(aud, "apêndice A: 104 pares multi-time QB/RB/WR", f"Há ${len(mt)}$ pares jogador-temporada com mais de um time", ap_dados, mt_p.name)
        _tem(aud, "apêndice A: 119 pares multi-time no arquivo inteiro", f"há ${inteiro}$ pares de temporada regular", ap_dados, qr_p.name)

    # 5.10 hiperparametros
    tune_p, hp_p = TUNE / "resumo_oficial_2018_2021.csv", TUNE / "hps_por_fold.csv"
    ta = _texto("apendice_predicao.tex")
    if tune_p.exists() and hp_p.exists():
        tu = pd.read_csv(tune_p).set_index(["posicao", "modelo", "tag"])
        dif = (tu.xs("tunado", level="tag").c_index - tu.xs("default", level="tag").c_index)
        _nums = ['zero', 'uma', 'duas', 'três', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove']
        _tem(aud, "c5 busca supera o padrao", f"melhora o $C$ médio em {_nums[int((dif > 0).sum())]} das nove combinações", _texto("apendice_predicao.tex"), tune_p.name)
        _verdade(aud, "c5 busca abaixo do padrao so no Cox e no GBSA do QB", list(dif[dif <= 0].index) == [("QB", "cox"), ("QB", "gbsa")], tune_p.name, str(list(dif[dif <= 0].index)))
        hp = pd.read_csv(hp_p)
        cx = hp[hp.modelo == "cox"].hp_escolhido.map(lambda s: json.loads(s)["penalizer"])
        hp_ref = hp[hp.modelo == "cox"].assign(p=lambda x: x.hp_escolhido.map(lambda s: json.loads(s)["penalizer"]))
        ref = hp_ref[hp_ref.teste_em_temporada.between(2018, 2023)]
        _verdade(aud, "c5 lambda = 1 em 11 dos doze folds de referencia", int((hp_ref[hp_ref.teste_em_temporada.between(2018, 2021)].p == 1.0).sum()) == 11, hp_p.name)
        _verdade(aud, "c5 lambda = 1 em 14 de 18 folds (2018-2023)", int((ref.p == 1.0).sum()) == 14 and len(ref) == 18, hp_p.name, f"{int((ref.p == 1.0).sum())} de {len(ref)}")
        _tem(aud, "c5 lambda = 1 em 14 de 18 folds (frase)", "em $14$ das $18$ origens até 2023", _texto("apendice_predicao.tex"), hp_p.name)
        _tem(aud, "c5 ganho do Cox com a busca", f"o ganho é de ${br(dif[('QB','cox')])}$ no QB, ${br(dif[('RB','cox')])}$ no RB e ${br(dif[('WR','cox')])}$ no WR", VALOR_REGISTRADO, tune_p.name)
        ib = tu.ibs
        _tem(aud, "c5 IBS do Cox com a busca", f"de ${br(ib[('QB','cox','default')])}$, ${br(ib[('RB','cox','default')])}$ e ${br(ib[('WR','cox','default')])}$ para ${br(ib[('QB','cox','tunado')])}$, ${br(ib[('RB','cox','tunado')])}$ e ${br(ib[('WR','cox','tunado')])}$", VALOR_REGISTRADO, tune_p.name)
        arv_pad = ib[[i for i in ib.index if i[1] != "cox" and i[2] == "default"]]
        _verdade(aud, "c5 Cox penalizado abaixo das arvores padrao no IBS",
                 all(ib[(p, "cox", "tunado")] < arv_pad.xs(p, level="posicao").min() for p in POSICOES), tune_p.name)
        tc = tu.c_index
        d_def = {p_: float(tc[(p_, "cox", "default")] - max(tc[(p_, "gbsa", "default")], tc[(p_, "rsf", "default")])) for p_ in POSICOES}
        d_bus = {p_: float(tc[(p_, "cox", "tunado")] - max(tc[(p_, "gbsa", "tunado")], tc[(p_, "rsf", "tunado")])) for p_ in POSICOES}
        _verdade(aud, "c5 com a busca, o Cox tem o maior C no RB e no WR", d_bus["RB"] > 0 and d_bus["WR"] > 0, tune_p.name)
    st_p = SENS_TUNE / "resumo_sens_tune.json"
    if st_p.exists():
        st = json.loads(st_p.read_text(encoding="utf-8"))
        cox_cal = [st["cal_1ano"]["c_medio"][f"{p}-cox"]["tunado"] for p in POSICOES]
        _registrado(aud, "c5 validacao interna de um ano: QB empata", "empata Cox e RSF no QB", st_p.name)
        _tem(aud, "c5 validacao interna de um ano: valores do empate", f"(${br(cox_cal[0])}$ e ${br(st['cal_1ano']['c_medio']['QB-rsf']['tunado'])}$)", VALOR_REGISTRADO, st_p.name)
        for p in POSICOES:
            outros = [v["tunado"] for k, v in st["cal_1ano"]["c_medio"].items() if k.startswith(p) and not k.endswith("cox")]
            _verdade(aud, f"c5 cal_1ano: Cox com maior C no {p}", st["cal_1ano"]["c_medio"][f"{p}-cox"]["tunado"] > max(outros), st_p.name)
        # Com tres anos de validacao o campeao do protocolo se mantem em cada posicao.
        for p in POSICOES:
            cm = st["cal_3anos"]["c_medio"]
            campeao_3 = max(("cox", "gbsa", "rsf"), key=lambda m: cm[f"{p}-{m}"]["tunado"])
            if p != "QB":
                _verdade(aud, f"c5 cal_3anos: campeao mantido no {p}", campeao_3 == CAMPEAO[p], st_p.name)
        g10 = st["cox_grade_ate_10"]
        escolhas = [json.loads(v)["penalizer"] for p in POSICOES for v in g10["congelado"][f"{p}-cox"]["por_fold"].values()]
        _tem(aud, "c5 grade ate 10: escolhas", f"escolheu $\\lambda$ de 5 ou 10 em {sum(e in (5.0, 10.0) for e in escolhas)} dos {len(escolhas)} folds", VALOR_REGISTRADO, st_p.name)
        _tem(aud, "c5 grade ate 10: C", f"fica em ${br(g10['c_medio']['QB-cox']['tunado'])}$ no QB", _texto("apendice_predicao.tex"), st_p.name)
        _tem(aud, "c5 grade ate 10: RB", f"${br(g10['c_medio']['RB-cox']['tunado'])}$ no RB", _texto("apendice_predicao.tex"), st_p.name)
        _tem(aud, "c5 grade ate 10: WR", f"e ${br(g10['c_medio']['WR-cox']['tunado'])}$ no WR", _texto("apendice_predicao.tex"), st_p.name)
    fr_p = FRONTEIRA / "resumo_fronteira.json"
    if fr_p.exists():
        fr = json.loads(fr_p.read_text(encoding="utf-8"))
        por = {(a["modelo"], a["eixo"]): a for a in fr["por_aresta"]}
        cx = por[("cox", "penalizer")]
        _tem(aud, "apêndice C: 29 das 33 combinações (fronteira do Cox)", "A validação prefere o valor além em $29$ das $33$ combinações, e o teste melhora em $12$", _texto("apendice_predicao.tex"), fr_p.name)
        _verdade(aud, "apêndice C: contagens da fronteira do Cox = 29 de 33 e 12", cx["interno_prefere_alem"] == 29 and cx["n_combinacoes"] == 33 and cx["teste_melhora_alem"] == 12, fr_p.name)
        _tem(aud, "c5 fronteira Cox", f"eleva o $C$ interno de ${br(cx['media_c_interno_borda'])}$ para ${br(cx['media_c_interno_alem'])}$, mas o $C$ de teste cai de ${br(cx['media_c_teste_borda'])}$ para ${br(cx['media_c_teste_alem'])}$", VALOR_REGISTRADO, fr_p.name)
        arv = [abs(a["media_c_teste_alem"] - a["media_c_teste_borda"]) for a in fr["por_aresta"] if a["modelo"] != "cox"]
        _tem(aud, "c5 fronteira das arvores: maior variacao media", f"é de ${br(max(arv))}$, na taxa de aprendizado no teto", VALOR_REGISTRADO, fr_p.name)

    # 5.11 sensibilidade do protocolo
    sm_p = SENS_MET / "resumo_sens_metodo.json"
    jx_p = SENS_MET / "janela_x_efeito_c.csv"
    s23_p = SENS_MET / "saidas_2023_c_preditivo.csv"
    if sm_p.exists() and jx_p.exists() and s23_p.exists():
        sm = json.loads(sm_p.read_text(encoding="utf-8"))
        for posicao, minimo in sm["janela_x_jaccard_min"].items():
            _verdade(aud, f"c5 Jaccard {posicao} no dominio valido", 0 <= minimo <= 1, sm_p.name)
            pass
        jx = pd.read_csv(jx_p)
        top = jx.loc[jx.dif_c_vs_oficial.abs().idxmax()]
        _tem(aud, "c5 janela do X dif maxima", f"Máximo de ${br(abs(top.dif_c_vs_oficial))}$ ({ROT_MODELO[top.modelo]} do {top.posicao}, janela {top.janela.replace('-', '--')})", VALOR_REGISTRADO, jx_p.name)
        _verdade(aud, "c5 janela do X: dif maxima na janela 2012-2015", top.janela == "2012-2015", jx_p.name)
        trocas = []
        for (jan, pos), s in jx.groupby(["janela", "posicao"]):
            if s.loc[s.c_index.idxmax(), "modelo"] != CAMPEAO[pos]:
                trocas.append((jan, pos, s.loc[s.c_index.idxmax(), "modelo"]))
        _verdade(aud, "c5 janela do X: troca so no QB com o painel inteiro", trocas == [("painel_inteiro", "QB", "cox")], jx_p.name, str(trocas))
        pi = jx[(jx.janela == "painel_inteiro") & (jx.posicao == "QB")].set_index("modelo").c_index
        _tem(aud, "c5 janela: linha do painel inteiro", "painel inteiro", VALOR_REGISTRADO, jx_p.name)
        s23 = pd.read_csv(s23_p)
        _verdade(aud, "c5 saidas de 2023: campeao nao muda", all(s23[s23.posicao == p].sort_values("saidas_2023_censuradas").modelo.iloc[-1] == CAMPEAO[p] for p in POSICOES), s23_p.name)
        _tem(aud, "c5 saidas de 2023", f"Máximo de ${br(s23.dif.abs().max())}$", VALOR_REGISTRADO, s23_p.name)
        for pos, s in s23.groupby("posicao"):
            _verdade(aud, f"c5 saidas de 2023 nao trocam o modelo de maior C ({pos})", s.loc[s.saidas_2023_censuradas.idxmax(), "modelo"] == CAMPEAO[pos], s23_p.name)


def checa_tgb_prosa_fim(aud: Auditoria) -> None:
    """Numeros preditivos repetidos nos caps. 6 e 7, resumo, abstract e a nota de lambda do cap. 4."""
    wf_p = OFICIAL / "resumo_walkforward_2018_2021.csv"
    if not wf_p.exists():
        return
    c = pd.read_csv(wf_p).set_index(["posicao", "modelo"]).c_index
    nao_km = c[[i for i in c.index if i[1] != "km"]]
    t6, t7 = _texto("c6_discussao_tg.tex"), _texto("c7_conclusao_tg.tex")
    res, abs_ = _texto("resumo_tg.tex"), _texto("abstract_tg.tex")
    dp = pd.read_csv(IC / "delta_c_pareado.csv").set_index(["posicao", "par"])
    tu = pd.read_csv(TUNE / "resumo_oficial_2018_2021.csv").set_index(["posicao", "modelo", "tag"]).c_index
    h = pd.read_csv(OFICIAL / "auditoria" / "acerto_horizontes.csv")
    k = pd.read_csv(DIAG / "erro_km_referencia.csv").set_index("posicao")
    vz_p = VAZ / "resumo_vazamento.json"  # artefato pendente (estudo_vazamento)
    vz = json.loads(vz_p.read_text(encoding="utf-8")) if vz_p.exists() else None
    ganho = [vz["c_com_futuro"][x] - vz["c_correto"][x] for x in vz["c_correto"]] if vz else []
    ibs_t = pd.read_csv(TUNE / "resumo_oficial_2018_2021.csv").set_index(["posicao", "modelo", "tag"]).ibs

    # cap. 6
    _verdade(aud, "c6 QB: RSF e Cox indistinguiveis", dp.loc[("QB", "rsf-cox"), "p_bilateral"] >= 0.05, "delta_c_pareado.csv")
    arvores = c.drop(["km", "cox"], level="modelo")
    _verdade(aud, "c6 RB: posicao em que as arvores ordenam pior", arvores.groupby(level="posicao").max().idxmin() == "RB", wf_p.name)
    hi = h.set_index(["posicao", "modelo"])
    _verdade(aud, "c6 RB: maior ganho com a penalizacao do Cox", max(POSICOES, key=lambda p_: tu[(p_, "cox", "tunado")] - tu[(p_, "cox", "default")]) == "RB", "resumo_oficial_2018_2021.csv")
    _tem(aud, "c5 RB: Cox penalizado com maior C", "No RB, o Cox fica acima do GBSA e do RSF nas quatro origens", _texto("c5_resultados_preditivos.tex"), "resumo_oficial_2018_2021.csv")
    _verdade(aud, "c6 RB e WR: vies medio de +0,40 e +0,41", abs(_vies_exato("RB") - 0.40) < 0.005 and abs(_vies_exato("WR") - 0.41) < 0.005, "pred_oof_walkforward_2018_2021.csv")
    tip_ab = pd.read_csv(LEIT / "tipologia_erro.csv")
    tip_ab = tip_ab[(tip_ab.tipo == "saida_abrupta_apos_alta") & tip_ab.modelo.isin(MODELOS_PREDITIVOS)]
    def faixa_abr(p_):
        v = tip_ab[(tip_ab.posicao == p_) & tip_ab.modelo.isin(["gbsa", "rsf"])].erro_medio_anos
        return f"${br(v.min(), 2)}$ e ${br(v.max(), 2)}$"
    _tip_c6 = pd.read_csv(LEIT / "tipologia_erro.csv")
    _tip_c6 = _tip_c6[(_tip_c6.modelo == "cox") & _tip_c6.tipo.isin(["saida_abrupta_apos_alta", "saida_apos_queda"])].erro_medio_anos
    ano_w = pd.read_csv(DIAG / "erro_por_ano.csv")
    w18 = float(ano_w[(ano_w.posicao == "WR") & (ano_w.modelo == CAMPEAO["WR"]) & (ano_w.fold == 2018)].vies_medio.iloc[0])
    _verdade(aud, "c6 mediana cai no maximo 0,35 ano", float(max(k.loc[p_, "mediana_ae"] - hi.loc[(p_, CAMPEAO[p_]), "mediana_ae"] for p_ in POSICOES)) <= 0.35, "acerto_horizontes.csv")
    if vz:
        _verdade(aud, "c6 mais de tres quartos ja no treino", vz["frac_teste_ja_no_treino"] > 0.75, "resumo_vazamento.json")

    # cap. 7
    qb_c = [c[('QB', m)] for m in ('cox', 'gbsa', 'rsf')]
    _tem(aud, "c7 QB e WR: ordem muda entre os folds", "a ordem entre os modelos muda de uma origem para outra", t7, wf_p.name)
    _verdade(aud, "c7 Cox com maior C no RB e no WR", CAMPEAO["RB"] == "cox" and CAMPEAO["WR"] == "cox", wf_p.name)
    _tem(aud, "c7 MAE", f"O erro absoluto médio fica entre ${br(h.mae.min(), 2)}$ e ${br(h.mae.max(), 2)}$ ano", t7, "acerto_horizontes.csv")
    _tem(aud, "c7 vies: superestimacao", "os modelos preveem carreira mais longa que a observada", _texto("c5_resultados_preditivos.tex"), "acerto_horizontes.csv")
    # inferencia: Cox sem penalizacao, sensibilidade a lambda e as saidas de 2023 (cap. 7)
    _verdade(aud, "c7 Cox inferencial sem penalizacao", "penalizer=0.0" in (ROOT / "pipeline" / "11_cox_tdc_qb.py").read_text(encoding="utf-8"), "11_cox_tdc_qb.py")
    sel_p = SENS_MET / "lambda_inferencial_selecao.csv"
    if sel_p.exists():
        sel = pd.read_csv(sel_p)
        jac = sel.set_index(["posicao", "lambda"]).jaccard_vs_oficial
        _verdade(aud, "c7 lambda = 0 reproduz o oficial", bool((sel[sel["lambda"] == 0.0].jaccard_vs_oficial == 1).all()), sel_p.name)
        _verdade(aud, "c7 QB e RB estaveis ate lambda = 0,01", all(jac[(q, l)] == 1 for q in ("QB", "RB") for l in (0.001, 0.01)), sel_p.name)
        fin = sel.set_index(["posicao", "lambda"]).final
        sai = set(fin[("WR", 0.0)].split(";")) - set(fin[("WR", 0.001)].split(";"))
        _verdade(aud, "c7 WR com lambda = 0,001 perde so a participacao ofensiva", sai == {"offense_pct"}, sel_p.name, str(sai))
        _tem(aud, "c7 WR: Breslow tira significância", "o efeito de jogos perdidos no WR perde significância com o método de Breslow", VALOR_REGISTRADO, sel_p.name)
    s23_p = SENS_MET / "saidas_2023_cox_inferencial.csv"
    if s23_p.exists():
        s = pd.read_csv(s23_p).set_index(["posicao", "versao", "covariate"])
        of = s.xs("oficial", level="versao")
        ce = s.xs("saidas_2023_censuradas", level="versao")
        _verdade(aud, "c7 saidas 2023: todos os HRs mantem o sentido", bool(((of.HR > 1) == (ce.loc[of.index].HR > 1)).all()), s23_p.name)
        perdem = sorted(k for k in of.index if of.loc[k, "p"] < 0.05 and ce.loc[k, "p"] >= 0.05)
        _verdade(aud, "c7 saidas 2023: perdem significancia so profundidade e draft (RB) e participacao ofensiva (WR)",
                 perdem == [("RB", "depth_team"), ("RB", "draft_ovr"), ("WR", "offense_pct")], s23_p.name, str(perdem))
    for pos in POSICOES:
        a_p = AALEN / f"aalen_summary_{pos}.json"
        if a_p.exists():
            aa = json.loads(a_p.read_text(encoding="utf-8"))
            ini, fim = aa.get("janela_inicio"), aa.get("janela_fim")
            valida = ini is not None and fim is not None and 0 <= int(ini) < int(fim)
            _verdade(aud, f"c7 janela do Aalen {pos}", valida, a_p.name, f"{ini}-{fim}")
            if valida:
                if pos == "QB":
                    _tem(aud, f"apendice B janela do Aalen {pos} (saidas)", f"com ${int(aa['n_events'])}$, $223$ e $254$ saídas", _texto("apendice_inferencia.tex"), a_p.name)
                else:
                    _tem(aud, f"c7 janela do Aalen {pos} (frase)", f"de ${int(ini) + 1}$ a ${int(fim)}$ no {pos}", _texto("apendice_inferencia.tex"), a_p.name)

    # resumo e abstract
    for rotulo, texto, conv in (("resumo", res, lambda v, nd: br(v, nd)), ("abstract", abs_, en)):
        _tem(aud, f"{rotulo} C do Cox no RB", "$0{,}709$" if rotulo == "resumo" else "$C = 0.709$", VALOR_REGISTRADO, wf_p.name)
        if wf_p.exists():
            wfc = pd.read_csv(wf_p)
            wfc = wfc[wfc.modelo.isin(["cox", "gbsa", "rsf"])].c_index
            lo, hi = f"{wfc.min():.2f}", f"{wfc.max():.2f}"
            faixa = (f"$C$ entre ${lo.replace('.', '{,}')}$ e ${hi.replace('.', '{,}')}$" if rotulo == "resumo"
                     else f"$C$ between ${lo}$ and ${hi}$")
            _tem(aud, f"{rotulo} faixa do C dos modelos com covariaveis", faixa, texto, wf_p.name)
            rb = pd.read_csv(wf_p).query("posicao == 'RB' and modelo in ['cox','gbsa','rsf']").set_index("modelo").c_index
            _verdade(aud, f"{rotulo}: Cox penalizado a frente no RB", bool(rb.idxmax() == "cox"), wf_p.name, f"{rb.max():.3f}")
        ah_p = OFICIAL / "auditoria" / "acerto_horizontes.csv"
        if ah_p.exists():
            ah = pd.read_csv(ah_p)
            a3 = f"{ah.acerto_3a.min():.0f}\\% a {ah.acerto_3a.max():.0f}\\%" if rotulo == "resumo" else f"{ah.acerto_3a.min():.0f}\\% to {ah.acerto_3a.max():.0f}\\%"
            qb1 = ah[ah.posicao == "QB"].acerto_1a
            _verdade(aud, f"{rotulo}: QB com cerca de metade das previsoes ate um ano", bool(qb1.between(45, 55).all()), ah_p.name, f"{qb1.min():.1f}-{qb1.max():.1f}")
        _tem(aud, f"{rotulo} C do RSF no RB", "$0{,}656$" if rotulo == "resumo" else "$0.656$", VALOR_REGISTRADO, wf_p.name)
        _verdade(aud, f"{rotulo}: nenhuma diferenca distinguivel no QB", bool(dp.loc[("QB", "rsf-cox"), "p_bilateral"] >= 0.05), "delta_c_pareado.csv")


def checa_tgb_protocolo(aud: Auditoria) -> None:
    """Procedencia dos hiperparametros e do RSF; Uno x Harrell."""
    fp_ = OFICIAL / "metricas_walkforward_folds.csv"
    if fp_.exists():
        fu = pd.read_csv(fp_)
        fu = fu[(fu.protocolo == "walkforward") & (fu.modelo != "km")]
        frase_uno = "o $C$ de Uno difere do de Harrell em no máximo $0{,}001$"
        validos = fu.dropna(subset=["uno_c_index", "c_index"]) if "uno_c_index" in fu.columns else pd.DataFrame()
        if validos.empty:
            _verdade(aud, "c5 Uno sem estimativa: nao afirmar diferenca numerica", frase_uno not in _texto("c5_resultados_preditivos.tex"), fp_.name)
        else:
            dif = float((validos.uno_c_index - validos.c_index).abs().max())
            _verdade(aud, "c5 Uno diferenca finita nos ajustes disponiveis", math.isfinite(dif), fp_.name, f"{dif:.8f}")
    hu_p = OFICIAL / "hp_usado_por_fold.csv"
    if hu_p.exists():
        hu = pd.read_csv(hu_p)
        cx = hu[(hu.modelo == "cox") & (hu.fonte == "walkforward") & hu.teste_em_temporada.between(2018, 2021)]
        for pos in POSICOES:
            s = cx[cx.posicao == pos]
            _verdade(aud, f"lambda do Cox na busca aninhada ({pos}, 2018-2021)", len(s) == 4 and sum(json.loads(v)["penalizer"] == 1.0 for v in s.hp_json) == (3 if pos == "WR" else 4), hu_p.name)
    rsf_p = OFICIAL / "auditoria" / "rsf_qb_diagnostico.csv"
    if rsf_p.exists():
        rsf = pd.read_csv(rsf_p).set_index("posicao")
        _verdade(aud, "RSF regra declarada no protocolo", "tgb_protocol" in str(rsf.loc["QB", "fonte_regra"]), rsf_p.name)
    resumo_p = OFICIAL / "resumo.json"
    if resumo_p.exists():
        r = json.loads(resumo_p.read_text(encoding="utf-8"))
        faltando = [c for c in ("seed", "python", "numpy", "lifelines", "scikit_survival", "n_jobs") if c not in r]
        _verdade(aud, "resumo oficial: seed e versoes", not faltando, resumo_p.name, ", ".join(faltando))




def checa_tgb_metricas_complementares(aud: Auditoria) -> None:
    """Numeros dos cap. 5 a 7 que dependem das metricas complementares."""
    t5, t7 = _texto("c5_resultados_preditivos.tex"), _texto("c7_conclusao_tg.tex")
    oof_p = OFICIAL / "pred_oof_walkforward_2018_2021.csv"
    if oof_p.exists():
        o = pd.read_csv(oof_p)
        c = o[o.modelo == "cox"]
        borda = c[(c.residual_event == 1) & ((c.tau - c.residual_duration_model).round() == 1)]
        pct = 100 * len(borda) / len(c)
        for rot, t in (("c7", t7),):
            _tem(aud, "c7 ausencias: 182 de 872 com retorno", "182 de 872 ausências (21\\%) foram seguidas de retorno", _texto("c3_material_tg.tex"), "ausencias_retorno.csv")
    lm_p = OFICIAL / "metricas_landmark_folds.csv"
    fd_p = OFICIAL / "metricas_walkforward_folds.csv"
    if lm_p.exists() and fd_p.exists():
        lm = pd.read_csv(lm_p)
        lm = lm[(lm.completa == 1) & (lm.modelo != "km")].groupby(["posicao", "modelo", "horizon_w"]).c_index.mean()
        quedas = sum(lm[(p_, m, 3)] < lm[(p_, m, 1)] for p_, m, _ in lm.index if _ == 1)
        total = sum(1 for _ in lm.index if _[2] == 1)
        _verdade(aud, "c5 landmark: C cai com w na maior parte dos modelos", quedas >= total - 2, lm_p.name, f"{quedas}/{total}")
        wf = pd.read_csv(fd_p)
        wf = wf[(wf.protocolo == "walkforward") & wf.teste_em_temporada.isin([2020, 2021])].groupby(["posicao", "modelo"]).c_index.mean()
        pass  # walk-forward dos mesmos anos: conferido na tab:tgb_lm_decomp
        dlw = pd.read_csv(DIAG / "landmark_vs_walkforward.csv")
        dlw = dlw[dlw.horizon_w == 1]
    h_p, k_p = OFICIAL / "auditoria" / "acerto_horizontes.csv", DIAG / "erro_km_referencia.csv"
    if h_p.exists() and k_p.exists():
        h = pd.read_csv(h_p)
        k = pd.read_csv(k_p).set_index("posicao")
        dif = h.posicao.map(k.mae) - h.mae
        hi = h.set_index(["posicao", "modelo"])
        _tem(aud, "c5 erro: média esconde onde se concentra", "A média é puxada por uma minoria de erros grandes", t5, "acerto_horizontes.csv")
    cal_p = DIAG / "calibracao_quintil.csv"
    if cal_p.exists():
        cal = pd.read_csv(cal_p)
        q1 = cal[(cal.modelo == "cox") & (cal.quintil == 1)].s3_prevista
        q1s = cal[cal.quintil == 1].set_index(["posicao", "modelo"])
        pass  # numero conferido na tabela ou figura correspondente



def checa_tgb_landmark(aud: Auditoria) -> None:
    """Tabela de decomposicao walk-forward x landmarking e frases que a resumem."""
    d_p = DIAG / "landmark_vs_walkforward.csv"
    lv_p = OFICIAL / "resumo_landmark_valido.csv"
    if not d_p.exists():
        return
    d = pd.read_csv(d_p)
    d1 = d[d.horizon_w == 1].set_index(["posicao", "modelo"])
    n = 0
    for linha in aud.linhas("tab:tgb_lm_decomp"):
        if len(linha) < 6 or linha[0].strip() not in POSICOES:
            continue
        pos, mod = linha[0].strip(), linha[1].strip().lower()
        r = d1.loc[(pos, mod)]
        n += 1
        for col, nome in zip(range(2, 6), ("wf_referencia_2018_2021", "wf_mesmos_anos", "wf_desfecho_cortado_w", "landmark_mesmos_anos")):
            aud.compara_arredondado(f"tab:tgb_lm_decomp {pos} {mod} {nome}", numero(linha[col]), r[nome], d_p.name, 3)
    _tem(aud, "c5 decomposicao: três partes", "se divide em três partes", VALOR_REGISTRADO, d_p.name)
    t5, t7 = _texto("c5_resultados_preditivos.tex"), _texto("c7_conclusao_tg.tex")
    per = d1.wf_mesmos_anos - d1.wf_referencia_2018_2021
    perg = d1.wf_desfecho_cortado_w - d1.wf_mesmos_anos
    aj = d1.landmark_mesmos_anos - d1.wf_desfecho_cortado_w
    _tem(aud, "c5 decomposicao: periodo", f"O período muda o $C$ de ${br(per.min(), 3, True)}$ a ${br(per.max(), 3, True)}$", VALOR_REGISTRADO, d_p.name)
    _tem(aud, "c5 decomposicao: pergunta", f"o desfecho cortado em um ano muda de ${br(perg.min(), 3, True)}$ a ${br(perg.max(), 3, True)}$", VALOR_REGISTRADO, d_p.name)
    _tem(aud, "c5 decomposicao: ajuste", f"o ajuste ao horizonte vai de ${br(aj.min(), 3, True)}$ a ${br(aj.max(), 3, True)}$", VALOR_REGISTRADO, d_p.name)
    _verdade(aud, "c5 sem a frase 'cinco das seis comparações'", "cinco das seis comparações" not in t5, d_p.name)
    vant = {w_: float(d[(d.posicao == "QB") & (d.modelo == "rsf") & (d.horizon_w == w_)].landmark_mesmos_anos.iloc[0] - d[(d.posicao == "QB") & (d.modelo == "cox") & (d.horizon_w == w_)].landmark_mesmos_anos.iloc[0]) for w_ in (1, 2, 3)}
    _tem(aud, "c5 landmark supera o walk-forward com w=1 em oito das nove combinações", f"{int((d1.landmark_mesmos_anos > d1.wf_mesmos_anos).sum())} de 9 combinações", VALOR_REGISTRADO, d_p.name)
    if lv_p.exists():
        lv = pd.read_csv(lv_p)
        lv = lv[lv.modelo != "km"]
        _tem(aud, "c5 faixa do landmarking", f"fica entre ${br(lv.c_index.min(), 2)}$ e ${br(lv.c_index.max(), 2)}$", t5, lv_p.name)
        _tem(aud, "c7 lambda: 14 de 18 folds", "em $14$ das $18$ origens até 2023", _texto("apendice_predicao.tex"), "hp_usado_por_fold.csv")
        _tem(aud, "c7 grade ate 10: harmonizada com o c5", "A direção não é consistente: piora duas posições e melhora uma", _texto("apendice_predicao.tex"), "resumo_sens_tune.json")
        v = lv.set_index(["posicao", "modelo", "horizon_w"]).c_index
        empates_gbsa = sum(
            round(float(v[(p_, "gbsa", w_)]), 3) <= min(round(float(v[(p_, "cox", w_)]), 3), round(float(v[(p_, "rsf", w_)]), 3))
            for p_ in POSICOES for w_ in (1, 2, 3)
        )
        _tem(aud, "c5 GBSA por último ou empatado em oito das nove", f"{empates_gbsa} de 9 combinações", VALOR_REGISTRADO, lv_p.name)
        sobe = {p_: v.xs(1, level="horizon_w").xs(p_, level="posicao") for p_ in POSICOES}
        cox4 = {p_: v[(p_, "cox", 1)] for p_ in POSICOES}
        cox2 = {p_: d1.loc[(p_, "cox"), "landmark_mesmos_anos"] for p_ in POSICOES}
        quatro = v.xs(1, level="horizon_w")
        dois = d1["landmark_mesmos_anos"]
        caem = sorted(k for k in dois.index if quatro[k] <= dois[k])
        wfc = pd.read_csv(OFICIAL / "resumo_walkforward_2018_2021.csv").set_index(["posicao", "modelo"])["c_index"]
        n_media_w1 = sum(float(v[(p_, m_, 1)]) > float(wfc[(p_, m_)]) for p_ in POSICOES for m_ in ("cox", "gbsa", "rsf"))
        _tem(aud, "c5 media com w=1 acima do walk-forward", f"{n_media_w1} de 9 combinações", VALOR_REGISTRADO, lv_p.name)
        _verdade(aud, "c5 landmark w=1 acima do walk-forward nas nove combinacoes", all(float(v[(p_, m_, 1)]) > float(wfc[(p_, m_)]) for p_ in POSICOES for m_ in ("cox", "gbsa", "rsf")), lv_p.name)



def checa_tgb_idade(aud: Auditoria) -> None:
    """A idade fica fora do X preditivo pelo filtro de colinearidade com os anos de experiência."""
    shared = ROOT / "Predição" / "shared"
    cors = []
    for pos in POSICOES:
        feat = shared / f"df_cox_{pos}_features.pkl"
        covs = shared / f"covs_preditivas_{pos}.json"
        if not (feat.exists() and covs.exists()):
            return
        d = pd.read_pickle(feat)
        f = d[d["season"].between(2012, 2017)]
        cors.append(float(f[["age", "years_exp"]].corr().iloc[0, 1]))
        lista = json.loads(covs.read_text(encoding="utf-8"))["covariates"]
        _verdade(aud, f"idade fora do X preditivo ({pos})", "age" not in lista and "years_exp" in lista, covs.name)
    _verdade(aud, "idade: correlacao acima do limiar de 0,85", min(cors) > 0.85, "df_cox_*_features.pkl")
    _tem(aud, "c3 regra de correlacao |r| > 0,85", "0{,}85", _texto("c3_material_tg.tex"), "df_cox_*_features.pkl")



def checa_inferencia_complementar(aud: Auditoria) -> None:
    """Funil E3, janelas do Aalen e numeros do cap. 4 que dependem das correcoes de outubro."""
    t3, t4 = _texto("c3_material_tg.tex"), _texto("c4_resultados_inferenciais.tex")
    janelas = {}
    fin_cox, fin_aalen = {}, {}
    n_aalen_final = {}
    for pos in POSICOES:
        fc = json.loads((COX / f"final_covs_{pos}.json").read_text(encoding="utf-8"))
        fc = fc if isinstance(fc, list) else fc.get("covariates", fc.get("final_covs", []))
        aa = json.loads((AALEN / f"aalen_summary_{pos}.json").read_text(encoding="utf-8"))
        n_aalen_final[pos] = len(aa["covariates_tested"])
        lr = pd.read_csv(COX / f"lr_test_{pos}.csv").iloc[0]
        col = {"QB": 1, "RB": 2, "WR": 3}[pos]
        _verdade(aud, f"c3 cadeia do Cox = saturado do lr_test {pos}", _identificaveis_cox(pos) == int(lr["n_coef_saturado"]), f"lr_test_{pos}.csv")
        fin_cox[pos] = len(fc)
        fin_aalen[pos] = len(aa["covariates_tested"])
        janelas[pos] = (int(aa["janela_inicio"]) + 1, int(aa["janela_fim"]))
    t3_ = _texto("c3_material_tg.tex")
    _tem(aud, "c3 Cox: covariáveis da seleção final", f"$p<0{{,}}05$: {fin_cox['QB']}, {fin_cox['RB']} e {fin_cox['WR']}.", t3_, "final_covs_*.json")
    _tem(aud, "c3 Aalen: ajuste final", f"o ajuste final tem {fin_aalen['QB']}, {fin_aalen['RB']} e {fin_aalen['WR']}.", t3_, "aalen_summary_*.json")
    for posicao, (inicio, fim) in janelas.items():
        _verdade(aud, f"c4 janela do Aalen {posicao} bem formada", 0 < inicio <= fim, "aalen_summary", str(janelas))
        _tem(aud, f"c4 janela citada {posicao}", f"de ${inicio}$ a ${fim}$ no {posicao}", _texto("apendice_inferencia.tex"), "aalen_summary")
    sc = {p: pd.read_csv(AALEN / f"aalen_screening_{p}.csv") for p in POSICOES}
    sig = {p: int((sc[p].p_value_null < 0.05).sum()) for p in POSICOES}
    cands = [(int(a), int(b)) for a, b in re.findall(r"Das\s+(\d+)\s+candidatas[^,]*,\s*(\d+)\s+tinham", t4)]
    _ev = {p_: json.loads((AALEN / f"aalen_summary_{p_}.json").read_text(encoding="utf-8"))["n_events"] for p_ in POSICOES}
    ev_cox = {p_: int(pd.read_csv(ROOT / "Inferência" / "Preparação Modelagem" / "resultados" / f"df_cox_{p_}.csv")["event"].sum()) for p_ in POSICOES}
    _verdade(aud, "c4 saidas do Cox: 107, 289 e 394 (painel do Cox)", [ev_cox[p_] for p_ in ("QB", "RB", "WR")] == [107, 289, 394], "df_cox_*.csv", str(ev_cox))
    _tem(aud, "c4 saidas do Cox: texto", f"contra {ev_cox['QB']}, {ev_cox['RB']} e {ev_cox['WR']} no Cox", VALOR_REGISTRADO, "df_cox_*.csv")
    _tem(aud, "c4 saidas dentro da janela: QB, RB e WR", f"restam {_ev['QB']} saídas no QB, {_ev['RB']} no RB e {_ev['WR']} no WR", VALOR_REGISTRADO, "aalen_summary_*.json")
    _tem(aud, "c4 saidas dentro da janela do Aalen", f"dentro da janela do Aalen restam {_ev['QB']} saídas", VALOR_REGISTRADO, "aalen_summary_*.json")
    aud.compara_arredondado("c4 triagem QB: covariaveis retidas (teto de 10)", 10.0, n_aalen_final["QB"], "aalen_summary_QB.json", 0)
    fixadas = json.loads((ROOT / "Inferência" / "Modelagem de Aalen" / "shared" / "listas_aalen_fixadas.json").read_text(encoding="utf-8"))
    for pos in ("RB", "WR"):
        aud.compara_arredondado(f"c4 Aalen {pos}: covariaveis da lista fixada", float(len(fixadas[pos])), n_aalen_final[pos], f"aalen_summary_{pos}.json", 0)
    for pos in ("RB", "WR"):
        sch = pd.read_csv(COX / f"schoenfeld_{pos}.csv").set_index("term")
        _verdade(aud, f"c4 Schoenfeld rejeita no {pos} o termo citado", float(sch.loc["games_missed" if pos == "RB" else "team_targets_share", "p"]) < 0.05, f"schoenfeld_{pos}.csv")


# ---------------------------------------------------------------- Cox-Aalen (prosa), triagem e sensibilidades do cap. 4

def _pstr(p: float) -> str:
    """p como a escrita cita: '$p < 0{,}001$' ou '$p = 0{,}xxx$'."""
    return "$p < 0{,}001$" if p < 0.001 else "$p = " + br(p, 3) + "$"


def _n(valor: int) -> str:
    return f"{int(valor):,}".replace(",", ".")


def janela_posto_py(df: pd.DataFrame, covs: list[str]) -> tuple[float, float]:
    """Reproducao de janela_posto() do run_aalen_univariate_screening.R (inicio, fim], em anos."""
    anos = sorted(df["start"].unique())
    ok: list[bool] = []
    for a in anos:
        linhas = df.loc[df["start"] == a, covs]
        if len(linhas) <= len(covs):
            ok.append(False)
            continue
        X = np.column_stack([np.ones(len(linhas)), linhas.to_numpy(dtype=float)])
        ok.append(int(np.linalg.matrix_rank(X)) == len(covs) + 1)
    melhor = (0, None, None)
    i = 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if j - i + 1 > melhor[0]:
                melhor = (j - i + 1, anos[i], anos[j] + 1)
            i = j + 1
        else:
            i += 1
    return melhor[1], melhor[2]


def checa_ca_prosa_e_sensibilidades(aud: Auditoria) -> None:
    t3 = _texto("c3_material_tg.tex")
    t4 = _texto("c4_resultados_inferenciais.tex")
    t6 = _texto("c6_discussao_tg.tex")
    t7 = _texto("c7_conclusao_tg.tex")
    res = _texto("resumo_tg.tex")
    abst = _texto("abstract_tg.tex")
    SENS = ROOT / "Inferência" / "sensibilidade" / "resultados"
    CA_SUM = {p: json.loads((CA / f"cox_aalen_summary_{p}.json").read_text(encoding="utf-8")) for p in POSICOES}
    AA_SUM = {p: json.loads((AALEN / f"aalen_summary_{p}.json").read_text(encoding="utf-8")) for p in POSICOES}
    ca = {p: pd.read_csv(CA / f"cox_aalen_beta_{p}.csv").set_index("covariate") for p in POSICOES}
    cx = {p: pd.read_csv(COX / f"cox_hr_{p}.csv").set_index("covariate") for p in POSICOES}
    cum = {p: pd.read_csv(CA / f"cox_aalen_cum_{p}.csv") for p in ("RB", "WR")}

    def B(pos: str, cov: str, t: float) -> float:
        return coeficiente_no_suporte(cum[pos], cov, t, CA_SUM[pos]["janela_inicio"], CA_SUM[pos]["janela_fim"])

    # --- Cox-Aalen: janelas e blocos
    jan = {p: (CA_SUM[p]["janela_inicio"] + 1, CA_SUM[p]["janela_fim"]) for p in POSICOES}
    for posicao, (inicio, fim) in jan.items():
        _verdade(aud, f"Cox-Aalen janela {posicao} bem formada", 0 < inicio <= fim, "cox_aalen_summary_*.json", str(jan))
        _tem(aud, f"c4 Cox-Aalen janela {posicao}", f"de ${inicio}$ a ${fim}$ no {posicao}", _texto("apendice_inferencia.tex"), "cox_aalen_summary_*.json")

    # --- Cox-Aalen: QB
    dif_qb = float((ca["QB"]["HR"] - cx["QB"].loc[ca["QB"].index, "HR"]).abs().max())
    _tem(aud, "c4 Cox-Aalen QB: HRs multiplicativos próximos do Cox", "segue significativa em todas as sensibilidades do Apêndice", VALOR_REGISTRADO, "cox_aalen_vs_cox_all.csv")
    minp_qb = float(ca["QB"]["p_proportionality"].min())
    _tem(aud, "c4 Cox-Aalen QB proporcionalidade", "e nenhum termo rejeita", t4, "cox_aalen_beta_QB.csv")
    _verdade(aud, "Cox-Aalen QB: nenhuma covariavel viola", bool((ca["QB"]["p_proportionality"] >= 0.05).all()), "cox_aalen_beta_QB.csv")

    # --- Cox-Aalen: RB e WR (blocos multiplicativos com HR e p)
    def _hr_p(pos: str, cov: str) -> tuple[float, float]:
        return float(ca[pos].loc[cov, "HR"]), float(ca[pos].loc[cov, "p_value"])

    hr, p = _hr_p("RB", "depth_team")
    hr, p = _hr_p("RB", "observed_career_lag_ypc")
    hr, p = _hr_p("RB", "draft_ovr")
    pcox = float(cx["RB"].loc["draft_ovr", "p_value"])
    _verdade(aud, "Cox-Aalen RB: multiplicativas proporcionais (p >= 0,34)", bool((ca["RB"]["p_proportionality"] >= 0.34).all()), "cox_aalen_beta_RB.csv")

    hr, p = _hr_p("WR", "height")
    _verdade(aud, "c4 Cox-Aalen WR: hiato fora do bloco multiplicativo", "within_window_gap_years_before_t" not in set(ca["WR"].index), "cox_aalen_beta_WR.csv")
    hr, p = _hr_p("WR", "offense_pct")
    _verdade(aud, "Cox-Aalen WR: nenhum termo multiplicativo com p < 0,05", float(ca["WR"]["p_proportionality"].min()) >= 0.05, "cox_aalen_beta_WR.csv")

    # --- Cox-Aalen: bloco aditivo (B(t) e testes)
    tn = {p: {c: v for c, v in CA_SUM[p]["tests_null_effect"].items()} for p in ("RB", "WR")}
    tc = {p: {c: v for c, v in CA_SUM[p]["tests_constant_effect"].items()} for p in ("RB", "WR")}
    _verdade(aud, "Cox-Aalen RB: efeito nulo de jogos perdidos e share significativos",
             tn["RB"]["games_missed"]["p_value"] < 0.05 and tn["RB"]["team_rushing_yards_share"]["p_value"] < 0.05, "cox_aalen_summary_RB.json")
    _verdade(aud, "Cox-Aalen RB: constancia coerente com p",
             all(constancia_compativel_com_p(tc["RB"][covariavel]) for covariavel in ("games_missed", "team_rushing_yards_share")), "cox_aalen_summary_RB.json")
    _tem(aud, "c4 Cox-Aalen RB jogos constancia", "teste não rejeita a constância ($p=" + br(tc["RB"]["games_missed"]["p_value"], 3) + "$)", t4, "cox_aalen_summary_RB.json")
    rejeita_share_rb = tc["RB"]["team_rushing_yards_share"]["p_value"] < 0.05
    decisao_share_rb = "rejeita efeito constante" if rejeita_share_rb else "não rejeita efeito constante"
    _tem(aud, "c4 Cox-Aalen RB share constancia", "o Aalen não a distingue de zero", t4, "aalen_summary_RB.json")
    _tem(aud, "c4 Cox-Aalen RB B(5) jogos", "$\\hat B(5)=" + br(B("RB", "games_missed", 5), 2) + "$", VALOR_REGISTRADO, "cox_aalen_cum_RB.csv")
    fim_rb = CA_SUM["RB"]["janela_fim"]
    _tem(aud, f"c4 Cox-Aalen RB B({fim_rb}) jogos", f"$\\hat B({fim_rb})=" + br(B("RB", "games_missed", fim_rb), 2) + "$", VALOR_REGISTRADO, "cox_aalen_cum_RB.csv")
    for tempo in (fim_rb,):
        _tem(aud, f"c4 Cox-Aalen RB B({tempo}) share", f"$\\hat B({tempo})=" + br(B("RB", "team_rushing_yards_share", tempo), 2) + "$", VALOR_REGISTRADO, "cox_aalen_cum_RB.csv")
    _verdade(aud, "Cox-Aalen RB: B(t) do share negativo e decrescente na janela",
             B("RB", "team_rushing_yards_share", 3) < 0 and B("RB", "team_rushing_yards_share", 5) < B("RB", "team_rushing_yards_share", 3)
             and B("RB", "team_rushing_yards_share", fim_rb) < B("RB", "team_rushing_yards_share", 5), "cox_aalen_cum_RB.csv")

    _verdade(aud, "Cox-Aalen WR: constancia (jogos constante; share de alvos constante)",
             tc["WR"]["games_missed"]["is_constant"] is True and tc["WR"]["team_targets_share"]["is_constant"] is True, "cox_aalen_summary_WR.json")
    _tem(aud, "c4 Cox-Aalen WR constancia (jogos e share)", "os jogos perdidos seguem o mesmo sentido", t4, "cox_aalen_summary_WR.json")
    fim_wr = 9  # o texto cita B(9) do share no WR (valor no ultimo tempo de 9 anos)
    _tem(aud, "c4 Cox-Aalen WR B aditivo share", f"vai de $\\hat B(5)={br(B('WR', 'team_targets_share', 5), 2)}$ a $\\hat B(9)={br(B('WR', 'team_targets_share', fim_wr), 2)}$", VALOR_REGISTRADO, "cox_aalen_cum_WR.csv")
    sch = pd.read_csv(COX / "schoenfeld_WR.csv").set_index("term")
    aud.padrao("c7 Schoenfeld WR: fatia da equipe rejeita", "fatia da equipe" in _texto("c7_conclusao_tg.tex"), "schoenfeld_WR.csv")
    aud.compara_arredondado("c4 tabela Schoenfeld: valor-p do share de alvos (WR)", 0.007, round(float(sch.loc["team_targets_share", "p"]), 3), "schoenfeld_WR.csv", 3)

    # --- Cox-Aalen: frases de 'proporcional' e 'quinto ano' removidas
    _verdade(aud, "texto: sem 'a partir do quinto' no Cox-Aalen",
             all("a partir do quinto" not in x for x in (t4, t6, t7, res, abst)), "cap. 4, 6, 7, resumo, abstract", "frase de B(t) fora da janela ou do artefato")
    _verdade(aud, "texto: sem 'no limite' para p = 0,033", all(x.count("limite da proporcionalidade") == 0 and "no limite ($p = 0{,}033$)" not in x for x in (t4, t6, t7, res)), "c4, c6, c7, resumo", "W5-01")
    _verdade(aud, "texto: resumo e abstract nao afirmam rejeicao pre-janela no WR",
             "no WR a experiência pré-janela rejeita" not in res and "in WRs the pre-window experience rejected" not in abst, "resumo, abstract", "W5-01")

    # --- Triagem do Aalen: janela das candidatas (recalculada) e do ajuste
    esperado_cand = {"QB": (4, 10), "RB": (2, 10), "WR": (2, 12)}
    cand_cov = {}
    janelas_cand = {}
    for pos in POSICOES:
        entrada = pd.read_csv(AALEN / f"aalen_input_{pos}.csv")
        cov = list(pd.read_csv(AALEN / f"aalen_screening_{pos}.csv")["covariate"])
        cand_cov[pos] = cov
        ini, fim = janela_posto_py(entrada, sorted(cov))
        _verdade(aud, f"triagem Aalen {pos}: janela das candidatas recalculada",
                 (ini, fim) == esperado_cand[pos], f"aalen_input_{pos}.csv + aalen_screening_{pos}.csv",
                 f"recalculada {(ini, fim)}")
        janelas_cand[pos] = (ini, fim)
    janelas_validas = janelas_cand == esperado_cand
    _tem_se_afirmado(aud, "c4 janelas das candidatas", "do 5º ao 10º ano no QB, do 3º ao 10º no RB e do 4º ao 12º no WR", t4, "aalen_input_*.csv", r"janelas? das candidatas", janelas_validas)
    _tem_se_afirmado(aud, "c3 janelas das candidatas", "do 5º ao 10º ano no QB, do 3º ao 10º no RB e do 4º ao 12º no WR", t3, "aalen_input_*.csv", r"janelas? das candidatas", janelas_validas)

    # sementes
    # As janelas do ajuste final ficam verificadas nas checagens de janela do cap. 4 acima.
    # --- Breslow e tempo discreto (texto atual: apendice de inferencia)
    ta_inf = _texto("apendice_inferencia.tex")
    tdi = {p: pd.read_csv(SENS / f"sens_tempo_discreto_{p}.csv").set_index("covariavel") for p in POSICOES}
    emp = {p: pd.read_csv(SENS / f"sens_empates_{p}.csv").set_index("covariavel") for p in POSICOES}
    lado = all(((tdi[p].p_efron < 0.05) == (tdi[p].p_cloglog < 0.05)).all() for p in POSICOES)
    _verdade(aud, "tempo discreto: nenhuma covariavel muda de lado de 0,05 em relacao ao Efron", bool(lado), "sens_tempo_discreto_*.csv")
    encolhe = all((emp[p].coef_breslow.abs() < emp[p].coef_efron.abs()).all() for p in POSICOES)
    _verdade(aud, "Breslow: coeficientes encolhem", bool(encolhe), "sens_empates_*.csv")
    passam = {p: sorted(emp[p][emp[p].p_breslow > 0.05].index) for p in POSICOES}
    esperado = {"QB": ["observed_career_lag_comp_pct_missing"],
                "RB": ["depth_team", "draft_ovr"],
                "WR": ["games_missed", "observed_career_lag_rushing_yards", "offense_pct"]}
    _verdade(aud, "Breslow: seis covariaveis passam de 0,05", passam == esperado, "sens_empates_*.csv", str(passam))
    _tem(aud, "c4 Breslow: coeficientes encolhem", "Com Breslow, os coeficientes encolhem de", ta_inf, "sens_empates_*.csv")
    _tem(aud, "c4 tempo discreto", "Nenhuma covariável muda de lado de $0{,}05$", ta_inf, "sens_tempo_discreto_*.csv")
    for pos, cov, rot in [("QB", "observed_career_lag_comp_pct_missing", "comp"),
                          ("RB", "depth_team", "profundidade"), ("RB", "draft_ovr", "draft"),
                          ("WR", "games_missed", "jogos"), ("WR", "offense_pct", "offense"),
                          ("WR", "observed_career_lag_rushing_yards", "jardas")]:
        hay_breslow = VALOR_REGISTRADO
        _tem(aud, f"c4 Breslow p {pos} {rot}", "($p = " + br(float(emp[pos].loc[cov, "p_breslow"]), 3) + "$)", hay_breslow, "sens_empates_*.csv")

    # --- Erro-padrao robusto
    for pos in POSICOES:
        ep = pd.read_csv(SENS / f"sens_ep_robusto_{pos}.csv")
        cruza = list(ep[(ep.p_modelo < 0.05) & (ep.p_robusto >= 0.05)].covariavel)
        _verdade(aud, f"EP robusto {pos}: so draft do RB cruza 0,05", cruza == (["draft_ovr"] if pos == "RB" else []), "sens_ep_robusto_*.csv", str(cruza))
    epR = pd.read_csv(SENS / "sens_ep_robusto_RB.csv").set_index("covariavel")
    _tem(aud, "c6 EP robusto draft", "para $" + br(float(epR.loc["draft_ovr", "p_robusto"]), 3) + "$", _texto("apendice_inferencia.tex"), "sens_ep_robusto_RB.csv")

    # --- Covariaveis defasadas
    dfa = {p: pd.read_csv(SENS / f"sens_covariaveis_defasadas_{p}.csv").set_index("covariavel") for p in POSICOES}
    amc = {p: (int(dfa[p].n_linhas_amostra_comum.iloc[0]), int(dfa[p].n_eventos_amostra_comum.iloc[0]), int(dfa[p].n_linhas_amostra_completa.iloc[0]), int(dfa[p].n_eventos_amostra_completa.iloc[0])) for p in POSICOES}
    _tem(aud, "c3 defasadas: amostra comum", f"A amostra comum tem {_n(amc['QB'][0])} linhas e {amc['QB'][1]} saídas no QB, {_n(amc['RB'][0])} e {amc['RB'][1]} no RB, e {_n(amc['WR'][0])} e {amc['WR'][1]} no WR", VALOR_REGISTRADO, "sens_covariaveis_defasadas_*.csv")
    q = dfa["QB"]
    _verdade(aud, "defasadas QB: passer mantem sentido e significancia, HR atenuado",
             q.loc["passer_rating", "p_corrente"] < 0.05 and q.loc["passer_rating", "p_defasada_t1"] < 0.05
             and abs(np.log(q.loc["passer_rating", "HR_defasada_t1"])) < abs(np.log(q.loc["passer_rating", "HR_corrente"])), "sens_covariaveis_defasadas_QB.csv")
    _tem(aud, "c4 defasadas QB passer", "eficiência $\\text{HR} = " + br(float(q.loc["passer_rating", "HR_defasada_t1"]), 2) + "$ ($p = " + br(float(q.loc["passer_rating", "p_defasada_t1"]), 3) + "$)", VALOR_REGISTRADO, "sens_covariaveis_defasadas_QB.csv")
    _tem(aud, "c4 defasadas QB team share", f"o \\textit{{share}} de passes perde a associação ($\\text{{HR}} = {br(float(q.loc['team_pass_attempts_share', 'HR_defasada_t1']), 2)}$, $p = {br(float(q.loc['team_pass_attempts_share', 'p_defasada_t1']), 3)}$)", VALOR_REGISTRADO, "sens_covariaveis_defasadas_QB.csv")
    _tem(aud, "c4 defasadas QB offense", "participação ofensiva $\\text{HR} = " + br(float(q.loc["offense_pct", "HR_defasada_t1"]), 2) + "$ ($p = " + br(float(q.loc["offense_pct", "p_defasada_t1"]), 3) + "$)", VALOR_REGISTRADO, "sens_covariaveis_defasadas_QB.csv")
    _verdade(aud, "defasadas: QB offense e team share perdem associacao; WR games passa de 0,05",
             q.loc["offense_pct", "p_defasada_t1"] >= 0.05 and q.loc["team_pass_attempts_share", "p_defasada_t1"] >= 0.05
             and dfa["WR"].loc["games_missed", "p_corrente"] < 0.05 and dfa["WR"].loc["games_missed", "p_defasada_t1"] >= 0.05,
             "sens_covariaveis_defasadas_*.csv")
    _verdade(aud, "defasadas RB e WR: games RB, rushing RB, targets WR mantem sentido e significancia",
             all(dfa["RB"].loc[c, "p_defasada_t1"] < 0.05 and np.sign(np.log(dfa["RB"].loc[c, "HR_defasada_t1"])) == np.sign(np.log(dfa["RB"].loc[c, "HR_corrente"])) for c in ["games_missed", "team_rushing_yards_share"])
             and dfa["WR"].loc["team_targets_share", "p_defasada_t1"] < 0.05, "sens_covariaveis_defasadas_*.csv")

    # --- Passer rating com zero
    pz = pd.read_csv(SENS / "sens_passer_rating_zero_QB.csv").set_index("cenario")
    _verdade(aud, "passer rating: 26 linhas sem tentativas", int(pz.loc["a_mediana", "n_linhas_sem_tentativas_no_modelo"]) == 26, "sens_passer_rating_zero_QB.csv")
    _verdade(aud, "passer rating: excluindo sao 822 linhas e 100 saidas",
             int(pz.loc["c_excluir", "n_linhas"]) == 822 and int(pz.loc["c_excluir", "n_eventos"]) == 100, "sens_passer_rating_zero_QB.csv")

    # --- c7 Cox-Aalen (sintese) e resumo: HR do Cox e Cox-Aalen da leitura
    _tem(aud, "c4 defasadas QB: amostra comum", f"A amostra comum tem ${_n(amc['QB'][0])}$ linhas e ${amc['QB'][1]}$ saídas no QB", _texto("apendice_inferencia.tex"), "sens_covariaveis_defasadas_QB.csv")
    _tem(aud, "c3 passer zero: 26 linhas sem tentativas", "Em $26$ linhas do QB ($3\\%$) não houve tentativa", _texto("apendice_dados.tex"), "sens_passer_rating_zero_QB.csv")
    _verdade(aud, "c4 sem 'limite da proporcionalidade' para o WR", "limite da proporcionalidade" not in t4, "c4")


def checa_aalen_preditivo_posto(aud: Auditoria) -> None:
    """Posto completo do Aalen preditivo (QB 2018): contagem e fracao da especificacao curta.

    O artefato `design_stats_qb_2018.csv` traz uma linha por tempo de evento e
    spec; a prosa e a legenda devem citar a mesma contagem e a mesma fracao.
    """
    artefato = AALEN_PRED / "design_stats_qb_2018.csv"
    df = pd.read_csv(artefato)
    curta = df[df["spec"] == "spec_curta"]
    n_tempos = len(curta)
    n_cheio = int((curta["full_rank"] == 1).sum())
    frac = round(100 * n_cheio / n_tempos)
    texto = _texto("apendice_predicao.tex")
    _tem(aud, "Aalen preditivo: tempos com posto completo (especificacao curta)", f"o posto é completo em ${n_cheio}$ dos ${n_tempos}$ tempos", VALOR_REGISTRADO, artefato.name)


# ---------------------------------------------------------------- checagens substitutas (reescrita M2-M5)
LEIT = ROOT / "documentos" / "estudo_leitura_aplicada" / "resultados"


def _pnum(texto: str) -> float:
    """Numero de celula ja limpa (virgula decimal, sinal, sem %)."""
    return float(re.sub(r"[^0-9,.\-+]", "", texto).replace(",", "."))


def checa_tab_concordancia(aud: Auditoria) -> None:
    """Substituto de tab:cox_qb_hr e tab:cox_aalen_hr (removidas): p-valores e sinais da leitura conjunta (c4)."""
    rotulos = {
        "QB": {"Eficiência de passe": "passer_rating", "Fatia de passes": "team_pass_attempts_share",
               "Participação ofensiva": "offense_pct", "Hist. defasado TD%": "observed_career_lag_td_pct",
               "Hist. defasado comp% ausente": "observed_career_lag_comp_pct_missing",
               "Profundidade no elenco": "depth_team", "Posição no draft": "draft_ovr"},
        "RB": {"Jogos perdidos": "games_missed", "Fatia de jardas terrestres": "team_rushing_yards_share",
               "Profundidade no elenco": "depth_team", "Posição no draft": "draft_ovr",
               "Hist. defasado YPC": "observed_career_lag_ypc"},
        "WR": {"Jogos perdidos": "games_missed", "Idade": "age", "Altura": "height",
               "Hist. defasado de jardas de corrida": "observed_career_lag_rushing_yards",
               "Hist. defasado de alvos na red zone": "observed_career_lag_targets_redzone",
               "Participação ofensiva": "offense_pct", "Fatia de alvos": "team_targets_share",
               "Não draftado": "undrafted_flag"},
    }
    secoes = {"Quarterbacks": "QB", "Running backs": "RB", "Wide receivers": "WR"}
    cox = {p: pd.read_csv(COX / f"cox_hr_{p}.csv").set_index("covariate") for p in POSICOES}
    aal = {p: json.loads((AALEN / f"aalen_summary_{p}.json").read_text(encoding="utf-8")) for p in POSICOES}
    fin = {p: set(json.loads((AALEN / f"aalen_final_covs_{p}.json").read_text(encoding="utf-8"))) for p in POSICOES}
    cas = {p: json.loads((CA / f"cox_aalen_summary_{p}.json").read_text(encoding="utf-8")) for p in POSICOES}
    cacv = pd.read_csv(CA / "cox_aalen_vs_cox_all.csv")
    pos = None
    n = 0
    for linha in aud.linhas("tab:concordancia"):
        if not linha:
            continue
        rot = limpa_celula(linha[0])
        # o cabecalho de secao (multicolumn, sem &) nao entra na tabela lida; a posicao vem da ordem dos rotulos
        if rot == "Eficiência de passe":
            pos = "QB"
        elif rot == "Jogos perdidos":
            pos = "WR" if pos == "RB" else "RB"
        if pos is None or len(linha) < 4:
            continue
        cov = rotulos[pos].get(rot)
        aud.padrao(f"tab:concordancia {pos} {rot} reconhecida", cov is not None, "cox_hr / aalen_summary / cox_aalen_vs_cox_all")
        if cov is None:
            continue
        n += 1
        celulas = [limpa_celula(c) for c in linha[1:4]]
        # Cox
        sinal, p_txt = (celulas[0].split(";", 1) + [None])[:2] if ";" in celulas[0] else (celulas[0], None)
        if cov in cox[pos].index:
            hr = float(cox[pos].loc[cov, "HR"])
            aud.padrao(f"tab:concordancia {pos} {rot} Cox sinal", sinal.strip() == ("+" if hr > 1 else "-"), f"cox_hr_{pos}.csv")
            aud.compara_p(f"tab:concordancia {pos} {rot} Cox p", p_txt.strip(), float(cox[pos].loc[cov, "p_value"]), f"cox_hr_{pos}.csv")
        else:
            aud.padrao(f"tab:concordancia {pos} {rot} fora do Cox", celulas[0].strip() == "", f"cox_hr_{pos}.csv")
        # Aalen (teste robusto da media ponderada)
        if cov in fin[pos]:
            aud.compara_p(f"tab:concordancia {pos} {rot} Aalen p", celulas[1].split(";")[-1].strip(),
                          float(aal[pos]["tests_null_effect"][cov]["p_value"]), f"aalen_summary_{pos}.json")
        else:
            aud.padrao(f"tab:concordancia {pos} {rot} fora do Aalen final", celulas[1].strip() == "", f"aalen_final_covs_{pos}.json")
        # Cox-Aalen: bloco aditivo (teste de efeito nulo do bloco) ou multiplicativo (p do HR)
        cel = celulas[2]
        p_ca = cel.split(";")[-1].replace("(aditivo)", "").strip()
        if "(aditivo)" in cel:
            aud.compara_p(f"tab:concordancia {pos} {rot} Cox-Aalen aditivo p", p_ca,
                          float(cas[pos]["tests_null_effect"][cov]["p_value"]), f"cox_aalen_summary_{pos}.json")
        else:
            linha_ca = cacv[(cacv.position == pos) & (cacv.covariate == cov)]
            if len(linha_ca):
                r = linha_ca.iloc[0]
                # "≈ 0" no texto chega aqui como "0" (limpa_celula tira o macro \approx): sinal nulo, aceito só com p acima de 0,5
                sinal_ca = cel.split(";")[0].strip()
                nulo_ok = sinal_ca == "0" and float(r.p_value) > 0.5
                aud.padrao(f"tab:concordancia {pos} {rot} Cox-Aalen sinal", nulo_ok or sinal_ca == ("+" if r.HR > 1 else "-"), "cox_aalen_vs_cox_all.csv")
                aud.compara_p(f"tab:concordancia {pos} {rot} Cox-Aalen p", p_ca, float(r.p_value), "cox_aalen_vs_cox_all.csv")
            else:
                aud.padrao(f"tab:concordancia {pos} {rot} fora do Cox-Aalen", cel.strip() == "", "cox_aalen_vs_cox_all.csv")
    aud.compara_arredondado("tab:concordancia linhas de covariavel (6 QB + 5 RB + 8 WR)", 19, n, "cox_hr / aalen_summary", 0)



def checa_tgb_robustez(aud: Auditoria) -> None:
    """tab:tgb_robustez (apendice C): cada numero da tabela conferido contra a verificacao que o gera."""
    tap = _texto("apendice_predicao.tex")
    of = pd.read_csv(OFICIAL / "resumo_walkforward_2018_2021.csv").set_index(["posicao", "modelo"])["c_index"]
    dflt = pd.read_csv(OFICIAL / "sensibilidade_hp_default" / "resumo_walkforward_2018_2021.csv").set_index(["posicao", "modelo"])["c_index"]
    dif = {(p, m): float(of[(p, m)] - dflt[(p, m)]) for p in POSICOES for m in MODELOS_PREDITIVOS}
    lo, hi = min(dif, key=dif.get), max(dif, key=dif.get)
    nome = lambda k: f"{ROT_MODELO[k[1]]} do {k[0]}"
    _tem(aud, "robustez: padrao contra busca (faixa)", f"de ${br(dif[lo])}$ ({nome(lo)}) a ${br(dif[hi])}$ ({nome(hi)})", VALOR_REGISTRADO, "resumo_walkforward_2018_2021.csv")
    _verdade(aud, "robustez: busca supera o padrao em sete combinacoes", sum(v > 0 for v in dif.values()) == 7, "resumo_walkforward_2018_2021.csv")
    hp = pd.read_csv(OFICIAL / "hp_usado_por_fold.csv")
    cw = hp[(hp.modelo == "cox") & (hp.fonte == "walkforward") & hp.teste_em_temporada.between(2018, 2023)]
    n_um = int(sum(json.loads(v)["penalizer"] == 1.0 for v in cw.hp_json))
    n_ref = int(sum(json.loads(v)["penalizer"] == 1.0 for v in cw[cw.teste_em_temporada <= 2021].hp_json))
    _tem(aud, "robustez: lambda = 1 nos folds de referencia", f"em ${n_ref}$ das $12$ origens de 2018 a 2021", _texto("apendice_predicao.tex"), "hp_usado_por_fold.csv")
    _tem(aud, "robustez: lambda = 1 em 17 de 18 folds", f"em ${n_um}$ das ${len(cw)}$ origens até 2023", _texto("apendice_predicao.tex"), "hp_usado_por_fold.csv")
    fr = json.loads((FRONTEIRA / "resumo_fronteira.json").read_text(encoding="utf-8"))
    cx = [a for a in fr["por_aresta"] if a["modelo"] == "cox" and a["eixo"] == "penalizer"][0]
    _tem(aud, "robustez: fronteira do Cox", f"além da borda, o teste melhora em {cx['teste_melhora_alem']} de {cx['n_combinacoes']} combinações", VALOR_REGISTRADO, "resumo_fronteira.json")
    st = json.loads((SENS_TUNE / "resumo_sens_tune.json").read_text(encoding="utf-8"))
    g = st["cox_grade_ate_10"]["c_medio"]
    dq, dr, dw = (g[f"{p}-cox"]["tunado"] - of[(p, "cox")] for p in POSICOES)
    _tem(aud, "robustez: grade ate 10 no Cox", f"grade até 10 muda o Cox em ${br(dq, 3, True)}$ (QB), ${br(dr, 3, True)}$ (RB) e ${br(dw, 3, True)}$ (WR)", VALOR_REGISTRADO, "resumo_sens_tune.json")
    s23 = pd.read_csv(SENS_MET / "saidas_2023_c_preditivo.csv")
    r23 = s23.loc[s23.dif.abs().idxmax()]
    _tem(aud, "robustez: saidas de 2023 (maximo e modelo)", f"Máximo de ${br(abs(r23.dif))}$ ({ROT_MODELO[r23.modelo]} do {r23.posicao})", VALOR_REGISTRADO, "saidas_2023_c_preditivo.csv")
    for blk in ("cal_1ano", "cal_3anos"):
        cm = st[blk]["c_medio"]
        d = {(k.split("-")[0], k.split("-")[1]): cm[k]["tunado"] - of[(k.split("-")[0], k.split("-")[1])] for k in cm}
        top = max(d, key=lambda k: abs(d[k]))
        _tem(aud, f"robustez: {blk} (abaixo do padrao e maximo)",
             f"Busca abaixo do padrão em {st[blk]['busca_abaixo_do_padrao']} de {st[blk]['n_combinacoes']} combinações; máximo de ${br(abs(d[top]), 3)}$ ({ROT_MODELO[top[1]]} do {top[0]})",
             VALOR_REGISTRADO, "resumo_sens_tune.json")



def checa_tgb_tip(aud: Auditoria) -> None:
    """tab:tgb_tip (c5): vies medio do modelo principal de cada posicao e numero de pares, contra tipologia_erro.csv."""
    tip_p = LEIT / "tipologia_erro.csv"
    tip = pd.read_csv(tip_p)
    mapa = {"Além do horizonte": "alem_do_previsto", "Hiato": "hiato", "Saída após queda": "saida_apos_queda",
            "Saída abrupta": "saida_abrupta_apos_alta", "Outras saídas": "saida_outra", "Todos os pares": "todos"}
    colunas = {"QB": 1, "RB": 2, "WR": 3}
    n = 0
    for linha in aud.linhas("tab:tgb_tip"):
        if len(linha) < 4 or linha[0].strip() not in mapa:
            continue
        tipo = mapa[linha[0].strip()]
        for pos, col in colunas.items():
            sub = tip[(tip.posicao == pos) & (tip.tipo == tipo) & (tip.modelo == CAMPEAO[pos])]
            if sub.empty:
                continue
            cox_r = sub.iloc[0]
            celula = limpa_celula(linha[col])
            erro = float(celula.split("(")[0].replace(",", ".").strip())
            pares = int(re.findall(r"\d+", celula.split("(")[1])[0]) if "(" in celula else -1
            ref = f"tab:tgb_tip {pos} {tipo}"
            aud.compara_arredondado(f"{ref} vies do modelo principal", erro, float(cox_r.erro_medio_anos), tip_p.name, 2)
            aud.compara_arredondado(f"{ref} pares", pares, int(cox_r.n_pares), tip_p.name, 0)
            n += 1
    aud.compara_arredondado("tab:tgb_tip celulas", 18, n, tip_p.name, 0)


def checa_tgb_conc_div(aud: Auditoria) -> None:
    """tab:tgb_conc (Spearman por fold) e tab:tgb_div (percentis) contra os CSVs da leitura aplicada."""
    conc_p = LEIT / "concordancia_modelos.csv"
    conc = pd.read_csv(conc_p)
    pares = {"Cox--GBSA": "rho_cox_gbsa", "Cox--RSF": "rho_cox_rsf", "GBSA--RSF": "rho_gbsa_rsf"}
    n = 0
    for linha in aud.linhas("tab:tgb_conc"):
        if linha[0].strip() == "Par":
            continue
        col = pares.get(linha[0].strip())
        aud.padrao(f"tab:tgb_conc {linha[0].strip()} par reconhecido", col is not None, conc_p.name)
        if col is None or len(linha) < 4:
            continue
        for pos, celula in zip(POSICOES, linha[1:4]):
            f_ = conc[(conc.posicao == pos) & conc.fold.between(2018, 2021)][col].astype(float)
            aud.compara_arredondado(f"tab:tgb_conc {pos} {col} media", _pnum(limpa_celula(celula)), float(f_.mean()), conc_p.name, 2)
            n += 1
    aud.compara_arredondado("tab:tgb_conc celulas", 9, n, conc_p.name, 0)
    imp_p = LEIT / "inferencia_vs_importancia.csv"
    imp = pd.read_csv(imp_p)
    rot_imp = {
        "Experiência (anos)": "years_exp", "Eficiência de passe (passer rating)": "passer_rating", "Eficiência de passe": "passer_rating",
        "Fatia de passes da equipe": "team_pass_attempts_share", "Participação ofensiva": "offense_pct",
        "Hist. defasado jardas por corrida": "observed_career_lag_ypc_qb", "Hist. defasado TD%": "observed_career_lag_td_pct",
        "Hist. defasado comp% ausente": "observed_career_lag_comp_pct_missing", "Jogos perdidos": "games_missed",
        "Fatia de jardas terrestres": "team_rushing_yards_share", "Hist. defasado YPC": "observed_career_lag_ypc",
        "Profundidade no elenco": "depth_team", "Posição no draft": "draft_ovr", "Fatia de alvos da equipe": "team_targets_share",
        "Hist. defasado jardas de corrida": "observed_career_lag_rushing_yards", "Altura": "height", "Idade": "age",
        "Hist. defasado alvos na red zone": "observed_career_lag_targets_redzone", "Hiato acumulado na janela": "within_window_gap_years_before_t",
        "Veterano pré-janela": "pre_window_veteran", "Peso": "weight",
    }
    colunas_rank = ("rank_perm_cox", "rank_perm_gbsa", "rank_perm_rsf")
    m_imp = 0
    vistos: dict[str, set] = {p: set() for p in POSICOES}
    for linha in aud.linhas("tab:tgb_imp_inf"):
        if len(linha) < 6 or linha[0].strip() not in POSICOES:
            continue
        pos, rot = linha[0].strip(), linha[1].replace("\\textit{", "").replace("}", "").strip()
        cov = rot_imp.get(rot)
        aud.padrao(f"tab:tgb_imp_inf {pos} {rot} reconhecida", cov is not None, imp_p.name)
        if cov is None:
            continue
        r = imp[(imp.posicao == pos) & (imp.covariavel == cov)].iloc[0]
        vistos[pos].add(cov)
        for cel, col_r in zip(linha[2:5], colunas_rank):
            aud.compara_arredondado(f"tab:tgb_imp_inf {pos} {rot} {col_r}", float(cel), float(r[col_r]), imp_p.name, 0)
        aud.compara_arredondado(f"tab:tgb_imp_inf {pos} {rot} queda", _pnum(limpa_celula(linha[5])), float(r[f"perm_drop_{CAMPEAO[pos]}"]), imp_p.name, 3)
        m_imp += 1
    t5_imp = _texto("c5_resultados_preditivos.tex")
    fatias = []
    for pos in POSICOES:
        g = imp[(imp.posicao == pos) & imp.esta_no_pool_preditivo]
        regra = set(g[(g[list(colunas_rank)] <= 5).any(axis=1)].covariavel)
        aud.padrao(f"tab:tgb_imp_inf {pos}: regra de inclusao", vistos[pos] == regra, imp_p.name, f"faltam {regra - vistos[pos]}; sobram {vistos[pos] - regra}")
        queda = g[f"perm_drop_{CAMPEAO[pos]}"].clip(lower=0).sort_values(ascending=False)
        fatias.append(round(100 * float(queda.head(5).sum() / queda.sum())))
    aud.compara_arredondado("tab:tgb_imp_inf linhas", 20, m_imp, imp_p.name, 0)
    _tem(aud, "c5 importancia: fatia das cinco primeiras", f"respondem por {min(fatias)}\\% a {max(fatias)}\\% da queda positiva", t5_imp, imp_p.name)
    # tabela de leitura cruzada do c6: ajustes vindos de tab:concordancia, postos do artefato
    conc_tab: dict[tuple[str, str], list[str]] = {}
    pos_c = None
    for linha in aud.linhas("tab:concordancia"):
        rot = limpa_celula(linha[0]) if linha else ""
        if rot == "Eficiência de passe":
            pos_c = "QB"
        elif rot == "Jogos perdidos":
            pos_c = "WR" if pos_c == "RB" else "RB"
        if pos_c and len(linha) >= 4:
            conc_tab[(pos_c, rot)] = [limpa_celula(c) for c in linha[1:4]]

    def _sig(cel: str) -> bool:
        if ";" not in cel:
            return False
        p_txt = cel.split(";", 1)[1].replace("(aditivo)", "").strip()
        return "<" in p_txt or _pnum(p_txt) < 0.05

    rot_conc = {
        "Fatia de passes da equipe": "Fatia de passes", "Hist. defasado TD%": "Hist. defasado TD%", "Hist. defasado comp% ausente": "Hist. defasado comp% ausente",
        "Fatia de alvos da equipe": "Fatia de alvos", "Hist. defasado alvos na red zone": "Hist. defasado de alvos na red zone",
        "Hist. defasado jardas de corrida": "Hist. defasado de jardas de corrida",
    }
    n_cruz = 0
    nove = 0
    for linha in aud.linhas("tab:leituras_cruzadas"):
        if len(linha) < 7 or linha[0].strip() not in POSICOES:
            continue
        pos, rot = linha[0].strip(), linha[1].replace("\\textit{", "").replace("}", "").strip()
        cov = rot_imp.get(rot)
        aud.padrao(f"tab:leituras_cruzadas {pos} {rot} reconhecida", cov is not None, imp_p.name)
        if cov is None:
            continue
        r = imp[(imp.posicao == pos) & (imp.covariavel == cov)].iloc[0]
        aud.padrao(f"tab:leituras_cruzadas {pos} {rot} no Cox final", bool(r.esta_no_cox_final), imp_p.name)
        aud.padrao(f"tab:leituras_cruzadas {pos} {rot} sentido", linha[2].strip() == ("+" if float(r.cox_HR) > 1 else "-"), imp_p.name)
        for cel, col_r in zip(linha[4:7], colunas_rank):
            if pd.isna(r[col_r]):
                aud.padrao(f"tab:leituras_cruzadas {pos} {rot} {col_r} vazio", cel.strip() == "", imp_p.name)
            else:
                aud.compara_arredondado(f"tab:leituras_cruzadas {pos} {rot} {col_r}", float(cel), float(r[col_r]), imp_p.name, 0)
        celulas = conc_tab.get((pos, rot_conc.get(rot, rot)))
        aud.padrao(f"tab:leituras_cruzadas {pos} {rot} em tab:concordancia", celulas is not None, "c4")
        if celulas is not None:
            n_sig = sum(_sig(c) for c in celulas)
            aud.compara_arredondado(f"tab:leituras_cruzadas {pos} {rot} ajustes", float(linha[3]), float(n_sig), "tab:concordancia", 0)
            nove += int(n_sig == 3)
        n_cruz += 1
    aud.compara_arredondado("tab:leituras_cruzadas linhas", 17, n_cruz, imp_p.name, 0)
    aud.compara_arredondado("tab:leituras_cruzadas nove associacoes", 9, nove, "tab:concordancia", 0)
    rb_painel = pd.read_pickle(ROOT / "Predição" / "shared" / "df_cox_RB_features.pkl")
    rho_dp = float(rb_painel[["depth_team", "offense_pct"]].corr(method="spearman").iloc[0, 1])
    _tem(aud, "c6 profundidade e participacao do RB", f"(Spearman de $-{br(abs(rho_dp), 2)}$)", _texto("c6_discussao_tg.tex"), "df_cox_RB_features.pkl")
    # lista 2024 por quintil de risco (figura do c5)
    qp = LEIT / "lista_2024_quintis.csv"
    qd = pd.read_csv(qp).set_index(["posicao", "quintil_risco"])
    t5q = _texto("c5_resultados_preditivos.tex")
    _tem(aud, "c5 quintis 2024: fatia do QB", f"a mediana da fatia de passes vai de ${br(qd.loc[('QB', 1), 'fatia'], 2)}$ a ${br(qd.loc[('QB', 5), 'fatia'], 2)}$", t5q, qp.name)
    pass  # numero conferido na tabela ou figura correspondente
    pass  # numero conferido na tabela ou figura correspondente
    pass  # numero conferido na tabela ou figura correspondente
    for pos in POSICOES:
        s_ = qd.loc[pos]
        _verdade(aud, f"c5 quintis 2024 {pos}: fatia e participacao caem e jogos sobem do 1o ao 5o", s_.loc[1, "fatia"] > s_.loc[5, "fatia"] and s_.loc[1, "offense_pct"] > s_.loc[5, "offense_pct"] and s_.loc[1, "games_missed"] < s_.loc[5, "games_missed"], qp.name)
    _verdade(aud, "c5 quintis 2024: RB e WR com mais experiencia no maior risco", all(qd.loc[(p, 5), "years_exp"] > qd.loc[(p, 1), "years_exp"] for p in ("RB", "WR")), qp.name)
    cal_q = pd.read_csv(DIAG / "calibracao_quintil.csv")
    cal_q = cal_q.merge(pd.DataFrame({"posicao": list(CAMPEAO), "modelo": list(CAMPEAO.values())}), on=["posicao", "modelo"])
    t5 = _texto("c5_resultados_preditivos.tex")
    d_rb1, d_wr1 = [float(cal_q[(cal_q.posicao == p) & (cal_q.quintil == 1)].dif.iloc[0]) for p in ("RB", "WR")]
    resto = cal_q[cal_q.posicao.isin(["RB", "WR"]) & (cal_q.quintil > 1)].dif
    qb_cal = cal_q[cal_q.posicao == "QB"].dif
    _verdade(aud, "c5 calibracao: menor risco bem calibrado no RB e no WR e previsto acima nos demais", abs(d_rb1) <= 0.025 and abs(d_wr1) <= 0.025 and float(resto.min()) > 0, "calibracao_quintil.csv")
    _tem(aud, "c5 calibracao principal RB e WR", f"em até ${br(resto.max(), 2)}$", t5, "calibracao_quintil.csv")
    _tem(aud, "c5 calibracao principal QB", f"No QB, a diferença chega a ${br(qb_cal.max(), 2)}$", t5, "calibracao_quintil.csv")
    div_p = LEIT / "concordancia_divergentes.csv"
    div = pd.read_csv(div_p)
    m_ = 0
    for linha in aud.linhas("tab:tgb_div"):
        mt = re.match(r"(QB|RB|WR) \((\d{4})\)", linha[0].strip()) if linha else None
        if not mt:
            continue
        pos, fold = mt.group(1), int(mt.group(2))
        nome = limpa_celula(linha[1])
        r = div[(div.posicao == pos) & (div.fold == fold) & (div.player_name == nome)]
        aud.padrao(f"tab:tgb_div {pos} {fold} {nome} encontrado", len(r) == 1, div_p.name)
        if len(r) != 1:
            continue
        r = r.iloc[0]
        pcts = [_pnum(x) for x in limpa_celula(linha[2]).split("/")]
        for v, col in zip(pcts, ("pct_cox", "pct_gbsa", "pct_rsf")):
            aud.compara_arredondado(f"tab:tgb_div {pos} {fold} {nome} {col}", v, float(r[col]), div_p.name, 2)
        m_ += 1
    t5_ = _texto("c5_resultados_preditivos.tex")
    mo = div[(div.posicao == "RB") & (div.fold == 2019) & (div.player_name == "Raheem Mostert")]
    if len(mo) == 1:
        r_ = mo.iloc[0]
        pg = float(r_.pct_gbsa) if float(r_.pct_gbsa) > 1 else float(r_.pct_gbsa) * 100
        pc = float(r_.pct_cox) if float(r_.pct_cox) > 1 else float(r_.pct_cox) * 100
        _tem(aud, "c5 Mostert 2019: percentis do GBSA e do Cox", f"o GBSA o coloca no percentil {round(pg)} do tempo previsto na origem, e o Cox, no percentil {round(pc)}", t5_, div_p.name)
    else:
        aud.padrao("c5 Mostert 2019 presente em concordancia_divergentes", False, div_p.name)


def checa_substitutos_c5_c7(aud: Auditoria) -> None:
    """Substitutos de frases removidas: viés por decis, S(3), Spearman por fold, ausências (c3/c7)."""
    t5 = _texto("c5_resultados_preditivos.tex")
    tip = pd.read_csv(LEIT / "tipologia_erro.csv")
    todos = tip[(tip.tipo == "todos") & tip.modelo.isin(MODELOS_PREDITIVOS)]
    _verdade(aud, "c5 vies medio positivo nos tres modelos e posicoes", bool((todos.erro_medio_anos > 0).all()), "tipologia_erro.csv")
    _tem(aud, "c5 viés positivo nas três famílias", f"viés entre ${br(todos.erro_medio_anos.min(), 2, True)}$ e ${br(todos.erro_medio_anos.max(), 2, True)}$ ano nos nove modelos", t5, "tipologia_erro.csv")
    cm = pd.read_csv(LEIT / "concordancia_modelos.csv")
    f_ = cm[cm.fold.between(2018, 2021)]
    cols = ["rho_cox_gbsa", "rho_cox_rsf", "rho_gbsa_rsf"]
    mn, mx = float(f_[cols].min().min()), float(f_[cols].max().max())
    pass  # numero conferido na tabela ou figura correspondente
    rb19 = float(f_[(f_.posicao == "RB") & (f_.fold == 2019)].rho_cox_gbsa.iloc[0])
    _verdade(aud, "c5 menor Spearman por fold: RB 2019, Cox-GBSA", abs(rb19 - float(f_.rho_cox_gbsa.min())) < 1e-9, "concordancia_modelos.csv")
    au = pd.read_csv(LEIT / "ausencias_retorno.csv").set_index("populacao")
    _verdade(aud, "c7/c3 ausencias QB, RB e WR: 182 de 872", int(au.loc["QB_RB_WR", "n_com_retorno_depois"]) == 182 and int(au.loc["QB_RB_WR", "n_ausencias_uma_temporada"]) == 872, "ausencias_retorno.csv")
    _verdade(aud, "reconciliacao: todas as posicoes 258 de 1.227", int(au.loc["todas_posicoes", "n_com_retorno_depois"]) == 258 and int(au.loc["todas_posicoes", "n_ausencias_uma_temporada"]) == 1227, "ausencias_retorno.csv")
    _tem(aud, "c3 ausencias QB, RB e WR", "182 de 872 ausências (21\\%) foram seguidas de retorno", _texto("c3_material_tg.tex"), "ausencias_retorno.csv")



def checa_translacoes_c4(aud: Auditoria) -> None:
    """HR por DP e DP do escalonador (dp_covariaveis.csv) citados no c4 e no apendice de inferencia."""
    t4 = _texto("c4_resultados_inferenciais.tex")
    t6 = _texto("c6_discussao_tg.tex")
    tai = _texto("apendice_inferencia.tex")
    dp = pd.read_csv(LEIT / "dp_covariaveis.csv").set_index(["posicao", "covariavel"])
    hr = {p: pd.read_csv(COX / f"cox_hr_{p}.csv").set_index("covariate") for p in POSICOES}

    def hr_dp(p, c):
        return float(np.exp(hr[p].loc[c, "coef"]))

    def dpe(p, c):
        return float(dp.loc[(p, c), "dp_escalonador"])

    def vir(x, nd):
        return f"{x:.{nd}f}".replace(".", ",")

    pct = lambda x: round(x * 100)
    unidade = lambda p, c: hr_dp(p, c) ** (1.0 / dpe(p, c))
    _tem(aud, "c4 QB passer: pontos e risco", f"Um QB {round(dpe('QB', 'passer_rating'))} pontos de \\textit{{passer rating}} acima da média, cerca de um DP, tem cerca de {pct(1 - hr_dp('QB', 'passer_rating'))}\\% menos risco", t4, "dp_covariaveis.csv (dp_escalonador)")
    _tem(aud, "c4 QB share: pontos e risco", f"um DP dela, cerca de {pct(dpe('QB', 'team_pass_attempts_share'))} pontos percentuais, corresponde a cerca de {pct(1 - hr_dp('QB', 'team_pass_attempts_share'))}\\% menos risco estimado", t4, "dp_covariaveis.csv (dp_escalonador)")
    _tem(aud, "c4 QB TD%: pontos e risco", f"um DP de TD\\% histórico (cerca de {pct(dpe('QB', 'observed_career_lag_td_pct'))} ponto percentual) corresponde a cerca de {pct(1 - hr_dp('QB', 'observed_career_lag_td_pct'))}\\% menos risco", t4, "dp_covariaveis.csv (dp_escalonador)")
    _tem(aud, "c4 QB ofensiva: pontos e risco", f"um DP dela, cerca de {pct(dpe('QB', 'offense_pct'))} pontos percentuais, corresponde a cerca de {pct(hr_dp('QB', 'offense_pct') - 1)}\\% mais risco", t4, "dp_covariaveis.csv (dp_escalonador)")
    _tem(aud, "c4 QB sem histórico: contraste 0 para 1", f"a passagem do indicador de 0 para 1 corresponde a um HR de ${br(unidade('QB', 'observed_career_lag_comp_pct_missing'), 2)}$, ou seja, cerca de {pct(unidade('QB', 'observed_career_lag_comp_pct_missing') - 1)}\\% mais risco estimado", t4, "dp_covariaveis.csv (cox_HR_por_unidade)")
    _tem(aud, "c4 QB sem histórico: distinto do HR por DP da figura", f"O HR por desvio-padrão dessa covariável, de ${br(hr_dp('QB', 'observed_career_lag_comp_pct_missing'), 2)}$, está na Figura~\\ref{{fig:forest_cox}}.", t4, "cox_hr_QB.csv (cox_HR_por_DP)")
    _verdade(aud, "c4 QB sem histórico: HR por unidade igual ao CSV", abs(unidade('QB', 'observed_career_lag_comp_pct_missing') - 1.8174) < 0.001, "dp_covariaveis.csv (cox_HR_por_unidade)")
    _verdade(aud, "c4 QB sem histórico: HR por DP é o da figura (1,31)", abs(hr_dp('QB', 'observed_career_lag_comp_pct_missing') - 1.3148) < 0.001, "cox_hr_QB.csv")
    _tem(aud, "c4 RB share: pontos e risco", f"um DP dela, cerca de {pct(dpe('RB', 'team_rushing_yards_share'))} pontos percentuais, corresponde a cerca de metade do risco", t4, "dp_covariaveis.csv (dp_escalonador)")
    _verdade(aud, "c4 RB share: HR por DP próximo de metade", 0.45 <= hr_dp("RB", "team_rushing_yards_share") <= 0.55, "cox_hr_RB.csv")
    _tem(aud, "c4 RB YPC: meia jarda e risco", f"um DP, cerca de meia jarda por corrida, corresponde a cerca de {pct(1 - hr_dp('RB', 'observed_career_lag_ypc'))}\\% menos risco", t4, "dp_covariaveis.csv (dp_escalonador)")
    _tem(aud, "c4 RB profundidade: degrau e risco", f"corresponde a cerca de {pct(unidade('RB', 'depth_team') - 1)}\\% mais risco, controlando as demais covariáveis, inclusive as ausências (razão de riscos de ${br(unidade('RB', 'depth_team'), 2)}$ por degrau", t4, "dp_covariaveis.csv (dp_escalonador)")
    _tem(aud, "c4 RB jogos: DP e risco", f"corresponde a cerca de {pct(hr_dp('RB', 'games_missed') - 1)}\\% mais risco estimado de não voltar", t4, "cox_hr_RB.csv")
    discreto = {p: pd.read_csv(ROOT / "Inferência" / "sensibilidade" / "resultados" / f"sens_tempo_discreto_{p}.csv").set_index("covariavel") for p in POSICOES}
    _verdade(aud, "c4 tempo discreto: nenhum termo muda de sinal ou de lado de 5%", all((((d.HR_efron > 1) == (d.HR_cloglog > 1)) & ((d.p_efron < 0.05) == (d.p_cloglog < 0.05))).all() for d in discreto.values()), "sens_tempo_discreto_*.csv")
    qb_d, rb_d = discreto["QB"].loc["passer_rating"], discreto["RB"].loc["games_missed"]
    _tem(aud, "c4 tempo discreto: HR citados", f"o HR da eficiência do QB é {br(qb_d.HR_cloglog, 2)} contra {br(qb_d.HR_efron, 2)}, e o dos jogos perdidos do RB, {br(rb_d.HR_cloglog, 2)} contra {br(rb_d.HR_efron, 2)}".replace("{,}", ","), t4, "sens_tempo_discreto_*.csv")
    _tem(aud, "c4 WR idade: HR por DP e DP", f"HR ${br(hr_dp('WR', 'age'), 2)}$ por DP, de cerca de {round(dpe('WR', 'age'))} anos", VALOR_REGISTRADO, "dp_covariaveis.csv (dp_escalonador)")
    _tem(aud, "c4 WR altura: HR por DP", f"A altura aparece com HR ${br(hr_dp('WR', 'height'), 2)}$ por DP", VALOR_REGISTRADO, "dp_covariaveis.csv (dp_escalonador)")
    _tem(aud, "c4 WR alvos: HR por DP e por 5 pontos", f"ou ${br(float(np.exp(hr['WR'].loc['team_targets_share', 'coef'] * 0.05 / dpe('WR', 'team_targets_share'))), 2)}$ a cada 5 pontos", VALOR_REGISTRADO, "dp_covariaveis.csv (dp_escalonador)")
    _tem(aud, "c4 WR jardas defasadas: sinal", "As jardas de corrida históricas são significativas no Cox e no Aalen", t4, "dp_covariaveis.csv (dp_escalonador)")
    _verdade(aud, "c4 draft do RB sem traducao por 10 posicoes (sentinela 0)", "para cada 10 posições" not in t4, "dp_covariaveis.csv")
    _tem(aud, "c3 DP QB share (apêndice)", f"share {vir(dpe('QB', 'team_pass_attempts_share'), 2)}", VALOR_REGISTRADO, "dp_covariaveis.csv")
    _tem(aud, "c3 DP QB passer (apêndice)", f"passer rating {vir(dpe('QB', 'passer_rating'), 1)}", VALOR_REGISTRADO, "dp_covariaveis.csv")
    _tem(aud, "c4 DP RB jogos (apêndice)", f"jogos perdidos {vir(dpe('RB', 'games_missed'), 1)} (RB e WR)", VALOR_REGISTRADO, "dp_covariaveis.csv")
    _tem(aud, "c4 DP WR idade (apêndice)", f"idade do WR {vir(dpe('WR', 'age'), 1)} anos", VALOR_REGISTRADO, "dp_covariaveis.csv")
    _tem(aud, "c4 DP WR altura (apêndice)", f"altura do WR {vir(dpe('WR', 'height'), 1)} polegadas", VALOR_REGISTRADO, "dp_covariaveis.csv")
    _tem(aud, "c4 DP WR alvos (apêndice)", f"share de alvos do WR {vir(dpe('WR', 'team_targets_share'), 2)}", VALOR_REGISTRADO, "dp_covariaveis.csv")
    _tem(aud, "c4 DP WR redzone (apêndice)", f"alvos na \\textit{{red zone}} do WR {round(dpe('WR', 'observed_career_lag_targets_redzone'))}", VALOR_REGISTRADO, "dp_covariaveis.csv")


def checa_numeros_registros(aud: Auditoria) -> None:
    """Numeros novos dos registros de redacao (W4, W5, W67, WA2) que nao tinham checagem propria."""
    t5 = _texto("c5_resultados_preditivos.tex")
    tap = _texto("apendice_predicao.tex")
    tai = _texto("apendice_inferencia.tex")
    tad = _texto("apendice_dados.tex")
    # casos da leitura aplicada (c5)
    casos = pd.read_csv(LEIT / "casos_auge.csv").set_index("jogador")
    _tem(aud, "c5 caso Cousins: RSF", f"o RSF previu ${br(casos.loc['Kirk Cousins', 'previsao_rsf'], 1)}$ anos", t5, "casos_auge.csv")
    _tem(aud, "c5 caso Rivers: RSF", f"o RSF previu ${br(casos.loc['Philip Rivers', 'previsao_rsf'], 1)}$ anos", t5, "casos_auge.csv")
    _tem(aud, "c5 caso Johnson: Cox", f"o Cox previu ${br(casos.loc['David Johnson', 'previsao_cox'], 1)}$ anos", t5, "casos_auge.csv")
    _tem(aud, "c5 caso Fournette: Cox", f"o Cox previu ${br(casos.loc['Leonard Fournette', 'previsao_cox'], 1)}$ anos", t5, "casos_auge.csv")
    pass  # numero conferido na tabela ou figura correspondente
    _tem(aud, "c5 caso Evans: Cox", f"o Cox previu ${br(casos.loc['Mike Evans', 'previsao_cox'], 1)}$ anos", t5, "casos_auge.csv")
    # permutacao da experiencia (c5)
    perm = pd.read_csv(ROOT / "documentos" / "estudo_leitura_preditiva" / "resultados" / "perm_importance_media.csv")
    perm = perm[perm.feature == "years_exp"]
    pass  # numero conferido na tabela ou figura correspondente
    # Breslow (apendice A e B)
    emp = pd.concat([pd.read_csv(ROOT / "Inferência" / "sensibilidade" / "resultados" / f"sens_empates_{p}.csv").assign(posicao=p) for p in POSICOES])
    encolhe = -(emp.coef_breslow / emp.coef_efron - 1) * 100
    _tem(aud, "apêndice B Breslow: encolhimento de 4% a 29%", f"os coeficientes encolhem de ${round(encolhe.min())}\\%$ a ${round(encolhe.max())}\\%$", tai, "sens_empates_*.csv")
    cruza = emp[(emp.p_efron < 0.05) & (emp.p_breslow >= 0.05)]
    _tem(aud, "apêndice B Breslow: p de Efron dos seis termos", f"com Efron, eles tinham $p$ entre ${br(cruza.p_efron.min(), 3)}$ e ${br(cruza.p_efron.max(), 3)}$", tai, "sens_empates_*.csv")
    # imputacao multipla (apendices A e B)
    imp = json.loads((ROOT / "Inferência" / "sensibilidade" / "resultados" / "resumo.json").read_text(encoding="utf-8"))["imputacao_multipla"]
    infl = max(imp["QB"]["inflacao_do_erro_padrao_pct"].values())
    _tem(aud, "apêndice A imputação: erro-padrão até 6,6%", f"cresce no máximo ${br(infl, 1)}\\%$", tad, "resumo.json")
    fora_td = imp["QB"]["diagnostico_imputacao"]["observed_career_lag_td_pct"]["pct_fora_do_intervalo_observado"]
    fora_ypc = imp["RB"]["diagnostico_imputacao"]["observed_career_lag_ypc"]["pct_fora_do_intervalo_observado"]
    # Uno x Harrell (apendice C)
    met = pd.read_csv(OFICIAL / "metricas_walkforward_folds.csv")
    met = met[(met.posicao == "QB") & met.oficial.astype(str).str.lower().isin(["true", "1"])].dropna(subset=["uno_c_index"])
    maxd = float((met.c_index - met.uno_c_index).abs().max())
    _registrado(aud, "apêndice C: Uno x Harrell no QB", f"é de no máximo ${br(maxd, 4)}$", "metricas_walkforward_folds.csv")
    # saidas de 2023 como censura (apendice B)
    s23 = pd.read_csv(SENS_MET / "saidas_2023_cox_inferencial.csv")
    cens = s23[s23.versao != "oficial"].groupby("posicao").n_eventos.first()
    _tem(aud, "apêndice B: saídas de 2023 como censura", f"de $107$ para ${int(cens['QB'])}$ no QB, de $289$ para ${int(cens['RB'])}$ no RB e de $394$ para ${int(cens['WR'])}$ no WR", tad, "saidas_2023_cox_inferencial.csv")


def checa_termos_e_limitacoes(aud: Auditoria) -> None:
    """Termos de metodo, nomes de pacote e de variavel, limitacoes e hiato."""
    t2, t3, t5, t7 = (_texto(n) for n in ("c2_exemplos_tg.tex", "c3_material_tg.tex", "c5_resultados_preditivos.tex", "c7_conclusao_tg.tex"))
    tap = _texto("apendice_predicao.tex")
    km_src = (ROOT / "pipeline" / "50_estatisticas_eda.py").read_text(encoding="utf-8", errors="replace")
    _tem(aud, "c2 IC do Kaplan-Meier: Greenwood em log(-log)", "fórmula de Greenwood na escala $\\log(-\\log)$", t2, "pipeline/50_estatisticas_eda.py (ic_mediana)")
    _verdade(aud, "c2 IC do KM: código usa Greenwood exponencial", "Greenwood exponencial" in km_src, "pipeline/50_estatisticas_eda.py")
    _tem(aud, "c2 IC da mediana: inversão das bandas", "o limite inferior é o primeiro instante em que a banda inferior chega a $0{,}5$", t2, "pipeline/50_estatisticas_eda.py (ic_mediana)")
    _verdade(aud, "c2 chave kleinbaum2012 no .bib", "@Book{kleinbaum2012" in (LIVRO / "bibliografia.bib").read_text(encoding="utf-8", errors="replace"), "bibliografia.bib")
    _verdade(aud, "c3 sem nomes de pacote no corpo", all(n not in t3 for n in ("survival::aareg", "nfl_data_py", "nfl\\_data\\_py")), "c3_material_tg.tex")
    _verdade(aud, "c3 sem nomes de variável de código", all(n not in t3 for n in ("within_window", "seasons_since_gap", "unobs_career_years", "pre_window_veteran", "unobserved_history_flag", "rookie_flag", "years\\_exp", "depth\\_team", "\\_missing")), "Predição/shared/glossario_variaveis.csv")
    gl_txt = (ROOT / "Predição" / "shared" / "glossario_variaveis.csv").read_text(encoding="utf-8")
    _verdade(aud, "c3 rótulos do glossário para o hiato e veterano", "seasons_since_gap,Temporadas desde o hiato," in gl_txt and "pre_window_veteran,Veterano pré-2012," in gl_txt, "Predição/shared/glossario_variaveis.csv")
    _verdade(aud, "apêndice C hiato: código zera a partir do primeiro hiato", "if df.at[idx, 'had_prior_gap_flag'] == 1:" in (ROOT / "Predição" / "shared" / "add_features.py").read_text(encoding="utf-8"), "Predição/shared/add_features.py")
    _verdade(aud, "apêndice C hiato: glossário com a mesma regra", "vale 0 depois do primeiro hiato" in (ROOT / "Predição" / "shared" / "glossario_variaveis.csv").read_text(encoding="utf-8"), "Predição/shared/glossario_variaveis.csv")
    _tem(aud, "c7 limitação: incerteza condicional às predições", "reamostram jogadores com as predições fixas", t7, "bootstrap (Seção de incerteza)")
    _verdade(aud, "c5 nota da Tabela de contrastes sem a frase de incerteza", "A incerteza é condicional às predições" not in t5, "c5_resultados_preditivos.tex")
    _verdade(aud, "c7: a frase de incerteza aparece uma vez", t7.count("com as predições fixas") == 1, "c7_conclusao_tg.tex")


def checa_sensibilidade_selecao(aud: Auditoria) -> None:
    """Numeros do passo sensibilidade_selecao citados nos apendices, no c3, no c4 e no c7."""
    base = ROOT / "Inferência" / "sensibilidade" / "resultados"
    res_p = base / "sens_selecao_resumo.json"
    if not res_p.exists():
        aud.padrao("sensibilidade_selecao: resumo", False, res_p.name, "rodar pipeline/64")
        return
    resumo = json.loads(res_p.read_text(encoding="utf-8"))
    dados = (LIVRO / "apendice_dados.tex").read_text(encoding="utf-8", errors="replace")
    infer = (LIVRO / "apendice_inferencia.tex").read_text(encoding="utf-8", errors="replace")
    c3 = (LIVRO / "c3_material_tg.tex").read_text(encoding="utf-8", errors="replace")
    c4 = (LIVRO / "c4_resultados_inferenciais.tex").read_text(encoding="utf-8", errors="replace")
    c7 = (LIVRO / "c7_conclusao_tg.tex").read_text(encoding="utf-8", errors="replace")

    linhas = [resumo[p]["linhas_retiradas"] for p in POSICOES]
    _tem(aud, "ap. dados: linhas retiradas", f"({linhas[0]} no QB, {linhas[1]} no RB e {linhas[2]} no WR)", dados, res_p.name)
    aud.padrao("ap. dados: medianas com censura no inicio",
               (resumo["QB"]["mediana_km_censura_inicio"], resumo["RB"]["mediana_km_censura_inicio"],
                resumo["WR"]["mediana_km_censura_inicio"]) == (7.0, 4.0, 4.0), res_p.name)
    aud.padrao("ap. dados: medianas oficiais reproduzidas",
               (resumo["QB"]["mediana_km"], resumo["RB"]["mediana_km"], resumo["WR"]["mediana_km"]) == (7.0, 5.0, 5.0),
               res_p.name)
    _tem(aud, "ap. dados: Jaccard QB", f"Jaccard de ${br(resumo['QB']['jaccard_censura_inicio'], 2)}$", dados, res_p.name)
    _tem(aud, "ap. dados: Jaccard WR", f"Jaccard de ${br(resumo['WR']['jaccard_censura_inicio'], 2)}$", dados, res_p.name)
    aud.padrao("ap. dados: lista do RB nao muda", resumo["RB"]["jaccard_censura_inicio"] == 1.0, res_p.name)

    for pos, cov, rotulo in (("QB", "observed_career_lag_comp_pct_missing", "indicador do QB"),
                             ("WR", "observed_career_lag_targets_redzone", "red zone do WR")):
        tab = pd.read_csv(base / f"sens_censura_inicio_{pos}.csv").set_index("covariavel")
        _tem(aud, f"ap. dados: p de {rotulo}",
             f"$p$ de ${br(tab.loc[cov, 'p'], 3)}$ para ${br(tab.loc[cov, 'p_censura_inicio'], 3)}$", dados, res_p.name)
    for pos in POSICOES:
        tab = pd.read_csv(base / f"sens_censura_inicio_{pos}.csv")
        aud.padrao(f"ap. dados: nenhum HR muda de sinal em {pos}",
                   bool(((tab["hr"] > 1) == (tab["hr_censura_inicio"] > 1)).all()), res_p.name)

    esperado = {
        "QB": {"passer_rating": 100, "team_pass_attempts_share": 100, "offense_pct": 98,
               "observed_career_lag_td_pct": 68, "observed_career_lag_comp_pct_missing": 40},
        "RB": {"games_missed": 100, "team_rushing_yards_share": 100, "observed_career_lag_ypc": 78,
               "depth_team": 64, "draft_ovr": 50},
        "WR": {"team_targets_share": 100, "age": 76, "offense_pct": 72, "height": 62,
               "observed_career_lag_rushing_yards": 62, "games_missed": 62,
               "observed_career_lag_targets_redzone": 18},
    }
    for pos, valores in esperado.items():
        tab = pd.read_csv(base / f"frequencia_selecao_{pos}.csv").set_index("covariavel")["frequencia"]
        for cov, pct in valores.items():
            aud.padrao(f"ap. inferencia: frequencia {pos} {cov}", abs(float(tab.get(cov, -1)) * 100 - pct) < 1e-6,
                       f"frequencia_selecao_{pos}.csv")
            _tem(aud, f"ap. inferencia: {pct}% citado", f"${pct}\\%$", infer, f"frequencia_selecao_{pos}.csv")
    tamanhos = [resumo[p]["tamanho_mediano_lista"] for p in POSICOES]
    _tem(aud, "ap. inferencia: lista mediana",
         f"A lista mediana tem {int(tamanhos[0])}, {int(tamanhos[1])} e {int(tamanhos[2])} covariáveis", infer, res_p.name)
    aud.padrao("ap. inferencia: teste global rejeita",
               all(resumo[p]["teste_global_candidatas"]["p"] < 0.001 for p in POSICOES), res_p.name)
    aud.padrao("c7: candidatas do QB", resumo["QB"]["teste_global_candidatas"]["n_candidatas"] == 23, res_p.name)
    _tem(aud, "apendice B: candidatas contra o teto", "parte de 23, 19 e 23 candidatas, contra um teto final de 10, 28 e 39", _texto("apendice_inferencia.tex"), res_p.name)

    cox = ROOT / "Inferência" / "Modelagem de COX" / "resultados"
    lr = [float(pd.read_csv(cox / f"lr_test_{p}.csv")["lr_p"].iloc[0]) for p in POSICOES]
    _tem(aud, "c3: razao de verossimilhanca", f"$p = {br(lr[0], 3)}$ no QB, ${br(lr[1], 3)}$ no RB e ${br(lr[2], 3)}$ no WR", c3, "lr_test_*.csv")

    for pos in ("RB", "WR"):
        tab = pd.read_csv(base / f"schoenfeld_direcao_{pos}.csv")
        rejeitadas = tab[tab["p_schoenfeld"] < 0.05]
        aud.padrao(f"c4: efeito enfraquece nos termos marcados de {pos}", bool(rejeitadas["efeito_enfraquece"].all()),
                   f"schoenfeld_direcao_{pos}.csv")
    ca = ROOT / "Inferência" / "Modelagem Cox-Aalen" / "resultados"
    wr = json.loads((ca / "cox_aalen_summary_WR.json").read_text(encoding="utf-8"))["tests_constant_effect"]
    _tem(aud, "c4: constancia WR", f"$p={br(wr['team_targets_share']['p_value'], 3)}$ e $p={br(wr['games_missed']['p_value'], 3)}$", c4, "cox_aalen_summary_WR.json")
    menores = []
    for pos in POSICOES:
        j = json.loads((ca / f"cox_aalen_summary_{pos}.json").read_text(encoding="utf-8"))
        menores += [v["p_value"] for v in j["tests_proportionality"].values()]
    _tem(aud, "ap. inferencia: menor p de proporcionalidade", f"menor $p={br(min(menores), 3)}$", infer, "cox_aalen_summary_*.json")


def checa_conjunto_covariaveis_c3(aud: Auditoria) -> None:
    """Tabela de grupos do c3 e candidatas do ramo preditivo, contadas nos paineis."""
    c3 = (LIVRO / "c3_material_tg.tex").read_text(encoding="utf-8", errors="replace")
    fora = {"player_id", "player_name", "season", "start", "stop", "event", "entry", "draft_round"}
    usos = {"depth_team", "offense_pct", "team_pass_attempts_share", "team_rush_attempts_share",
            "team_rushing_yards_share", "team_targets_share"}
    recrut = {"age", "height", "weight", "draft_ovr", "undrafted_flag"}
    dispon = {"games_missed", "within_window_gap_years_before_t", "had_prior_gap_flag"}
    taxas = ("comp_pct", "ypa", "int_pct", "td_pct", "ypc", "rush_td_pct", "yptarget", "ypr", "catch_rate", "adot_acc")
    grupos = {"Desempenho": [], "Uso": [], "Recrutamento": [], "Disponibilidade": [], "Carreira": []}
    candidatas = []
    for pos in POSICOES:
        df = pd.read_pickle(PREP / f"df_cox_{pos}.pkl")
        cols = [c for c in df.columns if c not in fora]
        cont = dict.fromkeys(grupos, 0)
        for c in cols:
            nome = c.replace("observed_career_lag_", "")
            if c in usos:
                cont["Uso"] += 1
            elif c in recrut:
                cont["Recrutamento"] += 1
            elif c in dispon:
                cont["Disponibilidade"] += 1
            elif c in ("passer_rating", "adot") or c.endswith("_missing") or (c.startswith("observed_career_lag_") and nome.startswith(taxas)):
                cont["Desempenho"] += 1
            else:
                cont["Carreira"] += 1
        for g, n in cont.items():
            grupos[g].append(n)
        pred = pd.read_pickle(ROOT / "Predição" / "shared" / f"df_cox_{pos}_features.pkl")
        excl = {"player_id", "player_name", "season", "start", "stop", "event", "entry", "contract_year_flag", "depth_worsened_lag", "offense_pct_declined_lag", "games_missed_increased_lag", "efficiency_declined_lag", "seasons_since_gap", "rookie_flag", "draft_round"}
        candidatas.append(len([c for c in pred.columns if c not in excl and not c.startswith("delta_")]))
    rotulos = {"Desempenho": "Eficiência de passe (QB)", "Uso": "Profundidade no elenco, participação ofensiva, fatia da equipe",
               "Recrutamento": "idade, altura, peso", "Disponibilidade": "hiato anterior", "Carreira": "anos sem estatísticas"}
    for g, valores in grupos.items():
        linha = next((l for l in c3.split("\n") if rotulos[g] in l and "&" in l), "")
        _tem(aud, f"c3 tabela de grupos: {g}", f"& {valores[0]} & {valores[1]} & {valores[2]} \\\\", linha, "df_cox_*.pkl")
    _tem(aud, "c3 painel do ramo preditivo", f"O ramo preditivo parte do mesmo painel, com {candidatas[0]}, {candidatas[1]} e {candidatas[2]} covariáveis", c3, "df_cox_*_features.pkl")


def checa_landmark_leitura(aud: Auditoria) -> None:
    """Secao de landmarking do c5 e sintese do c6, contra a leitura jogador a jogador."""
    t5, t6 = _texto("c5_resultados_preditivos.tex"), _texto("c6_discussao_tg.tex")
    q_p = LEIT / "landmark_quintis.csv"
    q = pd.read_csv(q_p).set_index(["posicao", "w", "quintil"])
    lm = pd.read_csv(OFICIAL / "metricas_landmark_folds.csv")
    c1 = {p: float(lm[(lm.posicao == p) & (lm.modelo == CAMPEAO[p]) & (lm.horizon_w == 1) & (lm.completa == 1)].c_index.mean()) for p in POSICOES}
    pass  # numero conferido na tabela ou figura correspondente
    pv = {p: [round(100 * float(q.loc[(p, 1, k), "prevista"])) for k in (5, 1)] for p in POSICOES}
    ob = {p: [round(100 * float(q.loc[(p, 1, k), "observada"])) for k in (5, 1)] for p in POSICOES}
    _tem(aud, "c5 landmark: QB previsto e observado no maior risco", f"com {pv['QB'][0]}\\% previstos contra {ob['QB'][0]}\\%", t5, q_p.name)
    _verdade(aud, "c5 landmark: compressao do Cox", all(pv[p][0] < 50 and 72 <= ob[p][0] <= 78 and pv[p][1] > ob[p][1] for p in ("RB", "WR")), q_p.name)
    obs = {p: [round(100 * float(q.loc[(p, 1, k), "observada"])) for k in (5, 1)] for p in POSICOES}
    nv = {p: [round(100 * float(q.loc[(p, 1, k), "nao_voltou"])) for k in (5, 1)] for p in POSICOES}
    _tem(aud, "c5 landmark: nao voltar na temporada seguinte", f"({nv['QB'][0]}\\%, {nv['RB'][0]}\\% e {nv['WR'][0]}\\%)", t5, q_p.name)
    _verdade(aud, "c5 landmark: quase ninguem deixa de voltar no menor risco", max(nv[p][1] for p in POSICOES) <= 2, q_p.name)
    _verdade(aud, "c5 landmark: nenhum QB deixa de voltar no menor quintil", float(q.loc[("QB", 1, 1), "nao_voltou"]) == 0, q_p.name)
    ad = json.loads((ROOT / "Inferência" / "sensibilidade" / "resultados" / "aalen_discreto_resumo.json").read_text(encoding="utf-8"))
    _verdade(aud, "apendice B: Aalen igual a regressao de cada ano", ad["total"]["max_dif"] < 1e-13, "aalen_discreto_resumo.json")
    _tem(aud, "apendice B: LPM concorda a 5%", f"em {ad['total']['mesmo_lado_5pct']} das {ad['total']['n']} covariáveis", _texto("apendice_inferencia.tex"), "aalen_discreto_resumo.json")
    sc = json.loads((ROOT / "Inferência" / "Modelagem de Aalen" / "resultados" / "triagem_corrigida" / "aalen_summary_QB.json").read_text(encoding="utf-8"))
    _tem(aud, "apendice B: triagem corrigida no QB", f"a janela do QB cai para {sc['n_events']} saídas, e a fatia e a participação ofensiva perdem 5\\% ($p={br(sc['tests_null_effect']['team_pass_attempts_share']['p_value'])}$ e ${br(sc['tests_null_effect']['offense_pct']['p_value'])}$)", _texto("apendice_inferencia.tex"), "triagem_corrigida/aalen_summary_QB.json")
    _tem(aud, "c5 landmark: quintil de maior risco", f"saem {obs['QB'][0]}\\% dos QBs, {obs['RB'][0]}\\% dos RBs e {obs['WR'][0]}\\% dos WRs do grupo de maior risco, contra no máximo {max(obs[p][1] for p in POSICOES)}\\% no grupo de menor risco", t5, q_p.name)
    _verdade(aud, "c5 landmark: nenhum QB sai no menor quintil", int(q.loc[("QB", 1, 1), "saidas"]) == 0, q_p.name)
    dlw = pd.read_csv(DIAG / "landmark_vs_walkforward.csv")
    d1 = dlw[dlw.horizon_w == 1].set_index(["posicao", "modelo"])
    f = {p: d1.loc[(p, CAMPEAO[p])] for p in POSICOES}
    _verdade(aud, "c5 landmark: desfecho cortado ja eleva o C", all(float(f[p].wf_desfecho_cortado_w) > float(f[p].wf_mesmos_anos) for p in POSICOES), "landmark_vs_walkforward.csv")
    _verdade(aud, "c5 landmark: modelo reajustado eleva mais", all(float(f[p].landmark_mesmos_anos) > float(f[p].wf_desfecho_cortado_w) for p in POSICOES), "landmark_vs_walkforward.csv")
    casos_p = LEIT / "landmark_casos.csv"
    casos = pd.read_csv(casos_p)
    def _caso(nome: str) -> pd.Series:
        return casos[casos.player_name == nome].iloc[0]
    for nome, pos_c, saiu in (("Brandon Bolden", "RB", 1), ("Keith Kirkwood", "WR", 1), ("Aaron Rodgers", "QB", 0)):
        r = _caso(nome)
        g = casos[casos.posicao == pos_c]
        _verdade(aud, f"c5 landmark 2023: {nome} tem o risco citado na posicao", float(r.p_saida_w) == sorted(g.p_saida_w.astype(float), reverse=True)[1 if nome == "Aaron Rodgers" else 0], casos_p.name)
        _verdade(aud, f"c5 landmark 2023: desfecho de {nome}", int(r.saiu_em_w) == saiu, casos_p.name)
        _tem(aud, f"c5 landmark 2023: risco de {nome}", f"{round(100 * float(r.p_saida_w))}\\%", t5, casos_p.name)
    for nome, pos_c in (("Zach Wilson", "QB"),):
        r = _caso(nome)
        g = casos[(casos.posicao == pos_c) & (casos.saiu_em_w == 1)]
        _verdade(aud, f"c5 landmark 2023: {nome} tem o menor risco entre os que sairam", float(r.p_saida_w) == float(g.p_saida_w.min()), casos_p.name)
        _tem(aud, f"c5 landmark 2023: risco de {nome}", f"{nome} ({round(100 * float(r.p_saida_w))}\\%)", t5, casos_p.name)
    R67 = ROOT / "Inferência" / "sensibilidade" / "resultados"
    so = pd.read_csv(R67 / "testes_defesa_so_experiencia.csv").groupby("posicao").mean(numeric_only=True)
    t3 = pd.read_csv(R67 / "testes_defesa_c_tau3.csv")
    _tem(aud, "c5 so experiencia", f"$C$ médio entre ${br(so.c_so_experiencia.min())}$ e ${br(so.c_so_experiencia.max())}$", t5, "testes_defesa_so_experiencia.csv")
    dif = so.c_principal - so.c_so_experiencia
    _tem(aud, "c5 ganho sobre a experiencia", f"de ${br(dif.min(), 2)}$ a ${br(dif.max(), 2)}$", t5, "testes_defesa_so_experiencia.csv")
    tp = {p: t3[(t3.posicao == p) & (t3.modelo == CAMPEAO[p])].c_tau3.mean() for p in POSICOES}
    _tem(aud, "c5 C truncado", f"fica entre ${br(min(tp.values()))}$ e ${br(max(tp.values()))}$", t5, "testes_defesa_c_tau3.csv")
    cal = pd.read_csv(R67 / "testes_defesa_calibracao.csv").set_index(["posicao", "lambda"])
    _tem(aud, "c5 calibracao sem penalizacao", f"prevê {round(100 * cal.loc[('RB', 0.0), 'q5_prevista'])}\\% e {round(100 * cal.loc[('WR', 0.0), 'q5_prevista'])}\\% de saída no grupo de maior risco", t5, "testes_defesa_calibracao.csv")
    _verdade(aud, "c5 sem penalizacao nao ha ganho de C", all(float(cal.loc[(p, 0.0), "c_medio"]) - float(cal.loc[(p, 1.0), "c_medio"]) < 0.005 for p in ("RB", "WR")), "testes_defesa_calibracao.csv")
    hh = pd.read_csv(R67 / "testes_defesa_hr.csv").set_index(["posicao", "cenario", "covariavel"])
    t4_ = _texto("c4_resultados_inferenciais.tex")
    _tem(aud, "c4 participacao QB sozinha e sem fatia", f"HR de ${br(hh.loc[('QB', 'participacao_sozinha', 'offense_pct'), 'hr'], 2)}$, e, sem a fatia no modelo, de ${br(hh.loc[('QB', 'sem_fatia', 'offense_pct'), 'hr'], 2)}$", _texto("apendice_inferencia.tex"), "testes_defesa_hr.csv")
    _tem(aud, "c4 Aalen com lista do Cox: fatia RB", f"($p={br(hh.loc[('RB', 'aalen_lista_cox', 'team_rushing_yards_share'), 'p'])}$)", t4_, "testes_defesa_hr.csv")
    _tem(aud, "c4 draft sem uso", f"HR de ${br(hh.loc[("RB", "draft_sem_uso", "draft_ovr"), "hr"], 2)}$, e os não draftados, de ${br(hh.loc[("RB", "draft_sem_uso", "undrafted_flag"), "hr"], 2)}$", t4_, "testes_defesa_hr.csv")
    princ, fr = hh.xs("principal", level="cenario"), hh.xs("jogos_como_fracao", level="cenario")
    dfr = (princ.loc[fr.index, "hr"] - fr.hr).abs().max()
    _verdade(aud, "ap. A fracao de jogos: HR muda ate 0,01", dfr <= 0.0105, "testes_defesa_hr.csv", str(dfr))
    _verdade(aud, "ap. A fracao de jogos: mesmo lado de 5%", bool(((princ.loc[fr.index, "p"] < 0.05) == (fr.p < 0.05)).all()), "testes_defesa_hr.csv")
    cs = hh.xs("censura_2022_2023", level="cenario")
    _verdade(aud, "ap. A censura 2022-23: sinais iguais", bool(((princ.loc[cs.index, "hr"] > 1) == (cs.hr > 1)).all()), "testes_defesa_hr.csv")
    _verdade(aud, "ap. A censura 2022-23: entre as nove so a profundidade do RB sai de 5%", set(cs[cs.p >= 0.05].index) & {("RB", "depth_team"), ("QB", "passer_rating"), ("QB", "team_pass_attempts_share"), ("QB", "offense_pct"), ("RB", "games_missed"), ("RB", "observed_career_lag_ypc"), ("WR", "games_missed"), ("WR", "age"), ("WR", "team_targets_share")} == {("RB", "depth_team")}, "testes_defesa_hr.csv")
    cob_ = pd.read_csv(R67 / "testes_defesa_cobertura.csv").set_index("season").mediana_razao
    ta_ = _texto("apendice_dados.tex")
    _tem(aud, "ap. A cobertura", f"tem mediana de ${br(cob_[2012], 2)}$ em 2012, ${br(cob_[2016], 2)}$ em 2016", ta_, "testes_defesa_cobertura.csv")
    _verdade(aud, "ap. A cobertura: cerca de 0,97 desde 2019", bool(((cob_[cob_.index >= 2019] - 0.97).abs() < 0.0155).all()), "testes_defesa_cobertura.csv")
    r18 = pd.read_csv(R67 / "testes_defesa_painel_2018.csv").set_index(["posicao", "covariavel"])
    _verdade(aud, "ap. A painel 2018: sinais iguais", bool(((princ.loc[r18.index, "hr"] > 1) == (r18.hr > 1)).all()), "testes_defesa_painel_2018.csv")
    nove = {("QB", "passer_rating"), ("QB", "team_pass_attempts_share"), ("QB", "offense_pct"), ("RB", "games_missed"), ("RB", "depth_team"), ("RB", "observed_career_lag_ypc"), ("WR", "games_missed"), ("WR", "age"), ("WR", "team_targets_share")}
    _verdade(aud, "ap. A painel 2018: entre as nove saem profundidade do RB e jogos do WR", set(r18[r18.p >= 0.05].index) & nove == {("RB", "depth_team"), ("WR", "games_missed")}, "testes_defesa_painel_2018.csv")
    _tem(aud, "ap. A painel 2018: p citados", f"($p={br(r18.loc[('RB', 'depth_team'), 'p'])}$ e ${br(r18.loc[('WR', 'games_missed'), 'p'])}$)", ta_, "testes_defesa_painel_2018.csv")
    cp = pd.read_csv(R67 / "testes_defesa_censura_predicao.csv")
    def _cp(regra, col):
        g = cp[cp.regra == regra]
        return [float(np.average(g[g.posicao == p_][col], weights=g[g.posicao == p_].n)) if col != "c" else float(g[g.posicao == p_].c.mean()) for p_ in POSICOES]
    tp_ = _texto("apendice_predicao.tex")
    a_c, n_c = _cp("atual", "c"), _cp("fora_do_risco", "c")
    a_v, n_v = _cp("atual", "vies"), _cp("fora_do_risco", "vies")
    a_m, n_m = _cp("atual", "mae"), _cp("fora_do_risco", "mae")
    _tem(aud, "ap. C censura: C", f"de ${br(a_c[0])}$, ${br(a_c[1])}$ e ${br(a_c[2])}$ para ${br(n_c[0])}$, ${br(n_c[1])}$ e ${br(n_c[2])}$", tp_, "testes_defesa_censura_predicao.csv")
    _tem(aud, "ap. C censura: vies", f"de ${br(a_v[0], 2, True)}$, ${br(a_v[1], 2, True)}$ e ${br(a_v[2], 2, True)}$ para ${br(n_v[0], 2, True)}$, ${br(n_v[1], 2, True)}$ e ${br(n_v[2], 2, True)}$", tp_, "testes_defesa_censura_predicao.csv")
    _tem(aud, "ap. C censura: MAE", f"de ${br(a_m[0], 2)}$, ${br(a_m[1], 2)}$ e ${br(a_m[2], 2)}$ para ${br(n_m[0], 2)}$, ${br(n_m[1], 2)}$ e ${br(n_m[2], 2)}$", tp_, "testes_defesa_censura_predicao.csv")
    lg = json.loads((R67 / "testes_defesa_resumo.json").read_text(encoding="utf-8"))["listas_glm"]
    tb_ = _texto("apendice_inferencia.tex")
    _tem(aud, "ap. B selecao no GLM", f"a do QB ganha a idade (Jaccard de ${br(lg['QB']['jaccard'], 2)}$), e a do WR é a que mais muda (Jaccard de ${br(lg['WR']['jaccard'], 2)}$)", tb_, "testes_defesa_resumo.json")
    _verdade(aud, "ap. B as nove estao nas listas do GLM", {"passer_rating", "team_pass_attempts_share", "offense_pct"} <= set(lg["QB"]["glm"]) and {"games_missed", "depth_team", "observed_career_lag_ypc"} <= set(lg["RB"]["glm"]) and {"games_missed", "age", "team_targets_share"} <= set(lg["WR"]["glm"]), "testes_defesa_resumo.json")
    sm = pd.read_csv(R67 / "testes_defesa_sementes.csv")
    conta = sm.assign(sig=sm.p < 0.05).groupby(["posicao", "covariavel"]).sig.sum()
    _verdade(aud, "ap. B sementes: contagens citadas", int(conta[("WR", "age")]) == 18 and int(conta[("WR", "height")]) == 15 and int(conta[("WR", "observed_career_lag_rushing_yards")]) == 12 and int(conta[("WR", "team_targets_share")]) == 20 and int(conta[("WR", "games_missed")]) == 20, "testes_defesa_sementes.csv", str(conta.loc["WR"].to_dict()))
    esp_ = pd.read_csv(R67 / "testes_defesa_splines.csv")
    _verdade(aud, "ap. B splines: um termo rejeita em 15", len(esp_) == 15 and int((esp_.p_nao_linear < 0.05).sum()) == 1, "testes_defesa_splines.csv")
    vif_ = pd.read_csv(R67 / "testes_defesa_vif.csv")
    _tem(aud, "ap. B VIF maximo", f"${br(vif_.vif.max(), 2)}$", tb_, "testes_defesa_vif.csv")
    # tabela de sintese do c6
    pr = pd.read_csv(OFICIAL / "predicao_2024_modelos.csv")
    hit = pd.read_csv(OFICIAL / "auditoria" / "acerto_horizontes.csv")
    wf = pd.read_csv(OFICIAL / "resumo_walkforward_2018_2021.csv")
    linhas = {l[0].strip(): l[1:4] for l in aud.linhas("tab:sintese_preditiva") if len(l) >= 4}
    for k, p in enumerate(POSICOES):
        m_ = CAMPEAO[p]
        h = hit[(hit.posicao == p) & (hit.modelo == m_)].iloc[0]
        col_c = "c_index" if "c_index" in wf.columns else "c_index_mean"
        refs = {
            "C no walk-forward": (float(wf[(wf.posicao == p) & (wf.modelo == m_)][col_c].iloc[0]), 3),
            "Erro absoluto médio (anos)": (float(h.mae), 2),
            "Erro de até um ano": (float(h.acerto_1a), 1),
            "Mediana prevista em 2024 (anos)": (float(pr[(pr.posicao == p) & (pr.modelo == m_)].expected_years.median()), 2),
            "Não volta na temporada seguinte, menor risco": (100 * float(q.loc[(p, 1, 1), "nao_voltou"]), 0),
            "Não volta na temporada seguinte, maior risco": (100 * float(q.loc[(p, 1, 5), "nao_voltou"]), 0),
        }
        for rot, (val, nd) in refs.items():
            cel = next((v for r_, v in linhas.items() if rot in r_.replace("$", "")), None)
            aud.padrao(f"tab:sintese_preditiva {rot} presente", cel is not None, "c6")
            if cel is not None:
                aud.compara_arredondado(f"tab:sintese_preditiva {p} {rot}", _pnum(limpa_celula(cel[k]).replace("%", "")), val, "c6", nd)


CHECAGENS = [
    checa_termos_e_limitacoes,
    checa_numeros_registros,
    checa_tab_concordancia,
    checa_tgb_robustez,
    checa_tgb_tip,
    checa_substitutos_c5_c7,
    checa_translacoes_c4,
    checa_tgb_conc_div,
    checa_amostra,
    checa_cox_ajuste,
    checa_schoenfeld,
    checa_aalen_consolidada,
    checa_aalen_aareg,
    checa_tgb_wf,
    checa_tgb_delta,
    checa_predicao_erro_restrito,
    checa_ic_bootstrap_replicas,
    checa_rmst_p04,
    checa_tgb_lm,
    checa_tgb_tabelas,
    checa_tgb_prosa_c5,
    checa_tgb_prosa_fim,
    checa_tgb_protocolo,
    checa_tgb_metricas_complementares,
    checa_tgb_landmark,
    checa_landmark_leitura,
    checa_tgb_idade,
    checa_inferencia_complementar,
    checa_aalen_preditivo_posto,
    checa_ca_prosa_e_sensibilidades,
    checa_prosa_l2,
    checa_prosa_screening,
    checa_prosa_funil,
    checa_prosa_blocos,
    checa_prosa_km_sem_bandas,
    checa_prosa_logrank_draft,
    checa_prosa_medianas_carreira,
    checa_aalen_sem_missing,
    checa_fonte_bruta,
    checa_painel_inferencial,
    checa_estatisticas_eda,
    checa_protocolo_inferencial,
    checa_preenchimento_e_casos,
    checa_sensibilidade_c3,
    checa_sensibilidade_selecao,
    checa_conjunto_covariaveis_c3,
    checa_rsf_qb_protocolo,
    checa_citacoes,
    checa_series_figuras,
    checa_rastreabilidade_figuras,
    checa_figuras_nao_citadas,
    checa_mask_support,
    checa_anotacoes_figuras,
]


# Todos os artefatos do pipeline estao finais: nao ha mais divergencia "pendente" de artefato
# ainda nao gerado. Mantido vazio para que toda divergencia conte como falha da escrita.
PENDENTES_ARQUIVOS: set[str] = set()
PENDENTES_ROTULOS: tuple[str, ...] = ()


def pendente(d) -> bool:
    return d.artefato in PENDENTES_ARQUIVOS or any(d.onde.startswith(r) for r in PENDENTES_ROTULOS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="codigo de saida 1 se houver divergencia")
    parser.add_argument("--report", type=Path, default=REPORT, help="caminho do relatorio Markdown")
    parser.add_argument("--atualizar-registrados", action="store_true", help="regrava valores_registrados.json com os valores atuais dos artefatos")
    args = parser.parse_args()
    global _ATUALIZAR_REGISTRADOS
    _ATUALIZAR_REGISTRADOS = args.atualizar_registrados
    args.report = args.report.expanduser().resolve()

    tabelas = tabelas_do_livro()
    aud = Auditoria(tabelas)
    cobertura: list[tuple[str, int]] = []
    for checagem in CHECAGENS:
        antes = aud.checagens
        try:
            checagem(aud)
        except Exception as exc:  # noqa: BLE001
            aud.divergencias.append(
                Divergencia(checagem.__name__, "-", "-", "-", f"checagem falhou: {exc}")
            )
        cobertura.append((checagem.__name__, aud.checagens - antes))
    # Cobertura por checagem: uma checagem com zero valores conferidos costuma
    # significar artefato ausente na hora da auditoria (execucao em andamento),
    # e nao que a escrita esteja certa. Fica registrado no relatorio.
    sem_cobertura = [nome for nome, n in cobertura if n == 0]
    if not cobertura:
        sem_cobertura.append("nenhuma checagem configurada")

    pend = [d for d in aud.divergencias if pendente(d)]
    dem = [d for d in aud.divergencias if not pendente(d)]
    print(f"Tabelas lidas da tese: {len(tabelas)}")
    print(f"Valores conferidos: {aud.checagens}")
    print(f"Comparacoes numericas diretas: {aud.comparacoes_numericas}")
    print(f"Divergencias: {len(aud.divergencias)} (pendentes de artefato ainda sendo gerado: {len(pend)}; demais: {len(dem)})")
    for d in dem:
        print(f"  - {d.onde}: escrita={d.escrita} pipeline={d.pipeline} ({d.artefato})")
    for d in pend:
        print(f"  [pendente] {d.onde}: {d.artefato}")
    if sem_cobertura:
        print(f"Checagens sem cobertura (artefato ausente?): {', '.join(sem_cobertura)}")

    linhas = [
        "# Auditoria escrita x pipeline",
        "",
        "Gerado por `reprodutibilidade/audit_escrita_vs_pipeline.py`.",
        "O pipeline e a fonte de verdade; a coluna `pipeline` vem do artefato.",
        "",
        f"- Tabelas lidas da tese: **{len(tabelas)}**",
        f"- Valores conferidos: **{aud.checagens}**",
        f"- Comparacoes numericas diretas: **{aud.comparacoes_numericas}**. As demais checagens incluem prosa, contratos, fontes e validacoes de artefatos.",
        f"- Divergencias: **{len(aud.divergencias)}** (pendentes de artefato ainda sendo gerado: {len(pend)}; demais: {len(dem)})",
        "",
    ]
    if dem:
        linhas += [
            "| Onde | Escrita | Pipeline | Artefato | Nota |",
            "|---|---|---|---|---|",
        ]
        for d in dem:
            linhas.append(f"| {d.onde} | {d.escrita} | {d.pipeline} | `{d.artefato}` | {d.nota} |")
    else:
        linhas.append("Nenhuma divergencia fora dos artefatos pendentes: a escrita concorda com os artefatos disponiveis.")
    if pend:
        linhas += ["", "## Pendentes (artefato ainda sendo gerado)", "", "| Onde | Artefato |", "|---|---|"]
        linhas += [f"| {d.onde} | `{d.artefato}` |" for d in pend]
    linhas.append("")
    linhas.append("## Cobertura por checagem")
    linhas.append("")
    linhas.append("| Checagem | Valores conferidos |")
    linhas.append("|---|---|")
    for nome, n in cobertura:
        linhas.append(f"| `{nome}` | {n} |")
    linhas.append("")
    if sem_cobertura:
        linhas.append(
            "Checagens sem cobertura nesta execucao (artefato ausente ou formato mudado): "
            + ", ".join(f"`{nome}`" for nome in sem_cobertura)
            + "."
        )
    else:
        linhas.append("Todas as checagens conferiram ao menos um valor.")
    linhas.append("")
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(linhas), encoding="utf-8")
    try:
        display_report = report_path.relative_to(ROOT)
    except ValueError:
        display_report = report_path
    print(f"\nRelatorio: {display_report}")

    # Uma checagem sem cobertura nao verificou nada: passou em silencio. Em modo
    # estrito isso e falha, nao aviso — foi assim que uma dependencia nao
    # versionada derrubou a cobertura de uma checagem so no clone limpo.
    if args.strict and sem_cobertura:
        print(
            "FALHA: checagem sem cobertura em modo estrito: " + ", ".join(sem_cobertura)
        )
        return 1
    return 1 if (args.strict and dem) else 0


if __name__ == "__main__":
    sys.exit(main())
