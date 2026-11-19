#!/usr/bin/env python3
"""Proveniencia e qualidade do CSV bruto.

O capitulo 3 afirma forma do arquivo, cobertura temporal, contagens por
modalidade e posicao, e descreve os problemas de qualidade que motivam os
filtros: jogos negativos, registros multi-time, chaves duplicadas, ausentes.
Este passo grava esses numeros, para que cada afirmacao do texto tenha
lastro reproduzivel.

Saidas (todas em ``DataBase/qualidade/``):

- ``proveniencia.json``: identidade do arquivo (sha256, bytes, forma,
  cobertura, modalidades, posicoes) e o que se sabe da origem;
- ``qualidade_resumo.csv``: contagens de qualidade em formato longo
  (metrica, escopo, valor), incluindo o que a tese cita;
- ``ausentes_por_coluna.csv``: perfil de ausentes das 660 colunas.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "run_all.py").exists() and (candidate / "DataBase").exists():
            return candidate
    raise RuntimeError("Project root not found")


ROOT = find_project_root()
os.chdir(ROOT)

BRUTO = ROOT / "DataBase" / "yearly_player_stats_offense.csv"
SAIDA = ROOT / "DataBase" / "qualidade"
POSICOES = ("QB", "RB", "WR")
SEASON_COLS = "season_"


def sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as handle:
        for bloco in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def registros(
    df: pd.DataFrame,
    escopo: str,
    linhas: list[dict],
    colunas_temporada: list[str],
) -> None:
    """Contagens de qualidade de um recorte (arquivo inteiro ou REG QB/RB/WR)."""
    linhas.append({"metrica": "linhas", "escopo": escopo, "valor": int(len(df))})
    linhas.append(
        {"metrica": "jogadores_distintos", "escopo": escopo, "valor": int(df["player_id"].nunique())}
    )

    chave = ["player_id", "season", "season_type"]
    duplicadas = int(df.duplicated(subset=chave, keep=False).sum())
    linhas.append({"metrica": "linhas_em_chave_duplicada", "escopo": escopo, "valor": duplicadas})

    if "games_missed" in df.columns:
        gm = pd.to_numeric(df["games_missed"], errors="coerce")
        linhas.append(
            {"metrica": "games_missed_negativos", "escopo": escopo, "valor": int((gm < 0).sum())}
        )
        linhas.append(
            {
                "metrica": "games_missed_minimo",
                "escopo": escopo,
                "valor": None if gm.dropna().empty else float(gm.min()),
            }
        )
    if "games_played_season" in df.columns:
        gp = pd.to_numeric(df["games_played_season"], errors="coerce")
        linhas.append(
            {
                "metrica": "games_played_season_maior_que_16",
                "escopo": escopo,
                "valor": int((gp > 16).sum()),
            }
        )

    # Registro multi-time: mesmo jogador e mesma temporada em mais de um time.
    por_temporada = df.groupby(["player_id", "season"])["team"].nunique()
    multi = por_temporada[por_temporada > 1]
    linhas.append(
        {"metrica": "pares_jogador_temporada_multi_time", "escopo": escopo, "valor": int(len(multi))}
    )
    # Denominador do "cerca de 2%": pares (jogador, temporada) apos colapsar as
    # passagens por varios times em um unico registro.
    pares = int(df.drop_duplicates(subset=["player_id", "season"]).shape[0])
    linhas.append({"metrica": "pares_jogador_temporada", "escopo": escopo, "valor": pares})
    linhas.append(
        {
            "metrica": "pct_pares_multi_time",
            "escopo": escopo,
            "valor": round(100 * len(multi) / pares, 4) if pares else None,
        }
    )

    # A escrita afirma que, nesses pares, as colunas de temporada nao divergem:
    # a duplicata repete os agregados da temporada e so o time muda.
    divergentes = 0
    if len(multi) and colunas_temporada:
        multi_idx = set(multi.index)
        for (player, season), grupo in df.groupby(["player_id", "season"]):
            if (player, season) not in multi_idx:
                continue
            if grupo[colunas_temporada].nunique(dropna=False).gt(1).any():
                divergentes += 1
    linhas.append(
        {"metrica": "pares_multi_time_com_season_divergente", "escopo": escopo, "valor": divergentes}
    )


def main() -> int:
    if not BRUTO.exists():
        raise FileNotFoundError(f"CSV bruto ausente: {BRUTO.relative_to(ROOT)}")

    df = pd.read_csv(BRUTO, low_memory=False)
    colunas_temporada = [
        c for c in df.columns if c.startswith(SEASON_COLS) and c != "season_type"
    ]
    SAIDA.mkdir(parents=True, exist_ok=True)

    print(f"bruto: {len(df)} linhas x {len(df.columns)} colunas")
    proveniencia = {
        "arquivo": str(BRUTO.relative_to(ROOT)),
        "sha256": sha256(BRUTO),
        "bytes": int(BRUTO.stat().st_size),
        "linhas": int(len(df)),
        "colunas": int(len(df.columns)),
        "temporadas": {
            "min": int(df["season"].min()),
            "max": int(df["season"].max()),
            "n": int(df["season"].nunique()),
        },
        "modalidades": {str(k): int(v) for k, v in df["season_type"].value_counts().items()},
        "n_posicoes": int(df["position"].nunique()),
        "posicoes": sorted(df["position"].dropna().unique().tolist()),
        # O que o repositorio NAO registra fica declarado como ausente, em vez de
        # ser preenchido por conveniencia: o lastro e o sha256 do arquivo usado.
        "data_coleta": None,
        "nflverse_versao": None,
        "nfl_data_py_versao": None,
        "url_origem": "https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022",
        "origem_declarada": (
            "Conjunto publico de Philip Hyde no Kaggle, montado com nfl_data_py a partir "
            "do nflverse (Capitulo 3). Versao anterior a atualizacao de 22/03/2026, com "
            "temporadas de 2012 a 2024, obtida antes de 23/03/2026 (primeiro commit do "
            "repositorio). A data exata da coleta nao esta registrada; o que identifica o "
            "arquivo usado e o sha256 acima."
        ),
        "gerado_por": "pipeline/49_qualidade_dados.py",
    }
    (SAIDA / "proveniencia.json").write_text(
        json.dumps(proveniencia, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    linhas: list[dict] = []
    registros(df, "arquivo_inteiro", linhas, colunas_temporada)
    for modalidade in ("REG", "POST"):
        sub = df[df["season_type"] == modalidade]
        registros(sub, f"modalidade_{modalidade}", linhas, colunas_temporada)
    reg = df[df["season_type"] == "REG"]
    for posicao in POSICOES:
        sub = reg[reg["position"] == posicao]
        registros(sub, f"REG_{posicao}", linhas, colunas_temporada)

    # Jogador que aparece em mais de um grupo de posicao (afeta a contagem de
    # jogadores distintos da uniao dos paineis).
    grupos = reg[reg["position"].isin(POSICOES)].groupby("player_id")["position"].nunique()
    linhas.append(
        {
            "metrica": "jogadores_em_mais_de_um_grupo_de_posicao",
            "escopo": "REG_QB_RB_WR",
            "valor": int((grupos > 1).sum()),
        }
    )

    resumo = pd.DataFrame(linhas, columns=["metrica", "escopo", "valor"])
    resumo.to_csv(SAIDA / "qualidade_resumo.csv", index=False)

    # Disponibilidade de snaps ofensivos por temporada no recorte modelado: a
    # escrita afirma que a serie e consistente na janela.
    analise = reg[reg["position"].isin(POSICOES)]
    for temporada, grupo in analise.groupby("season"):
        resumo.loc[len(resumo)] = {
            "metrica": "pct_ausente_offense_snaps",
            "escopo": f"temporada_{int(temporada)}",
            "valor": round(100 * float(grupo["offense_snaps"].isna().mean()), 4),
        }
    resumo.loc[len(resumo)] = {
        "metrica": "pct_ausente_offense_snaps",
        "escopo": "REG_QB_RB_WR",
        "valor": round(100 * float(analise["offense_snaps"].isna().mean()), 4),
    }
    resumo.to_csv(SAIDA / "qualidade_resumo.csv", index=False)

    # Casos multi-time com os valores por time: a tabela do Capitulo 3 cita
    # quatro casos concretos (yardas terrestres e recebidas repartidas).
    multi = (
        analise.groupby(["player_id", "season"])["team"]
        .nunique()
        .rename("n_times")
        .reset_index()
        .query("n_times > 1")
    )
    casos = analise.merge(multi[["player_id", "season"]], on=["player_id", "season"], how="inner")
    registros_casos = []
    for (jogador, temporada), grupo in casos.groupby(["player_id", "season"]):
        registros_casos.append(
            {
                "player_id": jogador,
                "player_name": grupo["player_name"].iloc[0],
                "position": grupo["position"].iloc[0],
                "season": int(temporada),
                "times": ";".join(sorted(grupo["team"].astype(str))),
                "n_times": int(grupo["team"].nunique()),
                "season_rushing_yards_por_time": ";".join(
                    f"{v:g}" for v in grupo["season_rushing_yards"].fillna(-1)
                ),
                "season_receiving_yards_por_time": ";".join(
                    f"{v:g}" for v in grupo["season_receiving_yards"].fillna(-1)
                ),
                "season_offense_snaps_por_time": ";".join(
                    f"{v:g}" for v in grupo["season_offense_snaps"].fillna(-1)
                ),
                "season_rushing_yards_max": float(grupo["season_rushing_yards"].max()),
                "season_receiving_yards_max": float(grupo["season_receiving_yards"].max()),
            }
        )
    pd.DataFrame(registros_casos).to_csv(SAIDA / "multitime_casos.csv", index=False)

    # Vinculo entre jogos perdidos negativos e games_played_season acima de 16.
    negativos = analise[pd.to_numeric(analise["games_missed"], errors="coerce") < 0]
    negativos.assign(
        games_missed=pd.to_numeric(negativos["games_missed"], errors="coerce"),
        games_played_season=pd.to_numeric(negativos["games_played_season"], errors="coerce"),
    )[
        ["player_id", "player_name", "position", "season", "team", "games_missed", "games_played_season"]
    ].to_csv(SAIDA / "games_missed_inconsistencias.csv", index=False)

    ausentes = (
        df.isna()
        .sum()
        .rename("n_ausentes")
        .to_frame()
        .assign(pct_ausentes=lambda t: (t["n_ausentes"] / len(df) * 100).round(4))
        .rename_axis("coluna")
        .reset_index()
        .sort_values(["n_ausentes", "coluna"], ascending=[False, True])
    )
    ausentes.to_csv(SAIDA / "ausentes_por_coluna.csv", index=False)

    print(f"gravado: {(SAIDA / 'proveniencia.json').relative_to(ROOT)}")
    print(f"gravado: {(SAIDA / 'qualidade_resumo.csv').relative_to(ROOT)} ({len(resumo)} linhas)")
    print(
        f"gravado: {(SAIDA / 'ausentes_por_coluna.csv').relative_to(ROOT)} "
        f"({len(ausentes)} colunas; {int((ausentes['n_ausentes'] > 0).sum())} com ausentes)"
    )
    print(
        f"gravado: {(SAIDA / 'multitime_casos.csv').relative_to(ROOT)} "
        f"({len(registros_casos)} pares multi-time)"
    )
    print(
        f"gravado: {(SAIDA / 'games_missed_inconsistencias.csv').relative_to(ROOT)} "
        f"({len(negativos)} linhas com jogos negativos)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
