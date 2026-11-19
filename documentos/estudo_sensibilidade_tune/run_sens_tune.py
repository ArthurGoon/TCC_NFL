#!/usr/bin/env python3
"""Sensibilidade da busca aninhada de hiperparametros.

Responde a duas limitacoes declaradas do tune oficial (`estudo_tune_preditivo`):

1. Janela da validacao interna. O oficial usa os 2 ultimos anos do treino. Aqui a busca
   inteira (Cox, GBSA e RSF) e refeita com 1 e com 3 anos.
2. Grade do Cox. O oficial vai ate lambda = 1, valor escolhido em 12 de 12 folds de referencia (2018 a 2021) e em 17 de 18 com 2022 e 2023 (hps_por_fold.csv). Aqui o
   Cox e refeito com a grade estendida ate 10, mantendo 2 anos de validacao interna.

Para cada braco registra: lambda congelado pela mesma regra do oficial (valor escolhido
na maioria dos folds; sem maioria, fica o padrao), quantas combinacoes posicao x familia
a configuracao escolhida pela busca fica abaixo do padrao no teste, e o C medio. Usa as
mesmas funcoes do tune oficial (`run_tune.py`), com pastas de saida proprias: nenhum
artefato oficial e sobrescrito.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RES = HERE / "resultados"
RES.mkdir(parents=True, exist_ok=True)

# Import pelo nome do modulo (e nao por caminho): os processos paralelos do joblib
# precisam reimportar as funcoes de pontuacao.
_PASTA_TUNE = str(ROOT / "documentos" / "estudo_tune_preditivo")
sys.path.insert(0, _PASTA_TUNE)
# Os processos do joblib (loky) nao herdam o sys.path do pai, so o ambiente.
import os  # noqa: E402

os.environ["PYTHONPATH"] = os.pathsep.join(
    [_PASTA_TUNE, str(ROOT / "Predição" / "shared"), str(ROOT / "Inferência" / "shared"), os.environ.get("PYTHONPATH", "")]
)
import run_tune as tune  # noqa: E402

GRADE_COX_ESTENDIDA = (0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0)


def roda_braco(nome: str, cal_years: int, modelos: tuple[str, ...], grade_cox=None) -> pd.DataFrame:
    """Busca aninhada + avaliacao no teste, com os parametros do braco."""
    pasta = RES / nome
    pasta.mkdir(parents=True, exist_ok=True)
    tune.RES = pasta
    tune.FIG = pasta
    tune.CAL_YEARS = cal_years
    if grade_cox is not None:
        tune.GRIDS["cox"] = tune._prepend_default("cox", [{"penalizer": p} for p in grade_cox])
    linhas, escolhas = [], []
    for pos in tune.POSITIONS:
        panel, covs = tune.load_panel(pos)
        for modelo in modelos:
            if modelo == "rsf" and not tune.rsf_permitido(pos):
                continue
            for season in tune.OFFICIAL:
                tr_raw, te_raw = tune.walkforward_split(panel, season)
                if len(tr_raw) < 50 or len(te_raw) < 10 or int(te_raw["residual_event"].sum()) < 1:
                    continue
                t0 = time.time()
                hp, info, _ = tune.choose_hp(modelo, tr_raw, covs, pos, season)
                escolhas.append({"braco": nome, "posicao": pos, "modelo": modelo, "teste_em_temporada": int(season),
                                 "hp": json.dumps(hp, default=str), "inner_c": info.get("inner_c")})
                for tag, h in (("default", tune.DEFAULTS[modelo]), ("tunado", hp)):
                    row = tune.evaluate_hp(modelo, h, tr_raw, te_raw, covs, season, tag, pos)
                    if row is not None:
                        row["braco"] = nome
                        linhas.append(row)
                print(f"[{nome}] {pos} {modelo} {season} hp={hp} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(escolhas).to_csv(pasta / "hps_por_fold.csv", index=False)
    folds = pd.DataFrame(linhas)
    folds.to_csv(pasta / "metricas_folds.csv", index=False)
    return folds, pd.DataFrame(escolhas)


def resume(nome: str, folds: pd.DataFrame, escolhas: pd.DataFrame) -> dict:
    media = folds.groupby(["posicao", "modelo", "tag"])["c_index"].mean().unstack("tag")
    perde = int((media["tunado"] < media["default"]).sum())
    congelado = {}
    for (pos, modelo), g in escolhas.groupby(["posicao", "modelo"]):
        contagem = Counter(g["hp"])
        valor, n = contagem.most_common(1)[0]
        maioria = n > len(g) / 2
        congelado[f"{pos}-{modelo}"] = {
            "valor_majoritario": json.loads(valor) if maioria else None,
            "n_folds_com_o_valor": int(n),
            "n_folds": int(len(g)),
            "por_fold": dict(zip(g["teste_em_temporada"].astype(int), g["hp"])),
        }
    return {
        "braco": nome,
        "n_combinacoes": int(len(media)),
        "busca_abaixo_do_padrao": perde,
        "c_medio": {f"{p}-{m}": {k: round(float(v), 4) for k, v in row.items()} for (p, m), row in media.iterrows()},
        "congelado": congelado,
    }


def main() -> None:
    resumo = {"oficial": "2 anos de validacao interna, grade do Cox ate 1 (estudo_tune_preditivo)"}
    for nome, cal, modelos, grade in (
        ("cal_1ano", 1, ("cox", "gbsa", "rsf"), None),
        ("cal_3anos", 3, ("cox", "gbsa", "rsf"), None),
        ("cox_grade_ate_10", 2, ("cox",), GRADE_COX_ESTENDIDA),
    ):
        folds, escolhas = roda_braco(nome, cal, modelos, grade)
        resumo[nome] = resume(nome, folds, escolhas)
        (RES / "resumo_sens_tune.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    # lambda do Cox: quantos folds escolhem o teto da grade estendida e o lambda majoritario
    cg = resumo["cox_grade_ate_10"]["congelado"]
    resumo["cox_grade_ate_10"]["lambda_por_posicao"] = {k: v["valor_majoritario"] for k, v in cg.items()}
    (RES / "resumo_sens_tune.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: (v if k == "oficial" else {kk: vv for kk, vv in v.items() if kk != "congelado"}) for k, v in resumo.items()}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
