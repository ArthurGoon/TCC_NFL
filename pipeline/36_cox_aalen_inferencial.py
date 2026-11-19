#!/usr/bin/env python3
"""Cox-Aalen inferencial (Scheike & Zhang 2002 / timereg::cox.aalen).

Mesmas covariaveis do Cox com covariaveis dependentes do tempo, com a divisao
entre termos aditivos e multiplicativos dada pelo Schoenfeld. Nao reajusta Cox nem Aalen.
"""
from __future__ import annotations

import json
import os
import subprocess
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

INFER_DIR = ROOT / "Inferência"
COX_RES = INFER_DIR / "Modelagem de COX" / "resultados"
AALEN_RES = INFER_DIR / "Modelagem de Aalen" / "resultados"
CA_DIR = INFER_DIR / "Modelagem Cox-Aalen"
CA_SHARED = CA_DIR / "shared"
CA_RES = CA_DIR / "resultados"
CA_FIG = CA_DIR / "figuras"
RSCRIPT = CA_SHARED / "run_cox_aalen_inference.R"

CA_RES.mkdir(parents=True, exist_ok=True)
CA_FIG.mkdir(parents=True, exist_ok=True)

def spec_da_posicao(pos: str) -> dict[str, list[str]]:
    """Divisao dos blocos lida do proprio Cox reportado e do seu teste de Schoenfeld.

    Cada covariavel do Cox final com Schoenfeld individual p < 0,05 vai para o bloco aditivo
    (efeito livre no tempo); as demais ficam no bloco multiplicativo. Nada e fixado a mao.
    """
    covs = pd.read_csv(COX_RES / f"cox_hr_{pos}.csv")["covariate"].tolist()
    sch = pd.read_csv(COX_RES / f"schoenfeld_{pos}.csv")
    sch = sch[sch["term"] != "GLOBAL"].set_index("term")["p"]
    aditivo = [c for c in covs if c in sch.index and float(sch[c]) < 0.05]
    return {"additive": aditivo, "prop": [c for c in covs if c not in aditivo]}


def _join_covs(covs: list[str]) -> str:
    return ",".join(covs) if covs else "-"


def janela_aalen(pos: str) -> tuple[str, str]:
    """Janela de posto completo do ajuste Aalen da posicao (start.time, max.time), lida do passo 14/15/16."""
    resumo = json.loads((AALEN_RES / f"aalen_summary_{pos}.json").read_text(encoding="utf-8"))
    return str(resumo["janela_inicio"]), str(resumo["janela_fim"])


def fit_position(pos: str) -> pd.DataFrame:
    spec = spec_da_posicao(pos)
    input_csv = COX_RES / f"schoenfeld_input_{pos}.csv"
    cox_hr_path = COX_RES / f"cox_hr_{pos}.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Entrada do Cox ausente: {input_csv}")
    if not cox_hr_path.exists():
        raise FileNotFoundError(f"HRs do Cox ausentes: {cox_hr_path}")
    ini, fim = janela_aalen(pos)

    cmd = [
        "Rscript",
        str(RSCRIPT),
        str(input_csv),
        str(CA_RES),
        pos,
        _join_covs(spec["additive"]),
        _join_covs(spec["prop"]),
        ini,
        fim,
    ]
    print(f"\n=== {pos} ===")
    print("  Aditivo:", spec["additive"] or "(somente intercepto)")
    print("  Multiplicativo:", spec["prop"])
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"cox.aalen falhou para {pos} (exit {completed.returncode})")

    beta = pd.read_csv(CA_RES / f"cox_aalen_beta_{pos}.csv")
    cox_hr = pd.read_csv(cox_hr_path)
    merged = beta.merge(
        cox_hr.rename(
            columns={
                "coef": "cox_coef",
                "HR": "cox_HR",
                "se": "cox_se",
                "CI_lower": "cox_CI_lower",
                "CI_upper": "cox_CI_upper",
                "p_value": "cox_p_value",
            }
        ),
        on="covariate",
        how="left",
    )
    merged["hr_ratio_ca_over_cox"] = merged["HR"] / merged["cox_HR"]
    merged["sign_match"] = (merged["coef"] * merged["cox_coef"]) > 0
    merged["position"] = pos
    out_path = CA_RES / f"cox_aalen_vs_cox_{pos}.csv"
    merged.to_csv(out_path, index=False)
    return merged


def build_unpenalized_comparison() -> None:
    """Compara o Cox oficial (lifelines, sem penalizacao: penalizer = 0), o `coxph` e o Cox-Aalen.

    O `coxph` vem do mesmo ajuste que sustenta o teste de Schoenfeld (exportado
    pelos passos de Cox). Ambos os ajustes Cox desta comparacao sao sem
    penalizacao; nao e um contraste entre ridge e ausencia de ridge.
    """
    origem = CA_RES / "cox_aalen_vs_cox_all.csv"
    if not origem.exists():
        print("  (sem cox_aalen_vs_cox_all.csv; comparacao nao penalizada pulada)")
        return
    base = pd.read_csv(origem)
    linhas = []
    for pos in ("QB", "RB", "WR"):
        sub = base[base["position"] == pos]
        if sub.empty:
            continue
        nao_pen = COX_RES / f"coxph_unpenalized_{pos}.csv"
        sch = COX_RES / f"schoenfeld_{pos}.csv"
        unp = pd.read_csv(nao_pen).set_index("covariate") if nao_pen.exists() else pd.DataFrame()
        schs = pd.read_csv(sch).set_index("term") if sch.exists() else pd.DataFrame()
        for _, r in sub.iterrows():
            cov = r["covariate"]
            linha = {
                "position": pos,
                "covariate": cov,
                "coxph_coef": unp.loc[cov, "coxph_coef"] if cov in unp.index else float("nan"),
                "coxph_HR": unp.loc[cov, "coxph_HR"] if cov in unp.index else float("nan"),
                "coxph_p": unp.loc[cov, "coxph_p"] if cov in unp.index else float("nan"),
                "schoenfeld_p": schs.loc[cov, "p"] if cov in schs.index else float("nan"),
                "ca_HR": r["HR"],
                "ca_p": r["p_value"],
                "p_proportionality": r["p_proportionality"],
                "lifelines_unpenalized_HR": r["cox_HR"],
                "lifelines_unpenalized_p": r["cox_p_value"],
            }
            linha["ratio_ca_over_coxph"] = (
                linha["ca_HR"] / linha["coxph_HR"] if linha["coxph_HR"] else float("nan")
            )
            linhas.append(linha)
    if not linhas:
        return
    saida = CA_RES / "cox_aalen_vs_coxph_unpenalized.csv"
    pd.DataFrame(linhas).to_csv(saida, index=False)
    print(f"  {saida.name}: {len(linhas)} comparacoes gravadas")


def main() -> int:
    print("Cox-Aalen inferencial")
    print(f"R: {RSCRIPT}")
    print(f"Input: {COX_RES / 'schoenfeld_input_{POS}.csv'}")

    rows: list[pd.DataFrame] = []
    for pos in ("QB", "RB", "WR"):
        rows.append(fit_position(pos))

    all_cmp = pd.concat(rows, ignore_index=True)
    all_cmp.to_csv(CA_RES / "cox_aalen_vs_cox_all.csv", index=False)
    build_unpenalized_comparison()

    print("\n=== Comparacao HR Cox vs exp(beta) Cox-Aalen ===")
    show = all_cmp[
        [
            "position",
            "covariate",
            "cox_HR",
            "HR",
            "hr_ratio_ca_over_cox",
            "sign_match",
            "p_value",
        ]
    ].copy()
    show["cox_HR"] = show["cox_HR"].round(3)
    show["HR"] = show["HR"].round(3)
    show["hr_ratio_ca_over_cox"] = show["hr_ratio_ca_over_cox"].round(3)
    print(show.to_string(index=False))

    n_flip = int((~all_cmp["sign_match"]).sum())
    print(f"\nInversoes de sinal vs Cox: {n_flip}/{len(all_cmp)}")
    print(f"Saidas: {CA_RES}")
    print(f"Figuras: {CA_FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
