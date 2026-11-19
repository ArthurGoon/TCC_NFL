#!/usr/bin/env python3
"""Testes de robustez dos dois ramos, com resposta medida.

Inferencia: VIF e HR marginal, forma funcional, selecao no modelo discreto, censura das
saidas de 2022 e 2023, jogos perdidos como fracao, codificacao do draft, Aalen com a lista
do Cox e sementes do sorteio do Cox-Aalen.
Predicao: referencia so com experiencia, C truncado em tres anos, calibracao por penalizacao
e intervalo dos quintis do landmarking.

Saidas em Inferência/sensibilidade/resultados/testes_defesa_*.csv e testes_defesa_resumo.json.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
SAIDA = ROOT / "Inferência" / "sensibilidade" / "resultados"
COX = ROOT / "Inferência" / "Modelagem de COX" / "resultados"
AALEN = ROOT / "Inferência" / "Modelagem de Aalen" / "resultados"
CA = ROOT / "Inferência" / "Modelagem Cox-Aalen" / "resultados"
OF = ROOT / "documentos" / "estudo_modelos_finais"
LEIT = ROOT / "documentos" / "estudo_leitura_aplicada" / "resultados"
POS = ("QB", "RB", "WR")
N_SEMENTES = 20


def _modelo_principal() -> dict[str, str]:
    """Modelo de maior C por posicao, lido do resumo oficial, sem lista fixa."""
    sys.path.insert(0, str(ROOT / "Predição" / "shared"))
    from tgb_protocol import modelos_destaque

    resumo = pd.read_csv(OF / "resultados" / "resumo_walkforward_2018_2021.csv")
    return {pos: par[0] for pos, par in modelos_destaque(resumo).items()}


PRINCIPAL = _modelo_principal()


def carrega(nome: str, caminho: Path):
    spec = importlib.util.spec_from_file_location(nome, caminho)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome] = mod
    spec.loader.exec_module(mod)
    return mod


def lista_final(pos: str) -> list[str]:
    d = json.loads((COX / f"final_covs_{pos}.json").read_text(encoding="utf-8"))
    return d["final_covs"] if isinstance(d, dict) and "final_covs" in d else d


def inferencia(s51, s64, tmp: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    hrs, vifs, listas = [], [], {}
    for pos in POS:
        painel = s51.construir_painel(s51.carregar_base(pos), pos, 50, "A")
        final = lista_final(pos)
        df_fit, cand = s64.candidatas_da_cadeia(painel)
        df_fit[["player_id", "start", "stop", "event"] + cand].to_csv(tmp / f"cand_{pos}.csv", index=False)
        (tmp / f"final_{pos}.txt").write_text("\n".join(final) + "\n", encoding="utf-8")
        subprocess.run(["Rscript", str(ROOT / "pipeline" / "67_testes_defesa.R"), str(tmp), pos], check=True, capture_output=True)
        glm = (tmp / f"glm_lista_{pos}.txt").read_text(encoding="utf-8").split("\n")
        glm = [x for x in glm if x]
        listas[pos] = {"cox": final, "glm": glm, "jaccard": len(set(final) & set(glm)) / len(set(final) | set(glm))}

        cols = [c for c in painel.columns if c not in s51.ID_COLS + s51.TIME_COLS and c != "draft_round"]
        imp = s51.apply_imputation_stats(painel, cols, s51.fit_imputation_stats(painel, cols))

        def ajuste(dados: pd.DataFrame, covs: list[str], rotulo: str) -> None:
            d = dados.copy()
            d[covs] = StandardScaler().fit_transform(d[covs])
            r = s51.ajustar_cox(d, covs).summary
            for c in covs:
                hrs.append({"posicao": pos, "cenario": rotulo, "covariavel": c, "hr": float(r.loc[c, "exp(coef)"]), "p": float(r.loc[c, "p"])})

        ajuste(imp, final, "principal")
        cens = imp.copy()
        cens.loc[cens["season"].isin([2022, 2023]) & (cens["event"] == 1), "event"] = 0
        ajuste(cens, final, "censura_2022_2023")
        if "games_missed" in final:
            fr = imp.copy()
            fr["games_missed"] = fr["games_missed"] / np.where(fr["season"] <= 2020, 16.0, 17.0)
            ajuste(fr, final, "jogos_como_fracao")
        if pos == "QB":
            ajuste(imp, ["offense_pct"], "participacao_sozinha")
            ajuste(imp, [c for c in final if c != "team_pass_attempts_share"], "sem_fatia")
        if pos == "RB":
            ajuste(imp, ["draft_ovr", "undrafted_flag"], "draft_sem_uso")
            ajuste(imp, final + ["undrafted_flag"], "final_com_indicador")
        m = imp[final].to_numpy(float)
        inv = np.linalg.inv(np.corrcoef(m, rowvar=False))
        vifs += [{"posicao": pos, "covariavel": c, "vif": float(v)} for c, v in zip(final, np.diag(inv))]
        # Aalen com a lista do Cox: isola o efeito da lista
        pasta = tmp / f"aalen_{pos}"
        pasta.mkdir(exist_ok=True)
        lista_aalen = [c for c in final if not c.endswith("_missing")]
        subprocess.run(["Rscript", str(ROOT / "Inferência" / "Modelagem de Aalen" / "shared" / "run_aalen_tied_inference.R"),
                        str(AALEN / f"aalen_input_{pos}.csv"), str(pasta), pos, ",".join(lista_aalen)], check=True, capture_output=True)
        s = json.loads((pasta / f"aalen_summary_{pos}.json").read_text(encoding="utf-8"))
        for c in lista_aalen:
            hrs.append({"posicao": pos, "cenario": "aalen_lista_cox", "covariavel": c, "hr": np.nan, "p": float(s["tests_null_effect"][c]["p_value"])})
        listas[pos]["janela_aalen_lista_cox"] = [s["janela_inicio"], s["janela_fim"], s["n_events"]]
    return pd.DataFrame(hrs), pd.DataFrame(vifs), listas


def cobertura() -> pd.DataFrame:
    """Jardas recebidas sobre jardas passadas por equipe e temporada, na propria base."""
    r = pd.read_csv(ROOT / "DataBase" / "yearly_player_stats_offense.csv", low_memory=False, usecols=["season", "season_type", "team", "passing_yards", "receiving_yards"])
    r = r[r.season_type == "REG"]
    t = r.groupby(["season", "team"])[["passing_yards", "receiving_yards"]].sum().reset_index()
    t = t[t.passing_yards > 0]
    t["razao"] = t.receiving_yards / t.passing_yards
    return t.groupby("season").razao.median().reset_index(name="mediana_razao")


def painel_recente(s51) -> pd.DataFrame:
    """Cox final so com as temporadas de 2018 em diante, de cobertura mais alta."""
    linhas = []
    for pos in POS:
        painel = s51.construir_painel(s51.carregar_base(pos), pos, 50, "A")
        final = lista_final(pos)
        cols = [c for c in painel.columns if c not in s51.ID_COLS + s51.TIME_COLS and c != "draft_round"]
        imp = s51.apply_imputation_stats(painel, cols, s51.fit_imputation_stats(painel, cols))
        d = imp[imp["season"] >= 2018].copy()
        d[final] = StandardScaler().fit_transform(d[final])
        r = s51.ajustar_cox(d, final).summary
        for c in final:
            linhas.append({"posicao": pos, "covariavel": c, "hr": float(r.loc[c, "exp(coef)"]), "p": float(r.loc[c, "p"]), "saidas": int(d["event"].sum())})
    return pd.DataFrame(linhas)


def sementes(s36, tmp: Path) -> pd.DataFrame:
    linhas = []
    for pos in POS:
        spec = s36.spec_da_posicao(pos)
        ini, fim = s36.janela_aalen(pos)
        for semente in range(1, N_SEMENTES + 1):
            pasta = tmp / f"ca_{pos}_{semente}"
            pasta.mkdir(exist_ok=True)
            env = dict(os.environ, COX_AALEN_SEED=str(semente))
            subprocess.run(["Rscript", str(s36.RSCRIPT), str(COX / f"schoenfeld_input_{pos}.csv"), str(pasta), pos,
                            s36._join_covs(spec["additive"]), s36._join_covs(spec["prop"]), ini, fim], check=True, capture_output=True, env=env)
            b = pd.read_csv(pasta / f"cox_aalen_beta_{pos}.csv")
            col = next(c for c in b.columns if c.lower().startswith("p"))
            nome = next(c for c in b.columns if c.lower() in ("covariate", "term", "covariavel"))
            for _, r in b.iterrows():
                linhas.append({"posicao": pos, "semente": semente, "bloco": "multiplicativo", "covariavel": r[nome], "p": float(r[col])})
            s = json.loads((pasta / f"cox_aalen_summary_{pos}.json").read_text(encoding="utf-8"))
            for c, v in (s.get("tests_null_additive") or s.get("tests_null_effect") or {}).items():
                if "Intercept" not in c:
                    linhas.append({"posicao": pos, "semente": semente, "bloco": "aditivo", "covariavel": c, "p": float(v["p_value"] if isinstance(v, dict) else v)})
    return pd.DataFrame(linhas)


def predicao(of) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ref, cal = [], []
    oficial = pd.read_csv(OF / "resultados" / "metricas_walkforward_folds.csv")
    for pos in POS:
        panel, covs = of.load_panel(pos)
        for season in (2018, 2019, 2020, 2021):
            tr_raw, te_raw = of.walkforward_split(panel, season)
            tr, te, use, _, _ = of.preprocess(tr_raw, te_raw, covs)
            te = of.mask_test_support(tr, te)
            t = of.candidate_times(tr["residual_duration_model"].to_numpy(), te["residual_duration_model"].to_numpy())
            risk, surv, _ = of.fit_predict("cox", {"penalizer": 0.0}, tr, te, ["years_exp"], "residual_duration_model", "residual_event", t)
            m = of.score(tr, te, "residual_duration_model", "residual_event", risk, surv, t, uno=False)
            col_t = "teste_em_temporada" if "teste_em_temporada" in oficial else "fold"
            base = oficial[(oficial.posicao == pos) & (oficial[col_t] == season)].set_index("modelo")
            ref.append({"posicao": pos, "origem": season, "c_so_experiencia": m["c_index"], "ibs_so_experiencia": m["ibs"],
                        "c_principal": float(base.loc[PRINCIPAL[pos], "c_index"]), "ibs_principal": float(base.loc[PRINCIPAL[pos], "ibs"]),
                        "ibs_km": float(base.loc["km", "ibs"])})
        for lam in (0.0, 0.1, 1.0):
            pedacos = []
            for t_l in of.LANDMARKS:
                if not of.landmark_completo(t_l, 1):
                    continue
                tr_raw, te_raw = of.landmark_split(panel, t_l, 1.0)
                tr, te, use, _, _ = of.preprocess(tr_raw, te_raw, covs)
                risk, surv, _ = of.fit_predict("cox", {"penalizer": lam}, tr, te, use, "lm_time", "lm_event", np.array([1.0]))
                c = concordance_index_censored(te["lm_event"].astype(bool).to_numpy(), te["lm_time"].astype(float).to_numpy(), risk)[0]
                o = pd.DataFrame({"p": 1 - surv[:, 0], "y": (te["lm_event"].astype(bool) & (te["lm_time"] <= 1)).astype(int).to_numpy()})
                o["q"] = pd.qcut(o["p"].rank(method="first"), 5, labels=False) + 1
                o["c"] = c
                pedacos.append(o)
            o = pd.concat(pedacos)
            cal.append({"posicao": pos, "lambda": lam, "c_medio": float(o.groupby("c").c.first().mean()),
                        "q5_prevista": float(o[o.q == 5].p.mean()), "q5_observada": float(o[o.q == 5].y.mean()),
                        "q1_prevista": float(o[o.q == 1].p.mean()), "q1_observada": float(o[o.q == 1].y.mean())})
    p = pd.read_csv(OF / "resultados" / "pred_oof_walkforward_2018_2021.csv")
    tr3 = []
    for (pos, mod, s), g in p[p.modelo != "km"].groupby(["posicao", "modelo", "snapshot_season"]):
        t3 = g.residual_duration_model.clip(upper=3.0).astype(float).to_numpy()
        e3 = (g.residual_event.astype(bool) & (g.residual_duration_model <= 3)).to_numpy()
        tr3.append({"posicao": pos, "modelo": mod, "origem": s, "c_tau3": concordance_index_censored(e3, t3, g.risk.to_numpy())[0]})
    return pd.DataFrame(ref), pd.DataFrame(cal), pd.DataFrame(tr3)


def censura_predicao(of) -> pd.DataFrame:
    """Walk-forward do modelo principal com o censurado fora do risco no ultimo intervalo.

    Na resposta do treino, o ativo em s e censurado em s - t, depois das saidas desse tempo,
    embora a saida ao fim de s ainda nao se conheca. A variante desloca esse tempo em meio ano.
    """
    linhas = []
    for pos in POS:
        panel, covs = of.load_panel(pos)
        for season in (2018, 2019, 2020, 2021):
            tr_raw, te_raw = of.walkforward_split(panel, season)
            tr, te, use, _, _ = of.preprocess(tr_raw, te_raw, covs)
            te = of.mask_test_support(tr, te)
            tau = float(2024 - season)
            alvo = np.minimum(te["residual_duration_model"].to_numpy(float), tau)
            alvo = np.where(alvo < 0.01, 0.0, alvo)
            aval = (te["residual_event"].to_numpy() == 1) | (te["residual_duration_model"].to_numpy(float) >= tau - 1e-9)
            for regra in ("atual", "fora_do_risco"):
                t = tr.copy()
                if regra == "fora_do_risco":
                    c = t["residual_event"] == 0
                    t.loc[c, "residual_duration_model"] = (t.loc[c, "residual_duration_model"] - 0.5).clip(lower=0.0005)
                hp = of.hp_do_fold(pos, PRINCIPAL[pos], season)
                risk, _, extra = of.fit_predict(PRINCIPAL[pos], hp, t, te, use, "residual_duration_model", "residual_event", np.array([0.25]))
                if PRINCIPAL[pos] == "cox":
                    previsto = of.expected_cox(extra["model"], te[use], tmax=tau)
                else:
                    previsto = of.expected_sksurv(extra["model"], te[use].to_numpy(float), tmax=tau, exige_dominio=False)
                previsto = np.asarray(previsto[0] if isinstance(previsto, tuple) else previsto, float)
                c_ = concordance_index_censored(te["residual_event"].astype(bool).to_numpy(), te["residual_duration_model"].astype(float).to_numpy(), risk)[0]
                erro = previsto[aval] - alvo[aval]
                linhas.append({"posicao": pos, "origem": season, "regra": regra, "c": c_, "vies": erro.mean(), "mae": np.abs(erro).mean(), "ate_1": (np.abs(erro) <= 1).mean(), "n": int(aval.sum())})
    return pd.DataFrame(linhas)


def ic_quintis() -> pd.DataFrame:
    d = pd.read_csv(LEIT / "landmark_jogadores.csv")
    d = d[(d.w == 1) & (d.completa == 1)]
    r = np.random.default_rng(20241012)
    saida = []
    for (pos, q), g in d[d.quintil.isin([1, 5])].groupby(["posicao", "quintil"]):
        por = g.groupby("player_id")[["saiu_em_w", "nao_voltou"]].agg(["sum", "count"])
        n = len(por)
        amostras = [r.integers(0, n, n) for _ in range(2000)]
        for col in ("saiu_em_w", "nao_voltou"):
            s, c = por[col]["sum"].to_numpy(), por[col]["count"].to_numpy()
            v = np.array([s[i].sum() / c[i].sum() for i in amostras])
            saida.append({"posicao": pos, "quintil": q, "medida": col, "taxa": s.sum() / c.sum(), "ic_baixo": np.percentile(v, 2.5), "ic_alto": np.percentile(v, 97.5)})
    return pd.DataFrame(saida)


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    s64 = carrega("_passo64", ROOT / "pipeline" / "64_sensibilidade_selecao.py")
    s51 = s64.s51
    s36 = carrega("_passo36", ROOT / "pipeline" / "36_cox_aalen_inferencial.py")
    sys.path.insert(0, str(OF))
    of = carrega("run_oficial", OF / "run_oficial.py")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        hrs, vifs, listas = inferencia(s51, s64, tmp)
        esp = pd.concat([pd.read_csv(tmp / f"splines_{p}.csv") for p in POS])
        sem = sementes(s36, tmp)
    ref, cal, tr3 = predicao(of)
    ic = ic_quintis()
    cob, rec = cobertura(), painel_recente(s51)
    cen = censura_predicao(of)
    for nome, df in (("hr", hrs), ("vif", vifs), ("splines", esp), ("sementes", sem), ("so_experiencia", ref), ("calibracao", cal), ("c_tau3", tr3), ("ic_quintis", ic), ("cobertura", cob), ("painel_2018", rec), ("censura_predicao", cen)):
        df.to_csv(SAIDA / f"testes_defesa_{nome}.csv", index=False)
    (SAIDA / "testes_defesa_resumo.json").write_text(json.dumps({"listas_glm": listas, "n_sementes": N_SEMENTES}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
