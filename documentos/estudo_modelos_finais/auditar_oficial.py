#!/usr/bin/env python3
"""Auditoria do pacote oficial: protocolo, top-k, erros OOF, face validity 2024.

Não refita modelos. Lê os CSVs de run_oficial.py e o painel cru.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Predição" / "shared"))
from tgb_protocol import (  # noqa: E402
    LANDMARK_WS,
    LANDMARKS,
    landmark_completo,
    OFFICIAL_SEASONS,
    RSF_MIN_EVENTOS,
    carrega_hps_aninhados,
    hp_aninhado,
    modelos_destaque,
    landmark_outcomes,
    load_covs,
    load_panel,
    restricted_time_identifiable,
    restricted_time_target,
    rsf_permitido,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "resultados"
OUT = RES / "auditoria"
OUT.mkdir(parents=True, exist_ok=True)

POSITIONS = ("QB", "RB", "WR")
# Modelo de maior C e modelo de contraste por posicao, lidos do resultado (sem lista fixa).
_DESTAQUE = modelos_destaque(pd.read_csv(RES / "resumo_walkforward_2018_2021.csv"))
CHAMPION = {pos: par[0] for pos, par in _DESTAQUE.items()}
CONTRASTE = {pos: par[1] for pos, par in _DESTAQUE.items()}
PRED_MODELS = {"QB": ("cox", "gbsa", "rsf"), "RB": ("cox", "gbsa", "rsf"), "WR": ("cox", "gbsa", "rsf")}


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def raw_meta(pos: str) -> pd.DataFrame:
    panel, _ = load_panel(pos)
    cols = [
        c
        for c in (
            "player_id",
            "player_name",
            "snapshot_season",
            "years_exp",
            "age",
            "depth_team",
            "offense_pct",
            "residual_duration",
            "residual_duration_model",
            "residual_event",
        )
        if c in panel.columns
    ]
    return panel[cols].copy()


def event_inventory() -> list[dict]:
    rows = []
    for pos in POSITIONS:
        panel, covs = load_panel(pos)
        last = panel.sort_values("snapshot_season").groupby("player_id").tail(1)
        rows.append(
            {
                "posicao": pos,
                "n_snapshots": int(len(panel)),
                "n_jogadores": int(panel.player_id.nunique()),
                "n_eventos_jogador": int(last["residual_event"].sum()),
                "n_covs_json": int(len(covs)),
                "n_ativos_2024": int((panel.snapshot_season == 2024).sum()),
                "residual_2024_mediana": fmt(panel.loc[panel.snapshot_season == 2024, "residual_duration"].median(), 3),
                "residual_2024_max": fmt(panel.loc[panel.snapshot_season == 2024, "residual_duration"].max(), 3),
                "eventos_2024": int(panel.loc[panel.snapshot_season == 2024, "residual_event"].sum()),
            }
        )
    return rows


def rsf_diagnostico(wf: pd.DataFrame, inv: list[dict]) -> list[dict]:
    """RSF por posição: eventos observados, se foi ajustado e onde a regra vive.

    A decisão vem de tgb_protocol.rsf_permitido (posição, não limiar); n_eventos e
    RSF_MIN_EVENTOS entram como diagnóstico. O resultado é conferido contra os folds.
    """
    inv_map = {r["posicao"]: r for r in inv}
    off = wf[wf.oficial == 1]
    rows = []
    for pos in POSITIONS:
        n_eventos = int(inv_map[pos]["n_eventos_jogador"])
        n_folds = int(len(off[(off.posicao == pos) & (off.modelo == "rsf")]))
        ajustado = bool(rsf_permitido(pos, n_eventos))
        if ajustado != (n_folds > 0):
            raise AssertionError(
                f"RSF {pos}: a regra diz ajustado={ajustado} e os folds oficiais têm {n_folds}"
            )
        rows.append(
            {
                "posicao": pos,
                "n_eventos": n_eventos,
                "rsf_ajustado": ajustado,
                "n_folds": n_folds,
                "regra_aplicada": (
                    "posicao_em_RSF_POSICOES_COM_RSF"
                    if ajustado
                    else "posicao_fora_de_RSF_POSICOES_COM_RSF"
                ),
                "limiar_eventos": int(RSF_MIN_EVENTOS),
                "fonte_regra": "Predição/shared/tgb_protocol.py::rsf_permitido",
            }
        )
    return rows


def followup_max(meta: dict[str, pd.DataFrame]) -> list[dict]:
    """Teto de follow-up observado no snapshot de teste de cada landmark.

    max_followup_observado é o máximo da duração residual no teste: com follow-up
    de um ano, o horizonte w=3 de tL=2023 não tem janela para ser avaliado.
    """
    rows = []
    for pos in POSITIONS:
        df = meta[pos]
        for t_l in LANDMARKS:
            te = df[df["snapshot_season"] == t_l]
            if te.empty:
                continue
            dur = pd.to_numeric(te["residual_duration_model"], errors="coerce")
            for w in LANDMARK_WS:
                lm = landmark_outcomes(te, float(w))
                rows.append(
                    {
                        "posicao": pos,
                        "landmark_tL": int(t_l),
                        "horizon_w": int(w),
                        "max_followup_observado": fmt(dur.max(), 3),
                        "n_teste": int(len(te)),
                        "n_eventos": int(lm["lm_event"].sum()),
                    }
                )
    return rows


def protocol_checks(wf: pd.DataFrame, lm: pd.DataFrame, inv: list[dict]) -> list[dict]:
    checks = []

    def add(ok: bool, codigo: str, detalhe: str):
        checks.append({"ok": bool(ok), "codigo": codigo, "detalhe": detalhe})

    off = wf[wf.oficial == 1]
    # HP por fold: cada linha do walk-forward oficial e do landmark tem de usar exatamente o
    # HP que a busca aninhada escolheu para aquele (posicao, modelo, origem). Sem HP global.
    hps = carrega_hps_aninhados()
    divergentes: list[str] = []
    conferidos = {"walkforward": 0, "landmark": 0}

    def confere(fonte: str, linhas, origem_col: str) -> None:
        for r in linhas.itertuples():
            origem = int(getattr(r, origem_col))
            try:
                esperado = hp_aninhado(hps, r.posicao, r.modelo, origem)
            except KeyError as exc:
                divergentes.append(f"{fonte} {r.posicao} {r.modelo} {origem}: sem HP na busca ({exc})")
                continue
            lido = json.loads(r.hp_json) if isinstance(r.hp_json, str) else None
            conferidos[fonte] += 1
            if lido != esperado:
                divergentes.append(f"{fonte} {r.posicao} {r.modelo} {origem}: lido={lido} busca={esperado}")

    if "hp_json" not in wf.columns:
        add(False, "hp_por_fold_walkforward", "coluna hp_json ausente: rode run_oficial.py novamente")
    else:
        off_m = off[off.modelo.isin(["cox", "gbsa", "rsf"])]
        confere("walkforward", off_m, "teste_em_temporada")
        modos_ok = ("hp_modo" in wf.columns) and set(wf.hp_modo.unique()) <= {"nested"}
        add(not divergentes and conferidos["walkforward"] > 0 and modos_ok,
            "hp_por_fold_walkforward",
            f"linhas conferidas={conferidos['walkforward']}; divergencias={len(divergentes)}; "
            f"modos={sorted(wf.hp_modo.unique()) if 'hp_modo' in wf.columns else 'ausente'}")

    lm_m = lm[lm.modelo.isin(["cox", "gbsa", "rsf"])].drop_duplicates(["posicao", "modelo", "landmark"])
    if "hp_json" in lm.columns:
        confere("landmark", lm_m, "landmark")
    add(len([d for d in divergentes if d.startswith("landmark")]) == 0 and conferidos["landmark"] > 0,
        "hp_por_fold_landmark",
        f"origens conferidas={conferidos['landmark']}; origem = HP do fold walk-forward de mesma temporada")

    # RSF em todas as posicoes: no QB ele tem de estar nos folds oficiais, como em RB e WR.
    rsf_por_pos = {pos: int(len(off[(off.posicao == pos) & (off.modelo == "rsf")])) for pos in POSITIONS}
    add(all(rsf_por_pos[p] > 0 and rsf_permitido(p) for p in POSITIONS), "rsf_todas_posicoes",
        f"folds oficiais com RSF por posicao={rsf_por_pos}")

    add(len(divergentes) == 0, "hp_sem_divergencia",
        "; ".join(divergentes[:5]) if divergentes else "nenhuma divergencia")

    km = off[off.modelo == "km"]
    add(bool((km.c_index == 0.5).all()), "km_piso", f"min={km.c_index.min()} max={km.c_index.max()}")

    ncov = {pos: int(off.loc[off.posicao == pos, "n_covs"].iloc[0]) for pos in POSITIONS}
    want = {pos: int(len(load_covs(pos))) for pos in POSITIONS}
    add(ncov == want, "n_covs", f"folds={ncov} json={want}")

    rsf_qb = off[(off.posicao == "QB") & (off.modelo == "rsf")]
    add(len(rsf_qb) == len(off[(off.posicao == "QB") & (off.modelo == "cox")]), "rsf_qb_dentro",
        f"n_folds_rsf_qb={len(rsf_qb)} (mesmo numero de folds do Cox no QB)")

    inv_map = {r["posicao"]: r for r in inv}
    # A regra vive no protocolo e é de posição: RSF_MIN_EVENTOS é diagnóstico, não limiar.
    add(all(rsf_permitido(p, inv_map[p]["n_eventos_jogador"]) for p in POSITIONS),
        "rsf_regra_protocolo",
        f"rsf_permitido em tgb_protocol (RSF_MIN_EVENTOS={RSF_MIN_EVENTOS}, diagnóstico)")
    add(inv_map["QB"]["n_eventos_jogador"] < RSF_MIN_EVENTOS, "rsf_qb_eventos",
        f"QB eventos={inv_map['QB']['n_eventos_jogador']} (diagnóstico; limiar de referência {RSF_MIN_EVENTOS})")

    add(inv_map["QB"]["eventos_2024"] == 0 and inv_map["RB"]["eventos_2024"] == 0
        and inv_map["WR"]["eventos_2024"] == 0,
        "2024_sem_rotulo",
        "eventos em 2024: " + ", ".join(f"{p}={inv_map[p]['eventos_2024']}" for p in POSITIONS))

    add(float(inv_map["QB"]["residual_2024_max"] or 0) < 0.05
        and float(inv_map["RB"]["residual_2024_max"] or 0) < 0.05,
        "2024_residual_zero",
        "residual max 2024: " + ", ".join(f"{p}={inv_map[p]['residual_2024_max']}" for p in POSITIONS))

    add(set(off.teste_em_temporada.unique()) == set(OFFICIAL_SEASONS),
        "oficial_2018_2021", str(sorted(off.teste_em_temporada.unique())))

    add("landmark" in set(lm.protocolo.unique()) and set(lm.horizon_w.unique()) == {1, 2, 3},
        "landmark_vanhouwelingen", f"w={sorted(lm.horizon_w.unique())} tL={sorted(lm.landmark.unique())}")

    uno = off[off.modelo != "km"]
    add(bool((uno.uno_degenerado == 1).all()), "uno_degenerado",
        "Uno=Harrell neste recorte; não reportar Uno como métrica extra")

    ibs_late = wf[(wf.oficial == 0) & (wf.modelo != "km")]["ibs"]
    add(bool(ibs_late.isna().all()), "ibs_2022_2023_vazio",
        "IBS ausente nos folds curtos, como o protocolo prevê")

    return checks


def fold_winners(wf: pd.DataFrame) -> list[dict]:
    rows = []
    off = wf[(wf.oficial == 1) & (wf.modelo != "km")]
    for pos in POSITIONS:
        for season, sub in off[off.posicao == pos].groupby("teste_em_temporada"):
            best = sub.loc[sub.c_index.idxmax()]
            champ = CHAMPION[pos]
            champ_c = float(sub.loc[sub.modelo == champ, "c_index"].iloc[0])
            rows.append(
                {
                    "posicao": pos,
                    "temporada": int(season),
                    "vencedor_fold": best.modelo,
                    "c_vencedor": fmt(best.c_index),
                    "campeao_oficial": champ,
                    "c_campeao": fmt(champ_c),
                    "campeao_venceu_fold": best.modelo == champ,
                }
            )
    return rows


def overlap_block(wide: pd.DataFrame, models: tuple[str, ...], k: int) -> dict:
    tops = {m: set(wide.nlargest(k, m).index) for m in models if m in wide.columns}
    out = {"k": k, "n": int(len(wide))}
    pairs = []
    mods = [m for m in models if m in wide.columns]
    for i, a in enumerate(mods):
        for b in mods[i + 1 :]:
            inter = len(tops[a] & tops[b])
            pairs.append({"par": f"{a}-{b}", "intersecao": inter, "jaccard": fmt(inter / len(tops[a] | tops[b]), 3)})
    out["pares"] = pairs
    if len(mods) >= 2:
        common = set.intersection(*[tops[m] for m in mods])
        out["intersecao_todos"] = int(len(common))
    ranks = {m: wide[m].rank(ascending=False, method="average") for m in mods}
    disp = {}
    for i, a in enumerate(mods):
        for b in mods[i + 1 :]:
            d = (ranks[a] - ranks[b]).abs()
            disp[f"{a}-{b}"] = {
                "mediana_abs_rank": fmt(d.median(), 2),
                "pct_rank_gt5": fmt(100 * (d > 5).mean(), 1),
                "pct_rank_gt10": fmt(100 * (d > 10).mean(), 1),
                "mediana_abs_anos": fmt((wide[a] - wide[b]).abs().median(), 2),
            }
    out["deslocamento"] = disp
    return out


def topk_tables(pred: pd.DataFrame, meta: dict[str, pd.DataFrame], k: int = 10) -> dict:
    payload = {}
    overlap = {}
    for pos in POSITIONS:
        mods = PRED_MODELS[pos]
        sub = pred[pred.posicao == pos].copy()
        m = meta[pos][["player_id", "snapshot_season", "years_exp", "age", "depth_team", "offense_pct"]].drop_duplicates(
            ["player_id", "snapshot_season"]
        )
        m = m.rename(columns={"years_exp": "years_exp_raw"})
        sub = sub.merge(m, on=["player_id", "snapshot_season"], how="left")
        wide_e = sub.pivot_table(index=["player_id", "player_name"], columns="modelo", values="expected_years")
        wide_r = sub.pivot_table(index=["player_id", "player_name"], columns="modelo", values="rank")
        extra = (
            sub.drop_duplicates("player_id")
            .set_index(["player_id", "player_name"])[["years_exp_raw", "age", "depth_team", "offense_pct"]]
        )
        champ = CHAMPION[pos]
        order = wide_e[champ].sort_values(ascending=False)
        top = []
        for (pid, name), _ in order.head(k).items():
            row = {
                "player_id": pid,
                "player_name": name,
                "years_exp": fmt(extra.loc[(pid, name), "years_exp_raw"], 2) if (pid, name) in extra.index else None,
                "age": fmt(extra.loc[(pid, name), "age"], 1) if (pid, name) in extra.index else None,
                "depth": fmt(extra.loc[(pid, name), "depth_team"], 2) if (pid, name) in extra.index else None,
            }
            for mname in mods:
                row[f"anos_{mname}"] = fmt(wide_e.loc[(pid, name), mname], 2)
                row[f"rank_{mname}"] = int(wide_r.loc[(pid, name), mname])
            top.append(row)
        bot = []
        for (pid, name), _ in order.tail(k).iloc[::-1].items():
            row = {
                "player_id": pid,
                "player_name": name,
                "years_exp": fmt(extra.loc[(pid, name), "years_exp_raw"], 2) if (pid, name) in extra.index else None,
            }
            for mname in mods:
                row[f"anos_{mname}"] = fmt(wide_e.loc[(pid, name), mname], 2)
                row[f"rank_{mname}"] = int(wide_r.loc[(pid, name), mname])
            bot.append(row)
        payload[pos] = {"top": top, "bottom": bot, "n": int(len(wide_e))}
        overlap[pos] = {str(kk): overlap_block(wide_e, mods, kk) for kk in (5, 10, 20)}
    return {"listas": payload, "overlap": overlap}


def named_sanity(pred: pd.DataFrame) -> list[dict]:
    want = {
        "QB": [
            "Patrick Mahomes",
            "Josh Allen",
            "Lamar Jackson",
            "Joe Burrow",
            "Justin Herbert",
            "C.J. Stroud",
            "Aaron Rodgers",
            "Joe Flacco",
            "Josh Johnson",
            "Matthew Stafford",
            "Dak Prescott",
            "Jalen Hurts",
        ],
        "RB": [
            "Christian McCaffrey",
            "Saquon Barkley",
            "Derrick Henry",
            "Bijan Robinson",
            "Jahmyr Gibbs",
            "Nick Chubb",
            "Ezekiel Elliott",
        ],
        "WR": [
            "Justin Jefferson",
            "Ja'Marr Chase",
            "CeeDee Lamb",
            "Tyreek Hill",
            "Davante Adams",
            "A.J. Brown",
            "Mike Evans",
            "Stefon Diggs",
        ],
    }
    rows = []
    for pos, names in want.items():
        sub = pred[pred.posicao == pos]
        champ = CHAMPION[pos]
        n = sub[sub.modelo == champ].shape[0]
        for name in names:
            hit = sub[sub.player_name.str.lower() == name.lower()]
            if hit.empty:
                # fuzzy contains
                hit = sub[sub.player_name.str.lower().str.contains(name.lower().split()[-1])]
                # still may be too wide; keep exact miss
                exact = sub[sub.player_name == name]
                hit = exact if not exact.empty else hit.head(0)
            if hit.empty:
                rows.append({"posicao": pos, "player_name": name, "presente": False})
                continue
            pid = hit.player_id.iloc[0]
            block = sub[sub.player_id == pid]
            rec = {"posicao": pos, "player_name": block.player_name.iloc[0], "presente": True, "n_elenco": n}
            for _, r in block.iterrows():
                rec[f"anos_{r.modelo}"] = fmt(r.expected_years, 2)
                rec[f"rank_{r.modelo}"] = int(r["rank"])
            rows.append(rec)
    return rows


def oof_errors(oof: pd.DataFrame) -> dict:
    """Erros só onde há rótulo: folds 2018-2021. 2024 não entra."""
    summary = []
    worst = []
    best = []
    shared = []
    # Tabela por horizonte: e a que a Tabela 5.3 da tese cita e a que a figura
    # tgb_acerto_horizontes consome. Antes este CSV nao tinha produtor nenhum.
    hits = []
    for pos in POSITIONS:
        mods = PRED_MODELS[pos]
        sub = oof[oof.posicao == pos].copy()
        sub["error_identifiable"] = restricted_time_identifiable(
            sub.residual_event.to_numpy(),
            sub.residual_duration_model.to_numpy(float),
            sub.tau.to_numpy(float),
        )
        sub["error_target_years"] = restricted_time_target(
            sub.residual_duration_model.to_numpy(float),
            sub.tau.to_numpy(float),
            sub.error_identifiable.to_numpy(bool),
        )
        ev = sub[sub.error_identifiable].copy()
        cens = ev[ev.residual_event == 0].copy()
        for m in mods:
            e = ev[ev.modelo == m]
            err = e.expected_years - e.error_target_years
            c = cens[cens.modelo == m]
            summary.append(
                {
                    "posicao": pos,
                    "modelo": m,
                    "n_pares": int(len(e)),
                    "n_pares_total": int(len(sub[sub.modelo == m])),
                    "n_excluidos_censura_pre_tau": int((~sub.loc[sub.modelo == m, "error_identifiable"]).sum()),
                    "n_censurados": int(len(c)),
                    "mae": fmt(err.abs().mean(), 2),
                    "mediana_ae": fmt(err.abs().median(), 2),
                    "vies": fmt(err.mean(), 2),
                    "pct_superestima": fmt(100 * (err > 1.0).mean(), 1),
                    "pct_subestima": fmt(100 * (err < -1.0).mean(), 1),
                    "pct_acerto_1a": fmt(100 * (err.abs() <= 1.0).mean(), 1),
                }
            )
            hits.append(
                {
                    "posicao": pos,
                    "modelo": m,
                    "n": int(len(e)),
                    "mae": fmt(err.abs().mean(), 2),
                    "mediana_ae": fmt(err.abs().median(), 2),
                    "vies": fmt(err.mean(), 2),
                    "acerto_1a": fmt(100 * (err.abs() <= 1.0).mean(), 1),
                    "acerto_2a": fmt(100 * (err.abs() <= 2.0).mean(), 1),
                    "acerto_3a": fmt(100 * (err.abs() <= 3.0).mean(), 1),
                    "acerto_5a": fmt(100 * (err.abs() <= 5.0).mean(), 1),
                }
            )
            tmp = e.assign(erro=err, ae=err.abs())
            for _, r in tmp.nlargest(8, "erro").iterrows():
                worst.append(
                    {
                        "tipo": "superestima",
                        "posicao": pos,
                        "modelo": m,
                        "player_name": r.player_name,
                        "temporada": int(r.snapshot_season),
                        "previsto": fmt(r.expected_years, 2),
                        "observado": fmt(r.residual_duration_model, 2),
                        "erro": fmt(r.erro, 2),
                    }
                )
            for _, r in tmp.nsmallest(5, "erro").iterrows():
                worst.append(
                    {
                        "tipo": "subestima",
                        "posicao": pos,
                        "modelo": m,
                        "player_name": r.player_name,
                        "temporada": int(r.snapshot_season),
                        "previsto": fmt(r.expected_years, 2),
                        "observado": fmt(r.residual_duration_model, 2),
                        "erro": fmt(r.erro, 2),
                    }
                )
            for _, r in tmp.nsmallest(6, "ae").iterrows():
                best.append(
                    {
                        "posicao": pos,
                        "modelo": m,
                        "player_name": r.player_name,
                        "temporada": int(r.snapshot_season),
                        "previsto": fmt(r.expected_years, 2),
                        "observado": fmt(r.residual_duration_model, 2),
                        "ae": fmt(r.ae, 2),
                    }
                )

        # erros compartilhados em todos os modelos na coorte identificavel (|erro|>2)
        champ = CHAMPION[pos]
        wide = ev.pivot_table(
            index=["player_id", "player_name", "snapshot_season", "error_target_years"],
            columns="modelo",
            values="expected_years",
        ).dropna()
        keys_over = None
        keys_under = None
        per_model = {}
        for m in mods:
            err = wide[m] - wide.index.get_level_values("error_target_years")
            over = set(err[err > 2.0].index)
            under = set(err[err < -2.0].index)
            per_model[m] = {"n_super": len(over), "n_sub": len(under)}
            keys_over = over if keys_over is None else keys_over & over
            keys_under = under if keys_under is None else keys_under & under
        shared.append(
            {
                "posicao": pos,
                "n_pares_identificaveis_alinhados": int(len(wide)),
                "superestima_todos": int(len(keys_over or [])),
                "subestima_todos": int(len(keys_under or [])),
                "pct_super_compartilhada": fmt(
                    100 * len(keys_over or []) / max(per_model[champ]["n_super"], 1), 1
                ),
                "por_modelo": per_model,
                "exemplos_super": [
                    {
                        "player_name": idx[1],
                        "temporada": int(idx[2]),
                        "observado": fmt(idx[3], 2),
                        **{m: fmt(wide.loc[idx, m], 2) for m in mods},
                    }
                    for idx in list(keys_over or [])[:8]
                ],
                "exemplos_sub": [
                    {
                        "player_name": idx[1],
                        "temporada": int(idx[2]),
                        "observado": fmt(idx[3], 2),
                        **{m: fmt(wide.loc[idx, m], 2) for m in mods},
                    }
                    for idx in list(keys_under or [])[:6]
                ],
            }
        )
    return {
        "resumo": summary,
        "extremos": worst,
        "acertos": best,
        "compartilhados": shared,
        "acertos_horizonte": hits,
    }


def rank_flip_notes(wf: pd.DataFrame) -> list[str]:
    notes = []
    off = wf[(wf.oficial == 1) & (wf.modelo != "km")]
    # QB 2021 Cox > GBSA
    qb21 = off[(off.posicao == "QB") & (off.teste_em_temporada == 2021)]
    notes.append(
        f"QB 2021: Cox {qb21.loc[qb21.modelo=='cox','c_index'].iloc[0]:.3f} "
        f"vs GBSA {qb21.loc[qb21.modelo=='gbsa','c_index'].iloc[0]:.3f} (Cox ganha o fold; média ainda é GBSA)."
    )
    rb21 = off[(off.posicao == "RB") & (off.teste_em_temporada == 2021)]
    notes.append(
        f"RB 2021: RSF {rb21.loc[rb21.modelo=='rsf','c_index'].iloc[0]:.3f} "
        f"vs GBSA {rb21.loc[rb21.modelo=='gbsa','c_index'].iloc[0]:.3f} (RSF ganha o fold; média ainda é GBSA)."
    )
    wr = off[off.posicao == "WR"]
    rsf_wins = 0
    for s, g in wr.groupby("teste_em_temporada"):
        if g.loc[g.c_index.idxmax(), "modelo"] == "rsf":
            rsf_wins += 1
    notes.append(f"WR: RSF vence {rsf_wins}/4 folds oficiais.")
    return notes


def landmark_valido(lm: pd.DataFrame, followup: list[dict]) -> pd.DataFrame:
    """C medio do landmarking so nas celulas com acompanhamento completo.

    Sair em ate w anos e ter a ultima temporada ate t_L + w, o que so se confirma vendo
    t_L + w + 1. Celulas sem essa temporada na base ficam fora; registra quantos t_L sobram.
    """
    fu = pd.DataFrame(followup)[["posicao", "landmark_tL", "horizon_w", "max_followup_observado"]]
    d = lm.merge(fu, left_on=["posicao", "landmark", "horizon_w"], right_on=["posicao", "landmark_tL", "horizon_w"], how="left")
    # A saida ao fim de t_L + w so se confirma pela ausencia em t_L + w + 1 (tgb_protocol.landmark_completo).
    d["valida"] = [landmark_completo(t, w) for t, w in zip(d["landmark"], d["horizon_w"])]
    ok = d[d["valida"]]
    resumo = (
        ok.groupby(["posicao", "modelo", "horizon_w"])
        .agg(c_index=("c_index", "mean"), n_tL=("landmark", "nunique"), tL=("landmark", lambda x: ",".join(str(int(v)) for v in sorted(set(x)))))
        .reset_index()
    )
    celulas = d.drop_duplicates(["posicao", "landmark", "horizon_w"])
    resumo.attrs["n_celulas"] = int(len(celulas))
    resumo.attrs["n_invalidas"] = int((~celulas["valida"]).sum())
    return resumo


def main() -> None:
    wf = pd.read_csv(RES / "metricas_walkforward_folds.csv")
    lm = pd.read_csv(RES / "metricas_landmark_folds.csv")
    pred = pd.read_csv(RES / "predicao_2024_modelos.csv")
    oof = pd.read_csv(RES / "pred_oof_walkforward_2018_2021.csv")
    wf_sum = pd.read_csv(RES / "resumo_walkforward_2018_2021.csv")
    lm_sum = pd.read_csv(RES / "resumo_landmark.csv")

    inv = event_inventory()
    checks = protocol_checks(wf, lm, inv)
    rsf_diag = rsf_diagnostico(wf, inv)
    winners = fold_winners(wf)
    meta = {pos: raw_meta(pos) for pos in POSITIONS}
    followup = followup_max(meta)
    topk = topk_tables(pred, meta, k=10)
    names = named_sanity(pred)
    errors = oof_errors(oof)

    # years_exp no CSV 2024 está padronizado: flag explícita
    raw_qb = meta["QB"]
    csv_rod = pred[(pred.player_name == "Aaron Rodgers") & (pred.modelo == "cox")]
    raw_rod = raw_qb[(raw_qb.player_name == "Aaron Rodgers") & (raw_qb.snapshot_season == 2024)]
    scaled_flag = {
        "csv_years_exp_rodgers": fmt(float(csv_rod.years_exp.iloc[0]), 3) if len(csv_rod) else None,
        "raw_years_exp_rodgers": fmt(float(raw_rod.years_exp.iloc[0]), 3) if len(raw_rod) else None,
        "interpretacao": "O CSV predicao_2024_modelos.csv grava years_exp DEPOIS da escala do treino. Não ler como anos de carreira. Usar o painel cru.",
    }

    payload = {
        "inventario": inv,
        "protocolo": checks,
        "n_checks_ok": int(sum(c["ok"] for c in checks)),
        "n_checks": int(len(checks)),
        "rsf_diagnostico": rsf_diag,
        "followup_max": followup,
        "fold_winners": winners,
        "notas_fold": rank_flip_notes(wf),
        "oficial_c": wf_sum.to_dict(orient="records"),
        "landmark_c": lm_sum.to_dict(orient="records"),
        "topk": topk,
        "nomes_conhecidos": names,
        "erros_oof": errors,
        "years_exp_escala": scaled_flag,
        "como_ler": {
            "2024": "Sem rótulo. Concordância e face validity apenas. Residual=0 para quem está no último ano do painel.",
            "oof": "Erros em anos nos folds 2018-2021 somente se evento==1 ou follow-up>=tau; censuras anteriores a tau são excluídas do mesmo modo em todos os modelos.",
            "c_index": "Ordenação de pares, não calibração de anos. Spearman alto + MAE de anos alto podem coexistir.",
            "landmark": "Estimando diferente do walk-forward. C maior não é inconsistência.",
        },
    }
    (OUT / "auditoria.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pd.DataFrame(checks).to_csv(OUT / "protocolo_checks.csv", index=False)
    pd.DataFrame(errors["resumo"]).to_csv(OUT / "erros_oof_resumo.csv", index=False)
    pd.DataFrame(errors["acertos_horizonte"]).to_csv(OUT / "acerto_horizontes.csv", index=False)
    pd.DataFrame(errors["extremos"]).to_csv(OUT / "erros_oof_extremos.csv", index=False)
    pd.DataFrame(winners).to_csv(OUT / "fold_winners.csv", index=False)
    pd.DataFrame(rsf_diag).to_csv(OUT / "rsf_qb_diagnostico.csv", index=False)
    pd.DataFrame(followup).to_csv(OUT / "followup_max.csv", index=False)
    lm_valido = landmark_valido(lm, followup)
    lm_valido.to_csv(RES / "resumo_landmark_valido.csv", index=False)
    print(f"landmark: {lm_valido.attrs['n_invalidas']} de {lm_valido.attrs['n_celulas']} celulas sem follow-up para o horizonte")
    print(f"checks {payload['n_checks_ok']}/{payload['n_checks']}")
    print(pd.DataFrame(rsf_diag).to_string(index=False))
    print(pd.DataFrame(followup).to_string(index=False))
    print(json.dumps(scaled_flag, ensure_ascii=False, indent=2))
    print(pd.DataFrame(errors["resumo"]).to_string(index=False))
    print("overlap k=10")
    for pos, blk in topk["overlap"].items():
        print(pos, blk["10"])


if __name__ == "__main__":
    main()
