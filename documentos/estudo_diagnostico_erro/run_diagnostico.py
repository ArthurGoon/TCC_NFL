#!/usr/bin/env python3
"""Mapa onde/quando/como do erro preditivo e calibracao por decil.

Responde ao pedido de diagnostico do erro sem reajustar modelo nenhum: tudo no
out-of-fold do pacote oficial, com as restricoes declaradas.

- Quando: vies com sinal por ano do fold nos pares com min(T, tau) identificavel:
  evento observado ou follow-up pelo menos igual a tau.
- Onde: erro por decil de risco previsto; os percentis sao normalizados dentro
  do fold, pois o escore de cada fold tem escala propria, e os decis saem no
  OOF agregado por posicao.
- Como: calibracao de sobrevivencia a tres anos por decil (S(3) prevista media
  contra Kaplan-Meier do mesmo OOF). Usam-se os folds 2018-2020: no fold de
  2021 o follow-up maximo observado nao alcancava tres anos. A agregacao mistura
  folds de janelas proprias — a curva e diagnostico, nao estimador formal.
- Limiar "grave": a coincidencia de superestimacao grave entre familias,
  replicada no corte de 2 anos e medida tambem em 3.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "Predição" / "shared"))

from thesis_figure_style import apply_thesis_rcparams, eixos_virgula, figsize_full, save_thesis_figure, thesis_model_label  # noqa: E402
from tgb_protocol import (
    landmark_completo,
    modelos_destaque,
    restricted_time_identifiable,
    restricted_time_target,
    walkforward_split,
)  # noqa: E402

OFICIAL = ROOT / "documentos" / "estudo_modelos_finais" / "resultados"
DEST = ROOT / "Escrita Final" / "figuras"
HERE = Path(__file__).resolve().parent
RES = HERE / "resultados"

POS = ("QB", "RB", "WR")
# Modelo de maior C e modelo de contraste por posicao, lidos do resultado (sem lista fixa).
_DESTAQUE = modelos_destaque(pd.read_csv(OFICIAL / "resumo_walkforward_2018_2021.csv"))
CHAMPS = {pos: par[0] for pos, par in _DESTAQUE.items()}
CONTRASTE = {pos: par[1] for pos, par in _DESTAQUE.items()}
CORTES = (2.0, 3.0)
HORIZONTE = 3.0

NAVY = "#2E4057"
GREEN = "#1D9E75"
ORANGE = "#D85A30"


def carrega() -> pd.DataFrame:
    """Une o out-of-fold com a matriz de sobrevivencia e classifica os decis."""
    oof = pd.read_csv(OFICIAL / "pred_oof_walkforward_2018_2021.csv")
    mat = pd.read_csv(OFICIAL / "oof_surv_matrix.csv")
    grade = (
        pd.read_csv(OFICIAL / "oof_times_grid.csv")
        .drop_duplicates(["posicao", "teste_em_temporada", "j"])
    )
    colunas = {}
    for (pos, fold), g in grade.groupby(["posicao", "teste_em_temporada"]):
        col = g[g.t == HORIZONTE]
        colunas[(str(pos), int(fold))] = f"s{int(col.j.iloc[0])}" if len(col) else None

    def pega(i, r):
        col = colunas.get((str(r.posicao), int(r.teste_em_temporada)))
        return np.nan if col is None else float(mat.at[i, col])

    mat["s_h"] = [pega(i, r) for i, r in mat.iterrows()]
    chaves = ["posicao", "snapshot_season", "modelo", "player_id"]
    mat2 = mat.rename(columns={"teste_em_temporada": "snapshot_season"})
    d = oof.merge(
        mat2[chaves + ["tempo", "evento", "risk", "g_subject", "s_h"]],
        on=chaves, how="left", validate="one_to_one",
        suffixes=("", "_matrz"),
    ).rename(columns={"snapshot_season": "fold", "tempo": "T", "s_h": "s3"})
    # Os dois escores vem do mesmo ajuste; divergencia aqui sera bug de juncao.
    if not np.allclose(
        d["risk"].to_numpy(float),
        d.pop("risk_matrz").to_numpy(float),
        equal_nan=True,
        atol=1e-12,
    ):
        raise SystemExit("risk do OOF diverge do risk da matriz exportada")
    if d["risk"].isna().any():
        raise SystemExit("juncao OOF x matriz perdeu linhas")
    if "tau" not in d:
        d["tau"] = 2024 - pd.to_numeric(d["fold"], errors="coerce")
    calculated_flag = restricted_time_identifiable(d["evento"], d["T"], d["tau"])
    calculated_target = restricted_time_target(d["T"], d["tau"], calculated_flag)
    if "error_identifiable" in d and not np.array_equal(
        d["error_identifiable"].astype(bool).to_numpy(), calculated_flag
    ):
        raise SystemExit("flag de erro em anos diverge da regra evento==1 ou follow-up>=tau")
    if "error_target_years" in d and not np.allclose(
        pd.to_numeric(d["error_target_years"], errors="coerce").to_numpy(float),
        calculated_target,
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise SystemExit("alvo de erro em anos diverge do alvo restrito identificável")
    d["error_identifiable"] = calculated_flag
    d["error_target_years"] = calculated_target
    for (position, fold), group in d.groupby(["posicao", "fold"], sort=False):
        if set(group.modelo.astype(str)) != {"cox", "gbsa", "rsf"}:
            raise SystemExit(f"modelos incompletos para erro pareado em {position}/{fold}")
        eligible_ids = [
            set(part.loc[part.error_identifiable, "player_id"].astype(str))
            for _, part in group.groupby("modelo", sort=False)
        ]
        if eligible_ids and any(ids != eligible_ids[0] for ids in eligible_ids[1:]):
            raise SystemExit(f"máscara de erro não pareada entre modelos em {position}/{fold}")
    # Percentil dentro do fold: o escore de cada ajuste tem escala propria.
    d["percentil"] = d.groupby(["posicao", "modelo", "fold"])["risk"].rank(pct=True)
    # Os tres folds com horizonte valido entram na calibracao.
    d["tem_h3"] = [colunas.get((r.posicao, r.fold)) is not None for r in d.itertuples()]
    # Falta S(3) apenas no fold 2021, onde o horizonte nao existe na grade.
    if d[d.s3.isna() & d.tem_h3].any().any():
        raise SystemExit("s3 ausente em fold com horizonte validavel")
    # Decil POR POSICAO: o percentil ja e comparavel entre folds (e litros 0..1),
    # mas o corte em decil tem de respeitar o tamanho de cada posicao.
    # Decil para o mapa de erro; quintil para a calibracao de probabilidade:
    # com 20-38 indivíduos por decil, o Kaplan-Meier fica quantizado demais.
    d["decil"] = d.groupby("posicao")["percentil"].transform(
        lambda s: pd.qcut(s, 10, labels=False, duplicates="drop")
    ) + 1
    d["quintil"] = d.groupby("posicao")["percentil"].transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop")
    ) + 1
    return d


def erro_por_ano(d: pd.DataFrame) -> pd.DataFrame:
    """Vies em anos apenas nos alvos restritos identificaveis, com coorte pareada."""
    ev = d[d.error_identifiable].copy()
    linhas = []
    for pos in POS:
        for modelo in sorted(ev[ev.posicao == pos].modelo.unique()):
            sub = ev[(ev.posicao == pos) & (ev.modelo == modelo)]
            for fold, g in sub.groupby("fold"):
                err = g.expected_years - g["error_target_years"]
                total = d[(d.posicao == pos) & (d.modelo == modelo) & (d.fold == fold)]
                linhas.append(
                    {
                        "posicao": pos,
                        "modelo": modelo,
                        "fold": int(fold),
                        "n_pares": int(len(g)),
                        "n_pares_total": int(len(total)),
                        "n_excluidos_censura_pre_tau": int((~total.error_identifiable).sum()),
                        "vies_medio": float(err.mean()),
                        "mae": float(err.abs().mean()),
                        "pct_super": float(100 * (g.expected_years > g["error_target_years"]).mean()),
                        "pct_sub": float(100 * (g.expected_years < g["error_target_years"]).mean()),
                    }
                )
    return pd.DataFrame(linhas)


def erro_por_decil(d: pd.DataFrame) -> pd.DataFrame:
    """Erro por decil nos mesmos alvos identificaveis para todos os modelos."""
    ev = d[d.error_identifiable].copy()
    linhas = []
    for pos in POS:
        for modelo in sorted(ev[ev.posicao == pos].modelo.unique()):
            sub = ev[(ev.posicao == pos) & (ev.modelo == modelo)]
            for dec, g in sub.groupby("decil"):
                linhas.append(
                    {
                        "posicao": pos,
                        "modelo": modelo,
                        "decil": int(dec),
                        "n_pares": int(len(g)),
                        "n_excluidos_censura_pre_tau": int(
                            ((d.posicao == pos) & (d.modelo == modelo) & (d.decil == dec) & ~d.error_identifiable).sum()
                        ),
                        "previsto_medio": float(g.expected_years.mean()),
                        "T_medio": float(g["error_target_years"].mean()),
                        "vies_medio": float((g.expected_years - g["error_target_years"]).mean()),
                        "mae": float((g.expected_years - g["error_target_years"]).abs().mean()),
                    }
                )
    return pd.DataFrame(linhas)


def calibracao_quintil(d: pd.DataFrame) -> pd.DataFrame:
    """S(3) prevista media contra Kaplan-Meier no OOF agregado, por quintil."""
    base = d[d.tem_h3]
    linhas = []
    for pos in POS:
        for modelo in sorted(base[base.posicao == pos].modelo.unique()):
            sub = base[(base.posicao == pos) & (base.modelo == modelo)]
            for dec, g in sub.groupby("quintil"):
                km = KaplanMeierFitter()
                km.fit(g["T"].to_numpy(float), event_observed=g.evento.astype(bool).to_numpy())
                s_km = float(km.predict(HORIZONTE))
                linhas.append(
                    {
                        "posicao": pos,
                        "modelo": modelo,
                        "quintil": int(dec),
                        "n": int(len(g)),
                        "s3_prevista": float(g.s3.mean()),
                        "s3_km": s_km,
                        "dif": float(g.s3.mean() - s_km),
                    }
                )
    return pd.DataFrame(linhas)


def grave_por_corte(d: pd.DataFrame) -> pd.DataFrame:
    """Coincidencia de superestimacao grave na coorte identificavel pareada."""
    ev = d[d.error_identifiable].copy()
    linhas = []
    for pos in POS:
        champ = CHAMPS[pos]
        wide = ev[ev.posicao == pos].pivot_table(
            index=["player_id", "player_name", "fold", "error_target_years"],
            columns="modelo", values="expected_years",
        ).dropna()
        mods = list(wide.columns)
        for corte in CORTES:
            conj = {}
            for m in mods:
                err = wide[m] - wide.index.get_level_values("error_target_years")
                conj[m] = set(err[err > corte].index)
            inter = set.intersection(*conj.values()) if conj else set()
            linhas.append(
                {
                    "posicao": pos,
                    "corte": corte,
                    "n_grave_campeao": len(conj.get(champ, set())),
                    "n_grave_todos": len(inter),
                    "pct_compartilhada": float(
                        100 * len(inter) / max(len(conj.get(champ, set())), 1)
                    ),
                    "por_modelo_n_grave": json.dumps(
                        {m: len(v) for m, v in conj.items()}
                    ),
                }
            )
    return pd.DataFrame(linhas)


def figura(d: pd.DataFrame, calib: pd.DataFrame, rif: dict) -> None:
    """Calibracao por quintil: S(3) prevista contra Kaplan-Meier observado."""
    # Mesma escala de fonte da figura do walk-forward (Fig. 5.1): rotulos 10,8 pt, ticks 9,9 pt.
    apply_thesis_rcparams(font_scale=0.9)
    fig, axes = plt.subplots(1, 3, figsize=figsize_full(aspect=0.40), sharey=True)
    # So o modelo principal de cada posicao: cada modelo forma os proprios quintis, e
    # um Kaplan-Meier por modelo no mesmo painel confundia a leitura do observado.
    principal = CHAMPS
    for ax, pos in zip(axes, POS):
        sub = calib[calib.posicao == pos]
        for modelo in ("cox", "gbsa", "rsf"):
            g = sub[sub.modelo == modelo].sort_values("quintil")
            for _, r in g.iterrows():
                rif[f"{pos}|{modelo}|{int(r.quintil)}"] = {
                    "prevista": round(float(r.s3_prevista), 4),
                    "km": round(float(r.s3_km), 4),
                }
        g = sub[sub.modelo == principal[pos]].sort_values("quintil")
        ax.plot(g.quintil, g.s3_prevista, linestyle="-", linewidth=1.5, marker="o",
                markersize=3.4, color=NAVY, label="prevista")
        ax.plot(g.quintil, g.s3_km, linestyle="none", marker="o", markersize=4.2,
                markerfacecolor="white", markeredgecolor=ORANGE,
                label="observada (Kaplan–Meier do quintil)")
        ax.set_title(f"{pos} ({thesis_model_label(principal[pos])})", fontsize=11.5, pad=3)
        ax.set_xlim(0.5, 5.5)
        ax.set_xticks(range(1, 6))
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.45, color="0.65")
        eixos_virgula(ax, x=False, y=True, casas_y=1)
    axes[0].set_ylabel("S(3) em três anos")
    # Legenda unica embaixo e rotulo de eixo unico: dentro do painel a legenda
    # cobria as curvas, e repetido por painel o rotulo invadia o vizinho.
    handles, labels = axes[0].get_legend_handles_labels()
    # Legenda no topo e rotulo do eixo x embaixo: os dois no rodape colidiam, e
    # sem rotulo o leitor nao sabe o que sao os numeros 1 a 5.
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.89))
    # Rotulo do eixo x no painel central, colado aos ticks (o supxlabel ficava longe do eixo).
    axes[1].set_xlabel("Quintil de risco previsto", labelpad=2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    DEST.mkdir(parents=True, exist_ok=True)
    save_thesis_figure(fig, DEST / "tgb_calibracao_quintil.pdf", also_png=False)
    plt.close(fig)


def erro_km_referencia(d: pd.DataFrame) -> pd.DataFrame:
    """Erro em anos do Kaplan-Meier, que preve o mesmo tempo para todos (referencia sem covariaveis).

    Mesmo protocolo do pacote oficial: em cada fold, KM ajustado no treino com rotulo em
    tempo real, media restrita ate tau = 2024 - S integrada exatamente na escada do KM,
    comparada com o alvo restrito identificavel dos mesmos jogadores OOF.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ofi_diag", ROOT / "documentos" / "estudo_modelos_finais" / "run_oficial.py"
    )
    ofi = importlib.util.module_from_spec(spec)
    sys.modules["ofi_diag"] = ofi
    spec.loader.exec_module(ofi)
    linhas, erros = [], []
    for pos in POS:
        panel, _ = ofi.load_panel(pos)
        for S in (2018, 2019, 2020, 2021):
            tr, te = ofi.walkforward_split(panel, S)
            tau = float(2024 - S)
            oof_group = d[(d.posicao == pos) & (d.fold == S)]
            eligible_group = oof_group[oof_group.error_identifiable]
            eligible_ids = set(eligible_group.player_id.astype(str))
            total_ids = set(oof_group.player_id.astype(str))
            if not eligible_ids.issubset(set(te.player_id.astype(str))):
                raise SystemExit(f"coorte OOF elegivel ausente do split KM {pos}/{S}")
            te = te[te.player_id.astype(str).isin(eligible_ids)].copy()
            identifiable = restricted_time_identifiable(
                te.residual_event.to_numpy(), te.residual_duration_model.to_numpy(float), tau
            )
            if len(te) != int(identifiable.sum()):
                raise SystemExit(f"split KM inclui alvo não identificável em {pos}/{S}")
            km = KaplanMeierFitter().fit(tr["residual_duration_model"], tr["residual_event"])
            sf = km.survival_function_.iloc[:, 0]
            t = sf.index.to_numpy(dtype=float)
            dentro = t < tau
            pontos = np.r_[t[dentro], tau]
            media = float(np.sum(sf.to_numpy()[dentro] * np.diff(pontos)))
            obs = restricted_time_target(
                te.residual_duration_model.to_numpy(float), tau, identifiable
            )
            erro = media - obs
            erros.append(pd.DataFrame({"posicao": pos, "fold": S, "erro": erro}))
            linhas.append(
                {
                    "posicao": pos,
                    "fold": S,
                    "n_pares": int(len(te)),
                    "n_pares_total_oof": int(len(total_ids)),
                    "n_excluidos_censura_pre_tau": int(len(total_ids) - len(eligible_ids)),
                    "previsto_km": media,
                    "soma_abs": float(np.abs(erro).sum()),
                    "soma_vies": float(erro.sum()),
                    "n_acerto_1a": int((np.abs(erro) <= 1).sum()),
                    "n_acerto_2a": int((np.abs(erro) <= 2).sum()),
                    "n_acerto_3a": int((np.abs(erro) <= 3).sum()),
                    "vies_fold": float(erro.mean()),
                }
            )
    f = pd.DataFrame(linhas)
    f.to_csv(RES / "erro_km_por_fold.csv", index=False)
    g = f.groupby("posicao")[["n_pares", "soma_abs", "soma_vies", "n_acerto_1a", "n_acerto_2a", "n_acerto_3a"]].sum()
    todos = pd.concat(erros, ignore_index=True)
    mediana = todos.groupby("posicao")["erro"].apply(lambda e: float(np.median(np.abs(e))))
    out = pd.DataFrame(
        {
            "posicao": g.index,
            "modelo": "km",
            "n_pares": g["n_pares"].to_numpy(),
            "mae": (g["soma_abs"] / g["n_pares"]).to_numpy(),
            "mediana_ae": mediana.reindex(g.index).to_numpy(),
            "vies": (g["soma_vies"] / g["n_pares"]).to_numpy(),
            "acerto_1a": (100 * g["n_acerto_1a"] / g["n_pares"]).to_numpy(),
            "acerto_2a": (100 * g["n_acerto_2a"] / g["n_pares"]).to_numpy(),
            "acerto_3a": (100 * g["n_acerto_3a"] / g["n_pares"]).to_numpy(),
        }
    )
    out.to_csv(RES / "erro_km_referencia.csv", index=False)
    return out


def landmark_decomposicao() -> pd.DataFrame:
    """De onde vem o C mais alto do landmarking, nos mesmos jogadores de teste.

    O landmarking em t_L e o walk-forward do fold S = t_L usam o mesmo treino (linhas
    anteriores a t_L, desfecho em tempo real) e o mesmo teste (retratos da temporada t_L).
    Mudam o desfecho de avaliacao (cortado em w) e o alvo do ajuste (cortado em w). A tabela
    separa: (a) periodo, walk-forward dos folds de referencia contra os mesmos anos do
    landmarking; (b) pergunta, o risco do walk-forward avaliado no desfecho cortado em w;
    (c) ajuste, o modelo treinado para o horizonte w.
    """
    from sksurv.metrics import concordance_index_censored

    oof = pd.read_csv(OFICIAL / "pred_oof_walkforward_2018_2021.csv")
    lm = pd.read_csv(OFICIAL / "metricas_landmark_folds.csv")
    anos = (2020, 2021)
    # mesmas origens nos dois protocolos, so as com acompanhamento completo para cada w
    completo = [landmark_completo(t, w) for t, w in zip(lm.landmark, lm.horizon_w)]
    lm = lm[lm.landmark.isin(anos) & (lm.modelo != "km") & completo].groupby(["posicao", "modelo", "horizon_w"]).c_index.mean()
    linhas = []
    for (pos, mod), g in oof.groupby(["posicao", "modelo"]):
        ref = []
        for _, f in g.groupby("snapshot_season"):
            ref.append(concordance_index_censored(f.residual_event.astype(bool).to_numpy(),
                                                  f.residual_duration_model.to_numpy(float), f.risk.to_numpy(float))[0])
        for w in (1, 2, 3):
            mesmos, cortado = [], []
            for S in (a for a in anos if landmark_completo(a, w)):
                f = g[g.snapshot_season == S]
                t = f.residual_duration_model.to_numpy(float)
                e = f.residual_event.astype(bool).to_numpy()
                r = f.risk.to_numpy(float)
                mesmos.append(concordance_index_censored(e, t, r)[0])
                cortado.append(concordance_index_censored(e & (t <= w), np.minimum(t, w), r)[0])
            linhas.append({
                "posicao": pos, "modelo": mod, "horizon_w": w,
                "wf_referencia_2018_2021": float(np.mean(ref)),
                "wf_mesmos_anos": float(np.mean(mesmos)),
                "wf_desfecho_cortado_w": float(np.mean(cortado)),
                "landmark_mesmos_anos": float(lm.get((pos, mod, w), np.nan)),
            })
    out = pd.DataFrame(linhas)
    out.to_csv(RES / "landmark_vs_walkforward.csv", index=False)
    return out


def main() -> None:
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    d = carrega()
    exclusions = d.loc[
        ~d.error_identifiable,
        ["posicao", "fold", "player_id", "player_name", "T", "tau", "evento"],
    ].drop_duplicates(["posicao", "fold", "player_id"])
    exclusions["motivo"] = "censura_pre_tau"
    exclusions["n_modelos_excluidos"] = exclusions.apply(
        lambda row: int(
            ((d.posicao == row.posicao) & (d.fold == row.fold) & (d.player_id == row.player_id) & ~d.error_identifiable).sum()
        ),
        axis=1,
    )
    ano = erro_por_ano(d)
    dec = erro_por_decil(d)
    calib = calibracao_quintil(d)
    grave = grave_por_corte(d)
    km_ref = erro_km_referencia(d)
    lm_dec = landmark_decomposicao()
    print(lm_dec[lm_dec.horizon_w == 1].round(3).to_string(index=False), flush=True)
    print(km_ref.round(3).to_string(index=False), flush=True)

    ano.to_csv(RES / "erro_por_ano.csv", index=False)
    dec.to_csv(RES / "erro_por_decil.csv", index=False)
    exclusions.to_csv(RES / "exclusoes_erro_anos.csv", index=False)
    calib.to_csv(RES / "calibracao_quintil.csv", index=False)
    grave.to_csv(RES / "grave_por_corte.csv", index=False)

    pontos_figura: dict = {}
    figura(d, calib, pontos_figura)
    (RES / "pontos_figura_calibracao.json").write_text(
        json.dumps(pontos_figura, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    resumo = {
        "estimando": "erro com sinal = E(previsto) - min(T, tau), apenas quando evento==1 ou follow-up>=tau; mesma coorte identificavel para todos os modelos",
        "decil": "percentil de risco normalizado dentro do fold, cortado em decil no OOF agregado",
        "calibracao": (
            "S(3) prevista media contra Kaplan-Meier do mesmo OOF, folds 2018-2020 "
            "(no fold de 2021 o horizonte de tres anos nao existe na grade); "
            "quintil porque o decil deixa o KM quantizado"
        ),
        "grave": "superestimacao grave = previsto - observado > corte, mesmo par jogador e fold nos modelos",
        "km_referencia": "erro em anos do Kaplan-Meier (mesmo tempo para todos) no mesmo protocolo: erro_km_referencia.csv",
        "n_linhas_oof": int(len(d)),
        "n_coortes_excluidas": int(d.loc[~d.error_identifiable, ["posicao", "fold", "player_id"]].drop_duplicates().shape[0]),
        "n_pares_excluidos_modelo": int((~d.error_identifiable).sum()),
        "ids_excluidos": d.loc[~d.error_identifiable, ["posicao", "fold", "player_id", "player_name", "T", "tau"]].drop_duplicates().to_dict(orient="records"),
        "segundos": round(time.time() - t0, 1),
    }
    (RES / "resumo_diagnostico.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"linhas OOF: {len(d)}; excluídos de erro em anos: {int((~d.error_identifiable).sum())}; calibração em {d.tem_h3.sum()} linhas (3 folds)", flush=True)
    for pos in POS:
        champ = CHAMPS[pos]
        a = ano[(ano.posicao == pos) & (ano.modelo == champ)]
        print(f"\n{pos} {champ.upper()} — vies por fold:", flush=True)
        print(
            ", ".join(f"{int(r.fold)}: {r.vies_medio:+.2f}" for _, r in a.iterrows()),
            flush=True,
        )
        dch = dec[(dec.posicao == pos) & (dec.modelo == champ)]
        d1, d10 = dch[dch.decil == 1], dch[dch.decil == 10]
        if d1.empty or d10.empty:
            continue
        p1, p10 = d1.iloc[0], d10.iloc[0]
        print(
            f"  decil 1: prev {p1.previsto_medio:.2f} T {p1.T_medio:.2f} vies {p1.vies_medio:+.2f}"
            f" | decil 10: prev {p10.previsto_medio:.2f} T {p10.T_medio:.2f} vies {p10.vies_medio:+.2f}",
            flush=True,
        )
        gc = grave[(grave.posicao == pos) & (grave.corte == 2.0)].iloc[0]
        g3 = grave[(grave.posicao == pos) & (grave.corte == 3.0)].iloc[0]
        print(
            f"  grave corte2: {gc.pct_compartilhada:.1f}% (n={gc.n_grave_campeao}) | corte3: {g3.pct_compartilhada:.1f}% (n={g3.n_grave_campeao})",
            flush=True,
        )
    for pos in POS:
        champ = CHAMPS[pos]
        c = calib[(calib.posicao == pos) & (calib.modelo == champ)]
        print(
            f"{pos} {champ.upper()} calibracao: dif S(3)(q1) {c[c.quintil==1].dif.iloc[0]:+.3f} -> (q5) {c[c.quintil==5].dif.iloc[0]:+.3f}",
            flush=True,
        )
    print(f"fim {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
