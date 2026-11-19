#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "run_all.py").exists() and (candidate / "DataBase").exists():
            return candidate
    raise RuntimeError("Project root not found")


ROOT = find_project_root()
os.chdir(ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

SCRIPT_PATH = Path(__file__).resolve()
CODIGO_PASSO = (
    '# ---- bloco 3 ----\n'
    'import pandas as pd\n'
    'import numpy as np\n'
    'import matplotlib.pyplot as plt\n'
    'from lifelines import KaplanMeierFitter, CoxPHFitter\n'
    'from lifelines.statistics import logrank_test, multivariate_logrank_test\n'
    'pass  # was: from lifelines.plotting import add_at_risk_counts\n'
    'from sklearn.preprocessing import StandardScaler\n'
    'import json, warnings\n'
    "warnings.filterwarnings('ignore')\n"
    "plt.rcParams['font.family'] = 'sans-serif'\n"
    '\n'
    'from pathlib import Path\n'
    '\n'
    'ROOT = Path.cwd().resolve()\n'
    'for candidate in [ROOT, ROOT.parent, ROOT.parent.parent, ROOT.parent.parent.parent, ROOT.parent.parent.parent.parent]:\n'
    "    if (candidate / 'Inferência' / 'Preparação Modelagem' / 'resultados').exists():\n"
    '        ROOT = candidate\n'
    '        break\n'
    '\n'
    "INFER_DIR = ROOT / 'Inferência'\n"
    "COX_DIR = INFER_DIR / 'Modelagem de COX'\n"
    "PREP_DIR = INFER_DIR / 'Preparação Modelagem'\n"
    "CURVES_DIR = COX_DIR / 'Curvas'\n"
    '\n'
    '\n'
    "POSICAO = 'WR'\n"
    "CORES = {'risco': '#D85A30', 'protetor': '#1D9E75', 'neutro': '#2E4057',\n"
    "         'destaque': '#F4A261', 'fundo': '#F8F9FA'}\n"
    '\n'
    "df_cox = pd.read_pickle(PREP_DIR / 'resultados' / f'df_cox_{POSICAO}.pkl')\n"
    '\n'
    "df_player = df_cox.groupby('player_id').agg(\n"
    "    duration=('stop', 'max'),\n"
    "    event=('event', 'max'),\n"
    "    entry=('start', 'min'),\n"
    "    age_entry=('age', 'first'),\n"
    "    draft_round=('draft_round', 'first'),\n"
    "    draft_ovr=('draft_ovr', 'first'),\n"
    "    height=('height', 'first'),\n"
    "    weight=('weight', 'first'),\n"
    ').reset_index()\n'
    '\n'
    'print(f\'{POSICAO}: {len(df_player)} jogadores | Eventos: {df_player["event"].sum()}\')\n'
    '\n'
    '# ---- bloco 5 ----\n'
    'kmf = KaplanMeierFitter()\n'
    "kmf.fit(df_player['duration'], event_observed=df_player['event'],\n"
    "        entry=df_player['entry'], label=f'Todos os {POSICAO}s')\n"
    '\n'
    '# Figura final: largura do texto e altura 0,165 da pagina (escala ~1 no LaTeX), fonte legivel.\n'
    'from thesis_figure_style import eixos_virgula, num_br\n'
    'fig, ax = plt.subplots(figsize=(6.2, 1.6))\n'
    "kmf.plot_survival_function(ax=ax, ci_show=False, color=CORES['neutro'], linewidth=1.5)\n"
    "ax.set_xlabel('Anos de experiência na NFL', fontsize=9, labelpad=1)\n"
    "ax.set_ylabel('Sobrevivência', fontsize=9, labelpad=2)\n"
    '\n'
    'median_s = kmf.median_survival_time_\n'
    "COR_MEDIANA = 'red'\n"
    'if not np.isnan(median_s) and not np.isinf(median_s):\n'
    "    ax.axhline(0.5, color=COR_MEDIANA, linestyle=':', linewidth=1.0)\n"
    "    ax.axvline(median_s, color=COR_MEDIANA, linestyle=':', linewidth=1.0, label=f'Mediana = {num_br(median_s, 1)} anos')\n"
    "ax.legend(fontsize=8.5, loc='upper right', frameon=False)\n"
    'ax.set_xlim(left=0); ax.set_ylim(0, 1)\n'
    "ax.tick_params(labelsize=8.5)\n"
    'ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); eixos_virgula(ax, x=False, y=True, casas_y=2)\n'
    '\n'
    'plt.tight_layout(pad=0.3)\n'
    "save_thesis_figure(fig, CURVES_DIR / 'figuras' / f'km_global_{POSICAO}.png', pad_inches=0.03)\n"
    'plt.close("all")\n'
    '\n'
    "print(f'Mediana de sobrevivência: {median_s:.2f} anos')\n"
    "print(f'Prob. sobreviver >5 anos: {kmf.predict(5):.3f}')\n"
    "print(f'Prob. sobreviver >10 anos: {kmf.predict(10):.3f}')\n"
    '\n'
    '# ---- bloco 7 ----\n'
    'fig, ax = plt.subplots(figsize=figsize_full(aspect=6/10))\n'
    '\n'
    'grupos_draft = {\n'
    "    'Rounds 1–2 (Elite)': df_player['draft_round'] <= 2,\n"
    "    'Rounds 3–5':         (df_player['draft_round'] > 2) & (df_player['draft_round'] <= 5),\n"
    "    'Rounds 6–7 / UDFA':  df_player['draft_round'] > 5,\n"
    '}\n'
    "cores_draft = [CORES['protetor'], CORES['destaque'], CORES['risco']]\n"
    '\n'
    'kmf_models = {}\n'
    'for (label, mask), cor in zip(grupos_draft.items(), cores_draft):\n'
    '    sub = df_player[mask]\n'
    '    if len(sub) < 2: continue\n'
    '    kmf_g = KaplanMeierFitter()\n'
    "    kmf_g.fit(sub['duration'], event_observed=sub['event'],\n"
    "              entry=sub['entry'], label=f'{label} (n={mask.sum()})')\n"
    '    kmf_g.plot_survival_function(ax=ax, ci_show=False, color=cor)\n'
    '    kmf_models[label] = kmf_g\n'
    '\n'
    'try:\n'
    "    draft_group = df_player['draft_round'].map(\n"
    "        lambda r: 'elite' if r<=2 else ('mid' if r<=5 else 'late'))\n"
    "    result = logrank_entrada_tardia(df_player['duration'], draft_group,\n"
    "                                    df_player['event'], entry=df_player['entry'])\n"
    "    p_title = f'\\nLog-rank p {format_p_value(result.p_value)}'\n"
    'except:\n'
    "    p_title = ''\n"
    '\n'
    "ax.set_title(f'KM por Round de Draft — {POSICAO}s{p_title}', fontsize=12, fontweight='bold')\n"
    "ax.set_xlabel('Anos de Carreira', fontsize=12)\n"
    "ax.set_ylabel('Prob. de Sobrevivência', fontsize=12)\n"
    'ax.legend(fontsize=11)\n'
    'try:\n'
    '    pass  # was: add_at_risk_counts(*kmf_models.values(), ax=ax, fontsize=8)\n'
    'except: pass\n'
    'plt.tight_layout()\n'
    "save_thesis_figure(fig, CURVES_DIR / 'figuras' / f'km_draft_{POSICAO}.png')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 9 ----\n'
    '# Seção removida na curadoria final do pipeline.\n'
    '\n'
    '# ---- bloco 11 ----\n'
    "df_player['faixa_etaria'] = pd.cut(df_player['age_entry'],\n"
    '                                    bins=[18, 22, 25, 40],\n'
    "                                    labels=['≤22 anos (jovem)', '23–25 anos', '>25 anos'])\n"
    '\n'
    'fig, ax = plt.subplots(figsize=figsize_full(aspect=6/10))\n'
    "cores_age = [CORES['protetor'], CORES['destaque'], CORES['risco']]\n"
    'kmf_models = {}\n'
    '\n'
    "for label, cor in zip(['≤22 anos (jovem)', '23–25 anos', '>25 anos'], cores_age):\n"
    "    sub = df_player[df_player['faixa_etaria'] == label].dropna(subset=['duration'])\n"
    '    if len(sub) < 2: continue\n'
    '    kmf_a = KaplanMeierFitter()\n'
    "    kmf_a.fit(sub['duration'], event_observed=sub['event'],\n"
    "              entry=sub['entry'], label=f'{label} (n={len(sub)})')\n"
    '    kmf_a.plot_survival_function(ax=ax, ci_show=False, color=cor)\n'
    '    kmf_models[label] = kmf_a\n'
    '\n'
    'try:\n'
    "    valid = df_player.dropna(subset=['faixa_etaria'])\n"
    "    result = logrank_entrada_tardia(valid['duration'], valid['faixa_etaria'],\n"
    "                                    valid['event'], entry=valid['entry'])\n"
    "    p_title = f'\\nLog-rank p {format_p_value(result.p_value)}'\n"
    'except:\n'
    "    p_title = ''\n"
    '\n'
    "ax.set_title(f'KM por Idade no Primeiro Ano Observado — {POSICAO}s{p_title}', fontsize=12, fontweight='bold')\n"
    "ax.set_xlabel('Anos de Carreira', fontsize=12)\n"
    "ax.set_ylabel('Prob. de Sobrevivência', fontsize=12)\n"
    'ax.legend(fontsize=11)\n'
    'try:\n'
    '    pass  # was: add_at_risk_counts(*kmf_models.values(), ax=ax, fontsize=8)\n'
    'except: pass\n'
    'plt.tight_layout()\n'
    "save_thesis_figure(fig, CURVES_DIR / 'figuras' / f'km_idade_{POSICAO}.png')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 13 ----\n'
    "depth_entry = df_cox.groupby('player_id')['depth_team'].first().reset_index()\n"
    "depth_entry.columns = ['player_id', 'depth_entry']\n"
    "df_p3 = df_player.merge(depth_entry, on='player_id', how='left').dropna(subset=['depth_entry'])\n"
    '\n'
    'grupos_depth = {\n'
    "    'Titular (depth=1)':  df_p3['depth_entry'] == 1,\n"
    "    'Reserva (depth=2)':  df_p3['depth_entry'] == 2,\n"
    "    'Profundo (depth≥3)': df_p3['depth_entry'] >= 3,\n"
    '}\n'
    "cores_depth = [CORES['protetor'], CORES['destaque'], CORES['risco']]\n"
    '\n'
    'fig, ax = plt.subplots(figsize=figsize_full(aspect=6/10))\n'
    'kmf_models = {}\n'
    '\n'
    'for (label, mask), cor in zip(grupos_depth.items(), cores_depth):\n'
    '    sub = df_p3[mask]\n'
    '    if len(sub) < 2: continue\n'
    '    kmf_d = KaplanMeierFitter()\n'
    "    kmf_d.fit(sub['duration'], event_observed=sub['event'],\n"
    "              entry=sub['entry'], label=f'{label} (n={mask.sum()})')\n"
    '    kmf_d.plot_survival_function(ax=ax, ci_show=False, color=cor)\n'
    '    kmf_models[label] = kmf_d\n'
    '\n'
    'try:\n'
    "    df_p3['depth_group'] = df_p3['depth_entry'].map(\n"
    "        lambda x: 'titular' if x==1 else ('reserva' if x==2 else 'profundo'))\n"
    "    result = logrank_entrada_tardia(df_p3['duration'], df_p3['depth_group'],\n"
    "                                    df_p3['event'], entry=df_p3['entry'])\n"
    "    p_title = f'\\nLog-rank p {format_p_value(result.p_value)}'\n"
    'except:\n'
    "    p_title = ''\n"
    '\n'
    "ax.set_title(f'KM por Profundidade no Elenco — {POSICAO}s{p_title}', fontsize=12, fontweight='bold')\n"
    "ax.set_xlabel('Anos de Carreira', fontsize=12)\n"
    "ax.set_ylabel('Prob. de Sobrevivência', fontsize=12)\n"
    'ax.legend(fontsize=11)\n'
    'try:\n'
    '    pass  # was: add_at_risk_counts(*kmf_models.values(), ax=ax, fontsize=8)\n'
    'except: pass\n'
    'plt.tight_layout()\n'
    "save_thesis_figure(fig, CURVES_DIR / 'figuras' / f'km_depth_{POSICAO}.png')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 15 ----\n'
    '# Seção removida na curadoria final do pipeline.\n'
)


def main() -> None:
    namespace = globals()
    namespace["__name__"] = "__passo__"
    namespace["__file__"] = str(SCRIPT_PATH)

    import sys as _sys
    _sys.path.insert(0, str(ROOT / "Predição/shared"))
    from thesis_figure_style import apply_thesis_rcparams, save_thesis_figure, figsize_full, format_p_value
    apply_thesis_rcparams()
    namespace["save_thesis_figure"] = save_thesis_figure
    namespace["figsize_full"] = figsize_full
    namespace["format_p_value"] = format_p_value
    # Log-rank com entrada tardia: o teste das curvas desenhadas com `entry=`.
    _sys.path.insert(0, str(ROOT / "Inferência/shared"))
    from survival_utils import logrank_entrada_tardia
    namespace["logrank_entrada_tardia"] = logrank_entrada_tardia
    exec(compile(CODIGO_PASSO, str(SCRIPT_PATH), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
