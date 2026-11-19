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
    'import matplotlib.gridspec as gridspec\n'
    'import seaborn as sns\n'
    'from scipy import stats\n'
    'from lifelines import KaplanMeierFitter\n'
    'from lifelines.statistics import logrank_test\n'
    'from pathlib import Path\n'
    'import sys\n'
    '\n'
    "plt.rcParams['font.family'] = 'sans-serif'\n"
    "pd.set_option('display.max_columns', None)\n"
    '\n'
    'ROOT = Path.cwd().resolve()\n'
    'for candidate in [ROOT, ROOT.parent, ROOT.parent.parent, ROOT.parent.parent.parent, ROOT.parent.parent.parent.parent]:\n'
    "    if (candidate / 'Inferência' / 'shared' / 'survival_utils.py').exists():\n"
    '        ROOT = candidate\n'
    '        break\n'
    '\n'
    "INFER_DIR = ROOT / 'Inferência'\n"
    "COX_DIR = INFER_DIR / 'Modelagem de COX'\n"
    "PREP_DIR = INFER_DIR / 'Preparação Modelagem'\n"
    "AALEN_DIR = INFER_DIR / 'Modelagem de Aalen'\n"
    '\n'
    "sys.path.append(str((INFER_DIR / 'shared').resolve()))\n"
    '\n'
    '\n'
    '\n'
    'from survival_utils import load_clean_position_data  # noqa: E402\n'
    '\n'
    "POSICAO = 'RB'\n"
    'CORES = {\n'
    "    'risco': '#D85A30',\n"
    "    'protetor': '#1D9E75',\n"
    "    'neutro': '#2E4057',\n"
    "    'destaque': '#F4A261',\n"
    "    'fundo': '#F8F9FA'\n"
    '}\n'
    "print('Análise Exploratória de Dados — RBs na NFL')\n"
    '\n'
    'df = load_clean_position_data(ROOT, POSICAO)\n'
    "for col in ['years_exp', 'draft_round', 'age', 'height', 'weight', 'draft_ovr', 'depth_team', 'offense_pct', 'games_missed', 'season', 'games_played_season']:\n"
    '    if col in df.columns:\n'
    "        df[col] = pd.to_numeric(df[col], errors='coerce')\n"
    '\n'
    "df['draft_round'] = df['draft_round'].fillna(8)\n"
    "df['undrafted_flag'] = (df['draft_round'] == 8).astype(int)\n"
    "df.loc[df['undrafted_flag'] == 1, 'draft_ovr'] = 0\n"
    "df = df.sort_values(['player_id', 'season']).reset_index(drop=True)\n"
    "df['career_length'] = df.groupby('player_id')['season'].transform('nunique')\n"
    'print(f\'Jogadores: {df["player_id"].nunique()} | Registros jogador-temporada: {len(df)}\')\n'
    '\n'
    '# Nota: career_length conta apenas temporadas observadas (2012-2024).\n'
    '# Veteranos pre-2012 tem carreira subestimada. A analise de sobrevivencia\n'
    '# usa formato counting-process (start, stop, event) que modela o truncamento.\n'
    '\n'
    '# Carrega evento real do df_cox para uso no Kaplan-Meier (Cell 11).\n'
    "df_cox = pd.read_pickle(PREP_DIR / 'resultados' / f'df_cox_{POSICAO}.pkl')\n"
    "player_event = df_cox.groupby('player_id')['event'].max().reset_index()\n"
    '\n'
    '# ---- bloco 6 ----\n'
    "cl = df.groupby('player_id')['season'].nunique()\n"
    '\n'
    'fig, ax = plt.subplots(figsize=figsize_full(aspect=0.55))\n'
    '\n'
    "ax.hist(cl, bins=range(1, cl.max()+2), color=CORES['neutro'], alpha=0.85, edgecolor='white')\n"
    "ax.axvline(cl.median(), color=CORES['risco'], linestyle='--', linewidth=2, label=f'Mediana = {cl.median():.0f}')\n"
    "ax.axvline(cl.mean(), color=CORES['protetor'], linestyle='--', linewidth=2, label=f'Média = {cl.mean():.1f}')\n"
    "ax.axvline(cl.quantile(0.25), color='#888888', linestyle=':', linewidth=1.5, label=f'P25 = {cl.quantile(0.25):.0f}')\n"
    "ax.axvline(cl.quantile(0.75), color='#888888', linestyle=':', linewidth=1.5, label=f'P75 = {cl.quantile(0.75):.0f}')\n"
    'pct_le3 = (cl <= 3).mean() * 100\n'
    "ax.set_xlabel('Número de Temporadas', fontsize=12)\n"
    "ax.set_ylabel('Frequência', fontsize=12)\n"
    "ax.set_title(f'Distribuição do Tamanho de Carreira — {POSICAO}s', fontsize=12, fontweight='bold')\n"
    "ax.legend(fontsize=10, loc='upper left')\n"
    "ax.text(0.98, 0.98, f'{pct_le3:.0f}% das carreiras\\ntêm ≤ 3 temporadas',\n"
    "             transform=ax.transAxes, ha='right', va='top', fontsize=11,\n"
    "             bbox=dict(boxstyle='round', facecolor=CORES['fundo'], alpha=0.8))\n"
    '\n'
    'plt.tight_layout()\n'
    "save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/dist_carreira_{POSICAO}.png')\n"
    'plt.close("all")\n'
    '\n'
    'print(f"Média: {cl.mean():.2f} | Mediana: {cl.median():.1f} | DP: {cl.std():.2f}")\n'
    'print(f"P25: {cl.quantile(0.25):.0f} | P50: {cl.quantile(0.50):.0f} | P75: {cl.quantile(0.75):.0f}")\n'
    'print(f"P90: {cl.quantile(0.90):.0f} | P95: {cl.quantile(0.95):.0f}")\n'
    'print(f"% carreiras ≤ 3 temporadas: {pct_le3:.1f}%")\n'
    '\n'
    '# ---- bloco 8 ----\n'
    "main_vars = ['years_exp', 'age', 'height', 'weight', 'draft_ovr',\n"
    "             'offense_pct', 'depth_team', 'games_missed', 'fantasy_points_ppr']\n"
    "main_vars += ['season_rushing_yards', 'season_rush_attempts', 'season_rush_touchdown', 'season_receptions', 'season_receiving_yards']\n"
    'main_vars = [v for v in main_vars if v in df.columns]\n'
    '\n'
    'desc = df[main_vars].describe().T\n'
    "desc['missing_%'] = df[main_vars].isna().mean() * 100\n"
    'print(desc.round(2).to_string())\n'
    '\n'
    '# ---- bloco 10 ----\n'
    'fig, axes = plt.subplots(2, 1, figsize=figsize_full(aspect=0.85))\n'
    '\n'
    "players_per_season = df.groupby('season')['player_id'].nunique()\n"
    "axes[0].bar(players_per_season.index, players_per_season.values, color=CORES['neutro'], alpha=0.85)\n"
    "axes[0].set_xlabel('Temporada', fontsize=12)\n"
    "axes[0].set_ylabel('Número de Jogadores', fontsize=12)\n"
    "axes[0].set_title(f'Jogadores Ativos por Temporada — {POSICAO}s', fontsize=12, fontweight='bold')\n"
    "axes[0].tick_params(axis='x', rotation=45)\n"
    '\n'
    "avg_exp = df.groupby('season')['years_exp'].mean()\n"
    "axes[1].plot(avg_exp.index, avg_exp.values, color=CORES['protetor'], linewidth=2, marker='o', markersize=4)\n"
    "axes[1].set_xlabel('Temporada', fontsize=12)\n"
    "axes[1].set_ylabel('Média de Anos de Experiência', fontsize=12)\n"
    "axes[1].set_title(f'Experiência Média por Temporada — {POSICAO}s', fontsize=12, fontweight='bold')\n"
    "axes[1].tick_params(axis='x', rotation=45)\n"
    '\n'
    'plt.tight_layout()\n'
    "save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/temporal_{POSICAO}.png')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 12 ----\n'
    '# 2.3 Curva de Sobrevivencia Kaplan-Meier por Round de Draft — RB\n'
    '# ---------------------------------------------------------------\n'
    "surv_data = df.groupby('player_id').agg(\n"
    "    duration=('career_length', 'max'),\n"
    "    draft_round=('draft_round', 'first'),\n"
    "    player_name=('player_name', 'first')\n"
    ').reset_index()\n'
    '\n'
    '# Merge com evento real do df_cox (eventos reais, ~40% censura).\n'
    "surv_data = surv_data.merge(player_event, on='player_id', how='inner')\n"
    '\n'
    'fig, axes = plt.subplots(2, 1, figsize=figsize_full(aspect=0.9))\n'
    '\n'
    '# Painel esquerdo: 8 grupos de draft\n'
    'kmf = KaplanMeierFitter()\n'
    "draft_labels = {1:'R1', 2:'R2', 3:'R3', 4:'R4', 5:'R5', 6:'R6', 7:'R7', 8:'UDFA'}\n"
    'colors = plt.cm.tab10(np.linspace(0, 1, 8))\n'
    '\n'
    'for rd, color in zip(range(1, 9), colors):\n'
    "    mask = surv_data['draft_round'] == rd\n"
    '    if mask.sum() >= 5:\n'
    "        kmf.fit(surv_data.loc[mask, 'duration'], \n"
    "                event_observed=surv_data.loc[mask, 'event'].values,\n"
    '                label=draft_labels[rd])\n'
    '        kmf.plot_survival_function(ax=axes[0], color=color, linewidth=1.5)\n'
    '\n'
    "axes[0].set_title('Kaplan-Meier por Round de Draft - RB', fontsize=12)\n"
    "axes[0].set_xlabel('Temporadas observadas (2012-2024)')\n"
    "axes[0].set_ylabel('Probabilidade de permanencia')\n"
    "axes[0].legend(title='Draft Round', fontsize=8, title_fontsize=9)\n"
    'axes[0].set_xlim(0, 14)\n'
    'axes[0].grid(alpha=0.3)\n'
    '\n'
    '# Log-rank R1 vs UDFA\n'
    "r1_mask = surv_data['draft_round'] == 1\n"
    "udfa_mask = surv_data['draft_round'] == 8\n"
    'if r1_mask.sum() >= 5 and udfa_mask.sum() >= 5:\n'
    '    from lifelines.statistics import logrank_test\n'
    '    lr = logrank_test(\n'
    "        surv_data.loc[r1_mask, 'duration'],\n"
    "        surv_data.loc[udfa_mask, 'duration'],\n"
    "        event_observed_A=surv_data.loc[r1_mask, 'event'].values,\n"
    "        event_observed_B=surv_data.loc[udfa_mask, 'event'].values\n"
    '    )\n'
    "    axes[0].text(0.95, 0.15, f'Log-rank R1 vs UDFA\\np {format_p_value(lr.p_value)} (expl.)',\n"
    "                 transform=axes[0].transAxes, ha='right', fontsize=9,\n"
    "                 bbox=dict(boxstyle='round', facecolor=CORES['fundo'], alpha=0.8))\n"
    '\n'
    '# Painel direito: 3 categorias agregadas\n'
    "surv_data['draft_tier'] = surv_data['draft_round'].apply(\n"
    "    lambda x: 'Early (R1-R2)' if x <= 2 else 'Mid (R3-R5)' if x <= 5 else 'Late (R6-UDFA)'\n"
    ')\n'
    '\n'
    "tier_colors = {'Early (R1-R2)': '#D85A30', 'Mid (R3-R5)': '#2E4057', 'Late (R6-UDFA)': '#1D9E75'}\n"
    '\n'
    "for tier in ['Early (R1-R2)', 'Mid (R3-R5)', 'Late (R6-UDFA)']:\n"
    "    mask = surv_data['draft_tier'] == tier\n"
    '    if mask.sum() >= 5:\n'
    "        kmf.fit(surv_data.loc[mask, 'duration'],\n"
    "                event_observed=surv_data.loc[mask, 'event'].values,\n"
    '                label=tier)\n'
    '        kmf.plot_survival_function(ax=axes[1], color=tier_colors[tier], linewidth=2)\n'
    '\n'
    "axes[1].set_title('KM por Tier de Draft (agregado) - RB', fontsize=12)\n"
    "axes[1].set_xlabel('Temporadas observadas (2012-2024)')\n"
    "axes[1].set_ylabel('Probabilidade de permanencia')\n"
    'axes[1].legend(fontsize=9)\n'
    'axes[1].set_xlim(0, 14)\n'
    'axes[1].grid(alpha=0.3)\n'
    '\n'
    'plt.tight_layout()\n'
    "FIG_DIR = Path(ROOT / 'Análise Exploratória de Dados/figuras')\n"
    "save_thesis_figure(fig, FIG_DIR / 'km_draft_RB.png')\n"
    'plt.close("all")\n'
    "print('Curvas KM geradas: km_draft_RB.png')\n"
    '\n'
    '# --- KM geral (nao estratificada) para referencia ---\n'
    'fig, ax = plt.subplots(figsize=figsize_full(aspect=5/8))\n'
    'kmf = KaplanMeierFitter()\n'
    "kmf.fit(surv_data['duration'], event_observed=surv_data['event'].values, label='RB')\n"
    "kmf.plot_survival_function(ax=ax, color=CORES['risco'], linewidth=2.5)\n"
    'median_surv = kmf.median_survival_time_\n'
    "ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)\n"
    "if median_surv != float('inf'):\n"
    "    ax.axvline(median_surv, color='gray', linestyle='--', alpha=0.5)\n"
    "    ax.text(median_surv + 0.3, 0.52, f'Mediana: {median_surv:.1f} temp.', fontsize=9)\n"
    "ax.set_title('Curva de Sobrevivencia Geral - RB', fontsize=12)\n"
    "ax.set_xlabel('Temporadas observadas (2012-2024)')\n"
    "ax.set_ylabel('Probabilidade de permanencia')\n"
    'ax.set_xlim(0, 14)\n'
    'ax.legend(fontsize=11)\n'
    'ax.grid(alpha=0.3)\n'
    'plt.tight_layout()\n'
    "save_thesis_figure(fig, FIG_DIR / 'km_overall_RB.png')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 15 ----\n'
    'fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n'
    '\n'
    '# Height vs career_length\n'
    "if 'height' in df.columns:\n"
    "    cl_player = df.drop_duplicates('player_id').copy()\n"
    "    axes[0].scatter(cl_player['height'], cl_player['career_length'], alpha=0.6, color=CORES['neutro'])\n"
    "    z = np.polyfit(cl_player['height'].dropna(), cl_player.loc[cl_player['height'].notna(), 'career_length'], 1)\n"
    '    p = np.poly1d(z)\n'
    "    axes[0].plot(cl_player['height'].sort_values(), p(cl_player['height'].sort_values()), \n"
    "                 color=CORES['risco'], linestyle='--', linewidth=2)\n"
    "    axes[0].set_xlabel('Altura (polegadas)', fontsize=12)\n"
    "    axes[0].set_ylabel('Temporadas na NFL', fontsize=12)\n"
    "    axes[0].set_title('Altura vs Longevidade', fontsize=12, fontweight='bold')\n"
    '\n'
    '# Weight vs career_length\n'
    "if 'weight' in df.columns:\n"
    "    cl_player = df.drop_duplicates('player_id').copy()\n"
    "    axes[1].scatter(cl_player['weight'], cl_player['career_length'], alpha=0.6, color=CORES['neutro'])\n"
    "    z = np.polyfit(cl_player['weight'].dropna(), cl_player.loc[cl_player['weight'].notna(), 'career_length'], 1)\n"
    '    p = np.poly1d(z)\n'
    "    axes[1].plot(cl_player['weight'].sort_values(), p(cl_player['weight'].sort_values()), \n"
    "                 color=CORES['risco'], linestyle='--', linewidth=2)\n"
    "    axes[1].set_xlabel('Peso (lbs)', fontsize=12)\n"
    "    axes[1].set_ylabel('Temporadas na NFL', fontsize=12)\n"
    "    axes[1].set_title('Peso vs Longevidade', fontsize=12, fontweight='bold')\n"
    '\n'
    '# Age at debut\n'
    "if 'age' in df.columns and 'years_exp' in df.columns:\n"
    "    debut_age = df[df['years_exp'] == 1]['age'].dropna()\n"
    "    axes[2].hist(debut_age, bins=15, color=CORES['neutro'], alpha=0.85, edgecolor='white')\n"
    "    axes[2].axvline(debut_age.mean(), color=CORES['risco'], linestyle='--', linewidth=2, label=f'Média = {debut_age.mean():.1f}')\n"
    "    axes[2].set_xlabel('Idade no Rookie Year', fontsize=12)\n"
    "    axes[2].set_ylabel('Frequência', fontsize=12)\n"
    "    axes[2].set_title('Idade de Estreia', fontsize=12, fontweight='bold')\n"
    '    axes[2].legend(fontsize=11)\n'
    '\n'
    'plt.tight_layout()\n'
    "save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/perfil_fisico_{POSICAO}.png')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 17 ----\n'
    "cl_player = df.drop_duplicates('player_id').copy()\n"
    "cl_player['draft_round'] = cl_player['draft_round'].fillna(8)\n"
    "cl_player['draft_group'] = cl_player['draft_round'].apply(\n"
    "    lambda x: f'Rodada {int(x)}' if x <= 7 else 'Não draftado')\n"
    '\n'
    "# Figura final: largura do texto e altura 0,165 da pagina (escala ~1 no LaTeX); a caixa de Kruskal-Wallis sai da figura (teste no texto e em fig_testes.csv).\n"
    'from thesis_figure_style import eixos_virgula\n'
    "fig, axes = plt.subplots(1, 2, figsize=(6.2, 1.6), gridspec_kw={'width_ratios': [1, 1]})\n"
    '\n'
    "draft_means = cl_player.groupby('draft_group')['career_length'].mean().sort_index(key=lambda s: s.map(lambda g: 99 if g.startswith('Não') else int(g.split()[-1])))\n"
    "colors_bar = [CORES['protetor'] if 'Rodada 1' in x or 'Rodada 2' in x \n"
    "              else CORES['destaque'] if 'Rodada 3' in x or 'Rodada 4' in x or 'Rodada 5' in x\n"
    "              else CORES['risco'] for x in draft_means.index]\n"
    "rot_barras = [x.replace('Rodada ', '') if x.startswith('Rodada') else 'Não\\ndraft.' for x in draft_means.index]\n"
    'axes[0].bar(rot_barras, draft_means.values, color=colors_bar, alpha=0.85)\n'
    "axes[0].set_xlabel('Rodada do draft', fontsize=8.5, labelpad=1)\n"
    "axes[0].set_ylabel('Média de\\ntemporadas', fontsize=8.5, labelpad=2)\n"
    "axes[0].tick_params(axis='both', labelsize=8)\n"
    '\n'
    "cl_player['longevidade_grupo'] = pd.cut(cl_player['career_length'],\n"
    "    bins=[0, 3, 7, 12, 99], labels=['≤3', '4–7', '8–12', '>12'])\n"
    "bp_data = [cl_player[(cl_player['longevidade_grupo']==g) & (cl_player['draft_ovr'] > 0)]['draft_ovr'].dropna().values \n"
    "           for g in ['≤3', '4–7', '8–12', '>12']]\n"
    'bp_data = [d for d in bp_data if len(d) > 0]\n'
    "bp_labels = [l for l, d in zip(['≤3', '4–7', '8–12', '>12'], \n"
    "             [cl_player[(cl_player['longevidade_grupo']==g) & (cl_player['draft_ovr'] > 0)]['draft_ovr'].dropna().values \n"
    "              for g in ['≤3', '4–7', '8–12', '>12']]) if len(d) > 0]\n"
    'bp = axes[1].boxplot(bp_data, tick_labels=bp_labels, patch_artist=True)\n'
    "for patch, color in zip(bp['boxes'], [CORES['risco'], CORES['destaque'], CORES['protetor'], CORES['neutro']][:len(bp_data)]):\n"
    '    patch.set_facecolor(color)\n'
    '    patch.set_alpha(0.7)\n'
    "axes[1].set_xlabel('Longevidade (temporadas)', fontsize=8.5, labelpad=1)\n"
    "axes[1].set_ylabel('Escolha geral\\nno draft', fontsize=8.5, labelpad=2)\n"
    "axes[1].tick_params(axis='both', labelsize=8)\n"
    'eixos_virgula(axes[0], x=False, y=True)\n'
    'eixos_virgula(axes[1], x=False, y=True)\n'
    'fig.subplots_adjust(left=0.09, right=0.995, top=0.97, bottom=0.27, wspace=0.26)\n'
    "save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/draft_{POSICAO}.png', pad_inches=0.03)\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 19 ----\n'
    'fig, axes = plt.subplots(2, 1, figsize=figsize_full(aspect=0.85))\n'
    '\n'
    "age_by_exp = df.groupby('years_exp')['age'].agg(['median','mean','count'])\n"
    "age_by_exp = age_by_exp[age_by_exp['count'] >= 5]\n"
    "axes[0].plot(age_by_exp.index, age_by_exp['median'], color=CORES['neutro'], linewidth=2, marker='o', markersize=5)\n"
    'axes[0].fill_between(age_by_exp.index,\n'
    "                     df.groupby('years_exp')['age'].quantile(0.25).reindex(age_by_exp.index),\n"
    "                     df.groupby('years_exp')['age'].quantile(0.75).reindex(age_by_exp.index),\n"
    "                     alpha=0.2, color=CORES['neutro'])\n"
    "axes[0].set_xlabel('Anos de Experiência', fontsize=12)\n"
    "axes[0].set_ylabel('Idade (mediana)', fontsize=12)\n"
    "axes[0].set_title(f'Trajetória Etária — {POSICAO}s', fontsize=12, fontweight='bold')\n"
    '\n'
    'try:\n'
    "    pivot = df.groupby(['season', 'age']).size().unstack(fill_value=0)\n"
    "    sns.heatmap(pivot.T, cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Nº Jogadores'})\n"
    "    axes[1].set_xlabel('Temporada', fontsize=12)\n"
    "    axes[1].set_ylabel('Idade', fontsize=12)\n"
    "    axes[1].set_title(f'Densidade Temporada × Idade — {POSICAO}s', fontsize=12, fontweight='bold')\n"
    'except Exception as e:\n'
    "    axes[1].text(0.5, 0.5, f'Erro no heatmap: {e}', ha='center', va='center')\n"
    '\n'
    'plt.tight_layout()\n'
    "save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/idade_{POSICAO}.png')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 21 ----\n'
    "main_lags = ['career_rushing_yards', 'career_rush_touchdown', 'career_rush_attempts', 'career_receiving_yards']\n"
    "print(f'Principais career_lags para RB:')\n"
    'for v in main_lags:\n'
    '    if v in df.columns:\n'
    "        print(f'  {v}: média={df[v].mean():.1f}, mediana={df[v].median():.1f}')\n"
    '\n'
    '# ---- bloco 23 ----\n'
    'fig, axes = plt.subplots(4, 1, figsize=figsize_full(aspect=1.35))\n'
    'axes = axes.flatten()\n'
    '\n'
    "main_lags = ['career_rushing_yards', 'career_rush_touchdown', 'career_rush_attempts', 'career_receiving_yards']\n"
    'for idx, var in enumerate(main_lags):\n'
    '    if var in df.columns:\n'
    "        axes[idx].hist(df[var].dropna(), bins=30, color=CORES['neutro'], alpha=0.85, edgecolor='white')\n"
    "        axes[idx].axvline(df[var].median(), color=CORES['risco'], linestyle='--', linewidth=2, label=f'Mediana = {df[var].median():.1f}')\n"
    "        axes[idx].set_xlabel(var.replace('career_', '').replace('_', ' ').title(), fontsize=11)\n"
    "        axes[idx].set_ylabel('Frequência', fontsize=11)\n"
    "        axes[idx].set_title(f'Distribuição: {var}', fontsize=12, fontweight='bold')\n"
    '        axes[idx].legend(fontsize=9)\n'
    '\n'
    'plt.tight_layout()\n'
    "save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/career_lag_dist_RB.png')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 25 ----\n'
    "cl_player = df.drop_duplicates('player_id').copy()\n"
    "main_lags = ['career_rushing_yards', 'career_rush_touchdown', 'career_rush_attempts', 'career_receiving_yards']\n"
    '\n'
    'fig, axes = plt.subplots(1, len(main_lags), figsize=(16, 4))\n'
    'if len(main_lags) == 1:\n'
    '    axes = [axes]\n'
    '\n'
    'for idx, var in enumerate(main_lags):\n'
    '    if var in df.columns:\n'
    "        cl_player = df.drop_duplicates('player_id').copy()\n"
    "        axes[idx].scatter(cl_player[var], cl_player['career_length'], alpha=0.5, color=CORES['neutro'])\n"
    "        corr = cl_player[[var, 'career_length']].corr().iloc[0, 1]\n"
    "        axes[idx].set_xlabel(var.replace('career_', '').replace('_', ' ').title(), fontsize=11)\n"
    "        axes[idx].set_ylabel('Temporadas na NFL', fontsize=11)\n"
    "        axes[idx].set_title(f'r = {corr:.3f}', fontsize=12, fontweight='bold')\n"
    '\n'
    'plt.tight_layout()\n'
    "save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/career_lag_corr_RB.png')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 27 ----\n'
    'fig, axes = plt.subplots(4, 1, figsize=figsize_full(aspect=1.35))\n'
    'axes = axes.flatten()\n'
    '\n'
    "main_lags = ['career_rushing_yards', 'career_rush_touchdown', 'career_rush_attempts', 'career_receiving_yards']\n"
    'for idx, var in enumerate(main_lags):\n'
    '    if var in df.columns:\n'
    "        evo = df.groupby('years_exp')[var].agg(['mean', 'median', 'std']).reset_index()\n"
    "        axes[idx].plot(evo['years_exp'], evo['mean'], marker='o', color=CORES['neutro'], linewidth=2, label='Média')\n"
    "        axes[idx].fill_between(evo['years_exp'], evo['mean'] - evo['std'], evo['mean'] + evo['std'], \n"
    "                               alpha=0.3, color=CORES['neutro'])\n"
    "        axes[idx].set_xlabel('Anos de Experiência', fontsize=11)\n"
    "        axes[idx].set_ylabel(var.replace('career_', '').replace('_', ' ').title(), fontsize=11)\n"
    "        axes[idx].set_title(f'Evolução: {var}', fontsize=12, fontweight='bold')\n"
    '        axes[idx].legend(fontsize=9)\n'
    '\n'
    'plt.tight_layout()\n'
    "save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/career_lag_evo_RB.png')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 29 ----\n'
    '# Taxas de eficiência acumulada\n'
    'fig, axes = plt.subplots(2, 1, figsize=figsize_full(aspect=0.85))\n'
    '\n'
    "if 'career_rushing_yards' in df.columns and 'career_rush_attempts' in df.columns:\n"
    "    df['ypc'] = df['career_rushing_yards'] / df['career_rush_attempts'].replace(0, np.nan)\n"
    "    ypc_by_exp = df.groupby('years_exp')['ypc'].mean()\n"
    "    axes[0].plot(ypc_by_exp.index, ypc_by_exp.values, marker='o', color=CORES['neutro'], linewidth=2)\n"
    "    axes[0].set_xlabel('Anos de Experiência', fontsize=12)\n"
    "    axes[0].set_ylabel('Yards/Carry', fontsize=12)\n"
    "    axes[0].set_title('Eficiência de Corrida (YPC)', fontsize=12, fontweight='bold')\n"
    '\n'
    "if 'career_rush_touchdown' in df.columns and 'career_rush_attempts' in df.columns:\n"
    "    df['td_pct'] = df['career_rush_touchdown'] / df['career_rush_attempts'].replace(0, np.nan) * 100\n"
    "    td_by_exp = df.groupby('years_exp')['td_pct'].mean()\n"
    "    axes[1].plot(td_by_exp.index, td_by_exp.values, marker='o', color=CORES['neutro'], linewidth=2)\n"
    "    axes[1].set_xlabel('Anos de Experiência', fontsize=12)\n"
    "    axes[1].set_ylabel('TD %', fontsize=12)\n"
    "    axes[1].set_title('Taxa de Touchdown', fontsize=12, fontweight='bold')\n"
    '\n'
    'plt.tight_layout()\n'
    "save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/eficiencia_acumulada_{POSICAO}.png')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 32 ----\n'
    '# Declínio sazonal: como as métricas por jogo evoluem com years_exp\n'
    'fig, axes = plt.subplots(2, 1, figsize=figsize_full(aspect=0.85))\n'
    '\n'
    "season_vars = ['rushing_yards', 'rush_touchdown']\n"
    '\n'
    'for idx, var in enumerate(season_vars):\n'
    '    if var in df.columns:\n'
    "        decl = df.groupby('years_exp')[var].agg(['mean', 'median']).reset_index()\n"
    "        axes[idx].plot(decl['years_exp'], decl['mean'], marker='o', color=CORES['neutro'], linewidth=2, label='Média')\n"
    "        axes[idx].plot(decl['years_exp'], decl['median'], marker='s', color=CORES['destaque'], linewidth=2, linestyle='--', label='Mediana')\n"
    "        axes[idx].set_xlabel('Anos de Experiência', fontsize=12)\n"
    "        axes[idx].set_ylabel(var.replace('_', ' ').title(), fontsize=12)\n"
    "        axes[idx].set_title(f'Declínio: {var}', fontsize=12, fontweight='bold')\n"
    '        axes[idx].legend(fontsize=11)\n'
    '\n'
    'plt.tight_layout()\n'
    "save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/declinio_sazonal_{POSICAO}.png')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 34 ----\n'
    '# Análise das variáveis delta (season-over-season change)\n'
    'fig, axes = plt.subplots(2, 1, figsize=figsize_full(aspect=0.85))\n'
    '\n'
    "delta_vars = ['delta_rushing_yards', 'delta_rush_touchdown']\n"
    '\n'
    'for idx, var in enumerate(delta_vars):\n'
    '    if var in df.columns:\n'
    "        axes[idx].hist(df[var].dropna(), bins=30, color=CORES['neutro'], alpha=0.85, edgecolor='white')\n"
    "        axes[idx].axvline(0, color=CORES['risco'], linestyle='--', linewidth=2, label='Sem mudança')\n"
    "        axes[idx].axvline(df[var].median(), color=CORES['destaque'], linestyle='--', linewidth=2, label=f'Mediana = {df[var].median():.1f}')\n"
    "        axes[idx].set_xlabel(var.replace('delta_', '').replace('_', ' ').title(), fontsize=12)\n"
    "        axes[idx].set_ylabel('Frequência', fontsize=12)\n"
    "        axes[idx].set_title(f'Δ Season-over-Season: {var}', fontsize=12, fontweight='bold')\n"
    '        axes[idx].legend(fontsize=11)\n'
    '\n'
    'plt.tight_layout()\n'
    "save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/delta_season_{POSICAO}.png')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 36 ----\n'
    "if 'games_missed' in df.columns:\n"
    '    fig, axes = plt.subplots(1, 2, figsize=figsize_dual_horizontal(), gridspec_kw={\'width_ratios\': [1, 1]})\n'
    '\n'
    "    gm_by_exp = df.groupby('years_exp')['games_missed'].median()\n"
    "    gm_by_exp = gm_by_exp[df.groupby('years_exp').size() >= 5]\n"
    "    axes[0].plot(gm_by_exp.index, gm_by_exp.values, color=CORES['risco'], linewidth=2, marker='o')\n"
    "    axes[0].set_xlabel('Anos de Experiência', fontsize=12)\n"
    "    axes[0].set_ylabel('Mediana jogos perdidos', fontsize=11)\n"
    '\n'
    "    cl_p = df.drop_duplicates('player_id')\n"
    "    gm_max = df.groupby('player_id')['games_missed'].max()\n"
    "    cl_p = cl_p.set_index('player_id')\n"
    "    cl_p['gm_max'] = gm_max\n"
    '\n'
    "    grupo_alto = cl_p[cl_p['gm_max'] > 4]['career_length'].dropna()\n"
    "    grupo_baixo = cl_p[cl_p['gm_max'] <= 4]['career_length'].dropna()\n"
    '\n'
    '    if len(grupo_alto) > 5 and len(grupo_baixo) > 5:\n'
    "        u_stat, p_mw = stats.mannwhitneyu(grupo_alto, grupo_baixo, alternative='two-sided')\n"
    '        bp = axes[1].boxplot([grupo_baixo.values, grupo_alto.values],\n'
    "                             tick_labels=['≤4 jogos', '>4 jogos'],\n"
    '                             patch_artist=True)\n'
    "        bp['boxes'][0].set_facecolor(CORES['protetor'])\n"
    "        bp['boxes'][1].set_facecolor(CORES['risco'])\n"
    "        for b in bp['boxes']:\n"
    '            b.set_alpha(0.7)\n'
    "        axes[1].tick_params(axis='x', labelsize=8)\n"
    "        axes[1].set_ylabel('Longevidade (temporadas)', fontsize=11)\n"
    "        axes[1].set_xlabel('Jogos perdidos (máx. carreira)', fontsize=11)\n"
    "        axes[1].text(0.96, 0.88, f'Mann-Whitney p {format_p_value(p_mw)} (expl.)',\n"
    "                     transform=axes[1].transAxes, ha='right', va='top', fontsize=8, clip_on=True,\n"
    "                     bbox=dict(boxstyle='round', facecolor=CORES['fundo'], alpha=0.8, pad=0.35))\n"
    '\n'
    '    finalize_dual_horizontal(fig, left=0.13, bottom=0.26, left_labelpad=12, right_labelpad=14)\n'
    "    save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/games_missed_{POSICAO}.png')\n"
    '    plt.close("all")\n'
    '\n'
    '# ---- bloco 38 ----\n'
    "if 'depth_team' in df.columns:\n"
    "    # Figura final: largura do texto e altura 0,2 da pagina (escala ~1 no LaTeX).\n"
    '    from thesis_figure_style import eixos_virgula\n'
    "    fig, axes = plt.subplots(1, 2, figsize=(6.2, 1.9), gridspec_kw={'width_ratios': [1, 1]})\n"
    '\n'
    "    dt_counts = df['depth_team'].value_counts().sort_index()\n"
    "    axes[0].bar([str(int(i)) for i in dt_counts.index], dt_counts.values, color=CORES['neutro'], alpha=0.85)\n"
    "    axes[0].set_xlabel('Profundidade no elenco (1 = titular)', fontsize=9, labelpad=1)\n"
    "    axes[0].set_ylabel('Frequência', fontsize=9, labelpad=2)\n"
    '\n'
    "    dt_by_exp = df.groupby('years_exp')['depth_team'].mean()\n"
    "    dt_by_exp = dt_by_exp[df.groupby('years_exp').size() >= 5]\n"
    "    axes[1].plot(dt_by_exp.index, dt_by_exp.values, color=CORES['risco'], linewidth=2, marker='o')\n"
    "    axes[1].set_xlabel('Anos de experiência', fontsize=9, labelpad=1)\n"
    "    axes[1].set_ylabel('Nível médio', fontsize=9, labelpad=2)\n"
    "    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))\n"
    "    for _ax in axes: _ax.tick_params(axis='both', labelsize=8.5)\n"
    '    eixos_virgula(axes[0], x=False, y=True)\n'
    '    eixos_virgula(axes[1], x=False, y=True, casas_y=1)\n'
    '    fig.subplots_adjust(left=0.085, right=0.995, top=0.97, bottom=0.22, wspace=0.25)\n'
    "    save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/depth_{POSICAO}.png', pad_inches=0.03)\n"
    '    plt.close("all")\n'
    '\n'
    '# ---- bloco 40 ----\n'
    "if 'offense_pct' in df.columns:\n"
    '    fig, axes = plt.subplots(2, 1, figsize=figsize_full(aspect=0.85))\n'
    '\n'
    "    cl_groups = pd.qcut(df['career_length'], q=3, labels=['Curta', 'Média', 'Longa'], duplicates='drop')\n"
    "    for group, color in zip(['Curta', 'Média', 'Longa'], [CORES['risco'], CORES['destaque'], CORES['protetor']]):\n"
    "        data = df[cl_groups == group]['offense_pct'].dropna()\n"
    '        axes[0].hist(data, bins=30, alpha=0.5, color=color, label=group, density=True)\n'
    "    axes[0].set_xlabel('offense_pct', fontsize=12)\n"
    "    axes[0].set_ylabel('Densidade', fontsize=12)\n"
    "    axes[0].set_title(f'Distribuição offense_pct por Longevidade — {POSICAO}s', fontsize=12, fontweight='bold')\n"
    '    axes[0].legend(fontsize=11)\n'
    '\n'
    "    off_by_exp = df.groupby('years_exp')['offense_pct']\n"
    '    med = off_by_exp.median()\n'
    '    q25 = off_by_exp.quantile(0.25)\n'
    '    q75 = off_by_exp.quantile(0.75)\n'
    "    valid = med.index[df.groupby('years_exp')['offense_pct'].count() >= 5]\n"
    "    axes[1].plot(med[valid].index, med[valid].values, color=CORES['neutro'], linewidth=2, marker='o')\n"
    "    axes[1].fill_between(med[valid].index, q25[valid].values, q75[valid].values, alpha=0.2, color=CORES['neutro'])\n"
    "    axes[1].set_xlabel('Anos de Experiência', fontsize=12)\n"
    "    axes[1].set_ylabel('offense_pct (mediana)', fontsize=12)\n"
    "    axes[1].set_title(f'Evolução do offense_pct — {POSICAO}s', fontsize=12, fontweight='bold')\n"
    '\n'
    '    plt.tight_layout()\n'
    "    save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/offense_pct_{POSICAO}.png')\n"
    '    plt.close("all")\n'
    'else:\n'
    "    print('offense_pct não disponível para esta posição.')\n"
    '\n'
    '# ---- bloco 42 ----\n'
    '# Fantasy Points como proxy de impacto\n'
    'fig, axes = plt.subplots(2, 1, figsize=figsize_full(aspect=0.85))\n'
    '\n'
    "if 'fantasy_points_ppr' in df.columns:\n"
    "    fp_by_exp = df.groupby('years_exp')['fantasy_points_ppr'].agg(['mean', 'median']).reset_index()\n"
    "    axes[0].plot(fp_by_exp['years_exp'], fp_by_exp['mean'], marker='o', color=CORES['neutro'], linewidth=2, label='Média')\n"
    "    axes[0].plot(fp_by_exp['years_exp'], fp_by_exp['median'], marker='s', color=CORES['destaque'], linewidth=2, linestyle='--', label='Mediana')\n"
    "    axes[0].set_xlabel('Anos de Experiência', fontsize=12)\n"
    "    axes[0].set_ylabel('Fantasy Points PPR', fontsize=12)\n"
    "    axes[0].set_title('Evolução dos Fantasy Points', fontsize=12, fontweight='bold')\n"
    '    axes[0].legend(fontsize=11)\n'
    '\n'
    "if 'career_fantasy_points_ppr' in df.columns:\n"
    "    cl_player = df.drop_duplicates('player_id').copy()\n"
    "    axes[1].scatter(cl_player['career_fantasy_points_ppr'], cl_player['career_length'], alpha=0.5, color=CORES['neutro'])\n"
    "    corr = cl_player[['career_fantasy_points_ppr', 'career_length']].corr().iloc[0, 1]\n"
    "    axes[1].set_xlabel('Career Fantasy Points (PPR)', fontsize=12)\n"
    "    axes[1].set_ylabel('Temporadas na NFL', fontsize=12)\n"
    "    axes[1].set_title(f'Career FP vs Longevidade (r={corr:.3f})', fontsize=12, fontweight='bold')\n"
    '\n'
    'plt.tight_layout()\n'
    "save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/fantasy_points_{POSICAO}.png')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 45 ----\n'
    'main_corr_vars = [\n'
    "    'years_exp', 'age', 'height', 'weight', 'draft_ovr',\n"
    "    'offense_pct', 'depth_team', 'games_missed', 'career_length'\n"
    ']\n'
    '\n'
    'position_specific = [\n'
    "    'season_rushing_yards', 'season_rush_attempts', 'season_receptions',\n"
    "    'season_receiving_yards', 'season_pass_attempts', 'season_passing_yards',\n"
    "    'season_targets', 'season_receiving_touchdown', 'season_total_yards'\n"
    ']\n'
    '\n'
    'main_corr_vars += [v for v in position_specific if v in df.columns]\n'
    'main_corr_vars = [v for v in main_corr_vars if v in df.columns]\n'
    '\n'
    "player_data = df.groupby('player_id')[main_corr_vars].agg(\n"
    "    {v: 'first' if v in ['years_exp', 'age', 'height', 'weight', 'draft_ovr', 'career_length'] else 'max' for v in main_corr_vars}\n"
    ')\n'
    '\n'
    "corr_matrix = player_data.corr(method='pearson')\n"
    "if 'career_length' in corr_matrix.columns:\n"
    "    _top = corr_matrix['career_length'].drop('career_length').abs().sort_values(ascending=False).head(11).index.tolist()\n"
    "    _keep = _top + ['career_length']\n"
    "    corr_matrix = corr_matrix.loc[_keep, _keep]\n"
    '\n'
    'corr_matrix = corr_matrix.rename(index=thesis_var_label, columns=thesis_var_label)\n'
    'n_corr = len(corr_matrix)\n'
    '_aspect = max(1.35, n_corr * 0.115)\n'
    '_annot = 9 if n_corr >= 10 else max(8, min(10, int(95 / n_corr)))\n'
    'fig, ax = plt.subplots(figsize=figsize_full(aspect=_aspect))\n'
    'mask = np.triu(np.ones_like(corr_matrix, dtype=bool))\n'
    'sns.heatmap(\n'
    "    corr_matrix, mask=mask, annot=True, fmt='.1f', cmap='RdBu_r',\n"
    "    center=0, vmin=-1, vmax=1, square=True, ax=ax,\n"
    "    annot_kws={'size': _annot}, linewidths=0.3,\n"
    "    cbar_kws={'label': 'Correlação de Pearson', 'shrink': 0.82}\n"
    ')\n'
    "ax.set_title(f'Matriz de Correlação — {POSICAO}s', fontsize=14, fontweight='bold')\n"
    "ax.tick_params(axis='x', labelsize=10)\n"
    "plt.setp(ax.get_xticklabels(), rotation=55, ha='right', rotation_mode='anchor')\n"
    "ax.tick_params(axis='y', labelsize=10)\n"
    'fig.subplots_adjust(bottom=0.30, left=0.24)\n'
    "save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/corr_matrix_{POSICAO}.png')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 47 ----\n'
    '# Pair plot das variáveis mais correlacionadas com career_length\n'
    "if 'career_length' in corr_matrix.columns:\n"
    "    top_vars = corr_matrix['career_length'].drop('career_length').abs().sort_values(ascending=False).head(4).index.tolist()\n"
    "    plot_vars = top_vars + ['career_length']\n"
    '\n'
    "    pair_df = df[plot_vars].drop_duplicates(subset=['career_length']).dropna()\n"
    '    if len(pair_df) > 10:\n'
    "        g = sns.pairplot(pair_df, corner=True, plot_kws={'alpha': 0.6, 'color': CORES['neutro']},\n"
    "                         diag_kws={'color': CORES['neutro']})\n"
    "        g.fig.suptitle(f'Pair Plot — Principais Correlaciones ({POSICAO})', y=1.02, fontsize=13, fontweight='bold')\n"
    '        plt.tight_layout()\n'
    "        save_thesis_figure(fig, ROOT / f'Análise Exploratória de Dados/figuras/pairplot_{POSICAO}.png')\n"
    '        plt.close("all")\n'
    '\n'
    '# ---- bloco 49 ----\n'
    '# Analise de Multicolinearidade\n'
    '# NOTA METODOLOGICA (auditoria maio/2026):\n'
    '# O VIF nao e calculado na EDA. VIF mede colinearidade em modelos\n'
    '# lineares de seccao cruzada e nao e diretamente informativo para modelos\n'
    '# de riscos proporcionais (Cox) ou aditivos (Aalen).\n'
    '#\n'
    '# A colinearidade e tratada na etapa de modelagem inferencial:\n'
    '#   - Cox TDC: stepwise backward + filter_collinear_covariates(|r| > 0.85)\n'
    '#     em survival_utils.py, linha 356-403\n'
    '#   - Aalen: adaptive_rank_filter() garante posto completo em cada t_j\n'
    '#     (Martinussen & Scheike 2006, Cap. 5)\n'
    '#\n'
    '# Para diagnostico exploratorio, a matriz de correlacao de Pearson\n'
    '# ja identifica pares altamente correlacionados no painel jogador-temporada.\n'
    'print("Multicolinearidade delegada a etapa de modelagem (Cox/Aalen).")\n'
    'print("Ver matriz de correlacao.")\n'
    '\n'
    '# ---- bloco 51 ----\n'
    "cl = df.groupby('player_id')['season'].nunique()\n"
    'pct_le3 = (cl <= 3).mean() * 100\n'
    '\n'
    'print(f"""\n'
    '### Principais Achados — EDA {POSICAO}\n'
    '\n'
    '1. **Distribuição de carreiras:** {pct_le3:.0f}% das carreiras observadas têm até 3 temporadas.\n'
    '2. **Draft:** jogadores mais valorizados no draft tendem a apresentar maior permanência observada.\n'
    '3. **Disponibilidade e uso:** variáveis como `games_missed`, `depth_team` e `offense_pct` ajudam a contextualizar o risco de saída.\n'
    '4. **Leitura metodológica:** esta EDA descreve padrões observados na janela 2012–2024 e não deve ser lida como reconstrução completa do histórico pré-2012.\n'
    '""")\n'
)


def main() -> None:
    namespace = globals()
    namespace["__name__"] = "__passo__"
    namespace["__file__"] = str(SCRIPT_PATH)

    import sys as _sys
    _sys.path.insert(0, str(ROOT / "Predição/shared"))
    from thesis_figure_style import apply_thesis_rcparams, save_thesis_figure, figsize_full, figsize_dual_horizontal, finalize_dual_horizontal, format_p_value, thesis_var_label
    apply_thesis_rcparams()
    namespace["save_thesis_figure"] = save_thesis_figure
    namespace["figsize_full"] = figsize_full
    namespace["figsize_dual_horizontal"] = figsize_dual_horizontal
    namespace["finalize_dual_horizontal"] = finalize_dual_horizontal
    namespace["format_p_value"] = format_p_value
    namespace["thesis_var_label"] = thesis_var_label
    exec(compile(CODIGO_PASSO, str(SCRIPT_PATH), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
