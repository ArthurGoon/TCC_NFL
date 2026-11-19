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
    'import seaborn as sns\n'
    'from lifelines import KaplanMeierFitter\n'
    'import warnings\n'
    'import sys\n'
    "warnings.filterwarnings('ignore')\n"
    "pd.set_option('display.max_columns', None)\n"
    "plt.rcParams['font.family'] = 'sans-serif'\n"
    '\n'
    'from pathlib import Path\n'
    '\n'
    'ROOT = Path.cwd().resolve()\n'
    'for candidate in [ROOT, ROOT.parent, ROOT.parent.parent, ROOT.parent.parent.parent, ROOT.parent.parent.parent.parent]:\n'
    "    if (candidate / 'DataBase' / 'yearly_player_stats_offense.csv').exists() and (candidate / 'Inferência').exists():\n"
    '        ROOT = candidate\n'
    '        break\n'
    '\n'
    "INFER_DIR = ROOT / 'Inferência'\n"
    "COX_DIR = INFER_DIR / 'Modelagem de COX'\n"
    "PREP_DIR = INFER_DIR / 'Preparação Modelagem'\n"
    "sys.path.append(str((INFER_DIR / 'shared').resolve()))\n"
    '\n'
    'from survival_utils import add_gap_aware_survival_structure, load_clean_position_data  # noqa: E402\n'
    '\n'
    "POSICAO = 'RB'\n"
    "CORES = {'risco': '#D85A30', 'protetor': '#1D9E75', 'neutro': '#2E4057',\n"
    "         'destaque': '#F4A261', 'fundo': '#F8F9FA'}\n"
    "print('Preparação da Modelagem Cox — RB')\n"
    '\n'
    '# ---- bloco 5 ----\n'
    'df = load_clean_position_data(ROOT, POSICAO)\n'
    '\n'
    "for col in ['years_exp','draft_round','age','height','weight','draft_ovr',\n"
    "            'depth_team','offense_pct','games_missed','season','games_played_season']:\n"
    '    if col in df.columns:\n'
    "        df[col] = pd.to_numeric(df[col], errors='coerce')\n"
    '\n'
    "df['draft_round'] = df['draft_round'].fillna(8)\n"
    "df['undrafted_flag'] = (df['draft_round'] == 8).astype(int)\n"
    "df.loc[df['undrafted_flag'] == 1, 'draft_ovr'] = 0\n"
    "df = df.sort_values(['player_id', 'season']).reset_index(drop=True)\n"
    '\n'
    'print(f\'Jogadores: {df["player_id"].nunique()}\')\n'
    'print(f\'Temporadas disponíveis: {sorted(df["season"].dropna().unique().astype(int))}\')\n'
    "print(f'Registros jogador-temporada: {len(df)}')\n"
    'print(f\'Temporadas multi-time: {int(df.get("multi_team_season_flag", 0).sum())}\')\n'
    '\n'
    '# ---- bloco 6 ----\n'
    "if 'games_played_season' in df.columns:\n"
    '    n_antes = len(df)\n'
    "    df = df[(df['games_played_season'] >= 1) | (df['offense_snaps'] > 0)].copy()\n"
    "    print(f'Removidos por inclusão mínima: {n_antes - len(df)}')\n"
    "elif 'offense_snaps' in df.columns:\n"
    '    n_antes = len(df)\n'
    "    df = df[df['offense_snaps'] > 0].copy()\n"
    "    print(f'Removidos por snaps == 0: {n_antes - len(df)}')\n"
    '\n'
    "print(f'Registros após filtro: {len(df)}')\n"
    'print(f\'Jogadores após filtro: {df["player_id"].nunique()}\')\n'
    '\n'
    "main_cols = ['years_exp','draft_round','age','height','weight','draft_ovr',\n"
    "             'depth_team','offense_pct','games_missed', 'team_rush_attempts_share', 'team_rushing_yards_share']\n"
    'main_cols = [c for c in main_cols if c in df.columns]\n'
    'missing_pct = df[main_cols].isna().mean().sort_values(ascending=False)\n'
    "print('\\nMissing por variável principal:')\n"
    'print(missing_pct[missing_pct > 0].to_string())\n'
    '\n'
    '# ---- bloco 8 ----\n'
    'df = add_gap_aware_survival_structure(df)\n'
    '\n'
    "assert (df['stop'] > df['start']).all(), 'ERRO: existem registros com stop <= start'\n"
    "print(f'✓ Painel start-stop construído: {len(df)} registros')\n"
    'print(f\'  Eventos: {int(df["event"].sum())} | Censurados: {int((df["event"]==0).sum())}\')\n'
    'print(f\'  Veteranos com histórico não observado: {int(df["unobserved_history_flag"].sum())}\')\n'
    'print(f\'  Jogadores com gap dentro da janela: {int(df.groupby("player_id")["had_prior_gap_flag"].max().sum())}\')\n'
    '\n'
    '# ---- bloco 9 ----\n'
    'players_with_gap = (\n'
    "    df.groupby('player_id', observed=True)['had_prior_gap_flag']\n"
    '      .max()\n'
    '      .sum()\n'
    ')\n'
    '\n'
    "max_gap = int(df['within_window_gap_years_before_t'].max()) if 'within_window_gap_years_before_t' in df.columns else 0\n"
    '\n'
    "print(f'Jogadores com retorno após hiato na janela: {int(players_with_gap)}')\n"
    "print(f'Maior hiato acumulado observado: {max_gap} temporada(s)')\n"
    "print('Readmissões são tratadas como continuidade da mesma carreira, sem reset de start/stop nem criação de IDs artificiais.')\n"
    "print(f'\\n✓ Painel final com entry: {df.shape}')\n"
    '\n'
    '# ---- bloco 11 ----\n'
    'CAREER_ADITIVAS = [\n'
    "    'career_rush_attempts', 'career_rushing_yards', 'career_rush_touchdown',\n"
    "    'career_touches', 'career_total_yards', 'career_offense_snaps',\n"
    "    'career_fantasy_points_ppr', 'career_receptions', 'career_receiving_yards',\n"
    "    'career_targets', 'career_rush_attempts_redzone', 'career_rush_touchdown_redzone',\n"
    "    'career_fumble', 'career_fumble_lost', 'career_yards_after_catch',\n"
    ']\n'
    '\n'
    'COMPONENTES_TAXA = [\n'
    "    ('career_rushing_yards',    'career_rush_attempts', 'career_lag_ypc'),\n"
    "    ('career_rush_touchdown',   'career_rush_attempts', 'career_lag_rush_td_pct'),\n"
    ']\n'
    '\n'
    'lag_cols = []\n'
    'rate_lag_cols = []\n'
    '\n'
    'for cv in CAREER_ADITIVAS:\n'
    "    sv  = cv.replace('career_', 'season_')\n"
    "    lag = cv.replace('career_', 'observed_career_lag_')\n"
    '    if cv in df.columns and sv in df.columns:\n'
    '        df[lag] = (df[cv] - df[sv]).clip(lower=0)\n'
    '        lag_cols.append(lag)\n'
    '    else:\n'
    "        print(f'AVISO: ausente → {cv} ou {sv}')\n"
    '\n'
    'for num_col, den_col, lag_name in COMPONENTES_TAXA:\n'
    "    lag_name = lag_name.replace('career_lag_', 'observed_career_lag_')\n"
    "    num_lag = num_col.replace('career_', 'observed_career_lag_')\n"
    "    den_lag = den_col.replace('career_', 'observed_career_lag_')\n"
    '    if num_lag in df.columns and den_lag in df.columns:\n'
    '        df[lag_name] = np.where(df[den_lag] > 50,\n'
    '                                df[num_lag] / df[den_lag],\n'
    '                                np.nan)\n'
    '        lag_cols.append(lag_name)\n'
    '        rate_lag_cols.append(lag_name)\n'
    '    else:\n'
    "        print(f'AVISO: componente ausente → {lag_name}')\n"
    '\n'
    "print(f'Total de covariáveis lag criadas: {len(lag_cols)}')\n"
    "print(f'Lag cols: {lag_cols}')\n"
    '\n'
    '# ---- bloco 13 ----\n'
    "CONTEXTO_POSICAO = ['depth_team', 'offense_pct', 'games_missed', 'age', 'years_exp',\n"
    "            'unobs_career_years', 'pre_window_veteran', 'first_observed_season',\n"
    "            'unobserved_history_flag', 'within_window_gap_years_before_t', 'had_prior_gap_flag',\n"
    "            'team_rush_attempts_share', 'team_rushing_yards_share']\n"
    '\n'
    "DEMOGRAFICAS = ['height', 'weight', 'draft_ovr', 'draft_round', 'undrafted_flag']\n"
    '\n'
    'CONTEXTO_POSICAO = [c for c in CONTEXTO_POSICAO if c in df.columns]\n'
    'DEMOGRAFICAS = [c for c in DEMOGRAFICAS if c in df.columns]\n'
    '\n'
    "ID_COLS   = ['player_id', 'player_name', 'season']\n"
    "TIME_COLS = ['start', 'stop', 'event', 'entry']\n"
    '\n'
    "print(f'Contexto: {CONTEXTO_POSICAO}')\n"
    "print(f'Demográficas: {DEMOGRAFICAS}')\n"
    '\n'
    '# ---- bloco 15 ----\n'
    'cov_cols = lag_cols + CONTEXTO_POSICAO + DEMOGRAFICAS\n'
    '\n'
    'cov_cols = list(dict.fromkeys(cov_cols))\n'
    'cov_cols = [c for c in cov_cols if c in df.columns]\n'
    '\n'
    'df_cox = df[ID_COLS + TIME_COLS + cov_cols].copy()\n'
    '\n'
    'missing_report = df_cox[cov_cols].isna().mean().sort_values(ascending=False)\n'
    "print('Missing por variável (antes de imputação):')\n"
    'print(missing_report[missing_report > 0].to_string())\n'
    '\n'
    'taxa_cols = [c for c in cov_cols if any(x in c for x in \n'
    "             ['comp_pct', 'ypa', 'int_pct', 'td_pct', 'ypc', 'yptarget', 'ypr',\n"
    "              'catch_rate', 'adot_acc', 'rush_td_pct', 'yptouch', 'reception_rate'])]\n"
    'for col in taxa_cols:\n'
    '    if col in df_cox.columns:\n'
    "        miss_col = f'{col}_missing'\n"
    '        df_cox[miss_col] = df_cox[col].isna().astype(int)\n'
    '\n'
    'cov_cols = [c for c in df_cox.columns if c not in ID_COLS + TIME_COLS]\n'
    '\n'
    'n_antes = len(df_cox)\n'
    'df_cox = df_cox.dropna(subset=TIME_COLS)\n'
    "print(f'\\nLinhas removidas por NaN em colunas de tempo/evento: {n_antes - len(df_cox)} '\n"
    "      f'({(n_antes - len(df_cox))/n_antes:.1%})')\n"
    "print('\\nMissing residual preservado para imputação posterior no treino/modelo.')\n"
    "print(f'\\nPainel final: {df_cox.shape}')\n"
    'print(f\'Jogadores únicos: {df_cox["player_id"].nunique()}\')\n'
    'print(f\'Eventos (encerramento): {df_cox["event"].sum()}\')\n'
    'print(f\'Taxa de censura: {1 - df_cox["event"].mean():.1%}\')\n'
    '\n'
    '# ---- bloco 16 ----\n'
    "eventos_por_ano = df_cox.groupby('stop').agg(\n"
    "    n_jogadores=('player_id', 'count'),\n"
    "    n_eventos=('event', 'sum')\n"
    ").assign(taxa_evento=lambda x: x['n_eventos']/x['n_jogadores'])\n"
    "print('Distribuição de eventos por ano de carreira:')\n"
    'print(eventos_por_ano.to_string())\n'
    '\n'
    'fig, ax = plt.subplots(figsize=(10, 5))\n'
    "ax.bar(eventos_por_ano.index, eventos_por_ano['taxa_evento'], \n"
    "       color=CORES['risco'], alpha=0.85)\n"
    "ax.set_xlabel('Ano de Carreira (stop)', fontsize=11)\n"
    "ax.set_ylabel('Taxa de Evento (encerramento)', fontsize=11)\n"
    "ax.set_title(f'Taxa de Encerramento por Ano — {POSICAO}s', fontsize=12, fontweight='bold')\n"
    'plt.tight_layout()\n'
    "plt.savefig(PREP_DIR / 'figuras' / f'taxa_evento_{POSICAO}.png', dpi=150, bbox_inches='tight')\n"
    'plt.close("all")\n'
    '\n'
    '# ---- bloco 18 ----\n'
    "df_cox.to_pickle(PREP_DIR / 'resultados' / f'df_cox_{POSICAO}.pkl')\n"
    "df_cox.to_csv(PREP_DIR / 'resultados' / f'df_cox_{POSICAO}.csv', index=False)\n"
    "print(f'Exportado: df_cox_{POSICAO}.pkl e df_cox_{POSICAO}.csv')\n"
    "print(f'Shape: {df_cox.shape}')\n"
    "print(f'Colunas exportadas: {df_cox.columns.tolist()}')\n"
)


def main() -> None:
    namespace = globals()
    namespace["__name__"] = "__passo__"
    namespace["__file__"] = str(SCRIPT_PATH)
    exec(compile(CODIGO_PASSO, str(SCRIPT_PATH), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
