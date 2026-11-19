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
    '# ---- bloco 2 ----\n'
    'import pandas as pd\n'
    'import numpy as np\n'
    'import matplotlib.pyplot as plt\n'
    'import json\n'
    'import subprocess\n'
    'import warnings\n'
    'from pathlib import Path\n'
    'import sys\n'
    "warnings.filterwarnings('ignore')\n"
    "pd.set_option('display.max_columns', None)\n"
    "plt.rcParams['font.family'] = 'sans-serif'\n"
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
    'INFER_COX_DIR = COX_DIR\n'
    '\n'
    "sys.path.append(str((INFER_DIR / 'shared').resolve()))\n"
    '\n'
    "AALEN_SHARED = AALEN_DIR / 'shared'\n"
    "AALEN_RES = AALEN_DIR / 'resultados'\n"
    "AALEN_FIG = AALEN_DIR / 'figuras'\n"
    'AALEN_RES.mkdir(parents=True, exist_ok=True)\n'
    'AALEN_FIG.mkdir(parents=True, exist_ok=True)\n'
    'from survival_utils import (\n'
    '    fit_imputation_stats, apply_imputation_stats,\n'
    '    fit_scaler, apply_scaler,\n'
    '    filter_collinear_covariates, force_missing_flags\n'
    ')\n'
    "POSICAO = 'QB'\n"
    "CORES = {'risco': '#D85A30', 'protetor': '#1D9E75', 'neutro': '#2E4057',\n"
    "         'destaque': '#F4A261', 'fundo': '#F8F9FA'}\n"
    "print(f'Inferência Aalen — {POSICAO}')\n"
    '\n'
    '# ---- bloco 4 ----\n'
    "df_cox = pd.read_pickle(PREP_DIR / 'resultados' / f'df_cox_{POSICAO}.pkl')\n"
    "print(f'Painel carregado: {df_cox.shape}')\n"
    'print(f\'Jogadores: {df_cox["player_id"].nunique()} | Eventos: {df_cox["event"].sum()}\')\n'
    'print(f\'Taxa de censura: {1 - df_cox["event"].mean():.1%}\')\n'
    "ID_COLS = ['player_id', 'player_name', 'season']\n"
    "TIME_COLS = ['start', 'stop', 'event', 'entry']\n"
    'cov_cols = [c for c in df_cox.columns if c not in ID_COLS + TIME_COLS]\n'
    "cov_cols = [c for c in cov_cols if not c.endswith('_missing')]\n"
    "cov_cols = [c for c in cov_cols if c not in ['years_exp', 'first_observed_season', 'draft_round']]\n"
    "print(f'\\nCovariáveis disponíveis: {len(cov_cols)}')\n"
    'for c in cov_cols:\n'
    "    print(f'  {c}')\n"
    '\n'
    '# ---- bloco 6 ----\n'
    'imputer_stats = fit_imputation_stats(df_cox, cov_cols)\n'
    'df_proc = apply_imputation_stats(df_cox, cov_cols, imputer_stats)\n'
    'scaler = fit_scaler(df_proc, cov_cols)\n'
    'df_proc = apply_scaler(df_proc, scaler, cov_cols)\n'
    "print(f'Dataset processado: {df_proc.shape}')\n"
    "print(f'Missing após imputação: {df_proc[cov_cols].isna().sum().sum()}')\n"
    '\n'
    '# ---- bloco 8 ----\n'
    'cov_cols_filtered = filter_collinear_covariates(df_cox, cov_cols, threshold=0.85, verbose=True)\n'
    "print(f'\\nCovariáveis após filtragem: {len(cov_cols_filtered)}')\n"
    '\n'
    '# ---- bloco 10 ----\n'
    "forbidden = ['years_exp', 'first_observed_season']\n"
    'present = [c for c in forbidden if c in cov_cols_filtered]\n'
    "assert len(present) == 0, f'Metadados presentes: {present}'\n"
    "print('Metadados ausentes — OK.')\n"
    "print(f'Covariáveis para screening: {len(cov_cols_filtered)}')\n"
    "export_cols = ['player_id', 'start', 'stop', 'event'] + cov_cols_filtered\n"
    'df_export = df_proc[export_cols].copy()\n'
    "for col in ['start', 'stop', 'event']:\n"
    "    df_export[col] = pd.to_numeric(df_export[col], errors='coerce')\n"
    'for col in cov_cols_filtered:\n'
    "    df_export[col] = pd.to_numeric(df_export[col], errors='coerce')\n"
    "csv_path = AALEN_RES / f'aalen_input_{POSICAO}.csv'\n"
    'df_export.to_csv(csv_path, index=False)\n'
    "print(f'Exportado: {csv_path}')\n"
    '\n'
    '# ---- bloco 12 ----\n'
    "screening_csv = AALEN_RES / f'aalen_screening_{POSICAO}.csv'\n"
    "cov_list = ','.join(cov_cols_filtered)\n"
    "cmd = ['Rscript', str(AALEN_SHARED / 'run_aalen_univariate_screening.R'), str(csv_path), str(screening_csv), cov_list, POSICAO]\n"
    "print('Executando screening univariado...')\n"
    'result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)\n'
    'print(result.stdout)\n'
    'if result.returncode != 0:\n'
    "    raise RuntimeError(f'R screening failed')\n"
    'screening = pd.read_csv(screening_csv)\n'
    "screening = screening.sort_values(['p_value_null', 'covariate'])\n"
    "print(f'\\nScreening: {len(screening)} covs testadas')\n"
    'print(f\'Significativas (p < 0.05): {screening["significant_null"].sum()}\')\n'
    "print(screening[['covariate', 'p_value_null', 'significant_null']].head(10).to_string(index=False))\n"
    '\n'
    '# ---- bloco 14 ----\n'
    "n_events = int(df_cox['event'].sum())\n"
    'max_vars = 10\n'
    "print(f'Eventos: {n_events} → Máximo: {max_vars}')\n"
    "sig_screening = screening[screening['significant_null'] == True].copy()\n"
    "selected_from_screening = sig_screening['covariate'].tolist()\n"
    "print(f'\\nSignificativas no screening: {len(selected_from_screening)}')\n"
    'if len(selected_from_screening) > max_vars:\n'
    '    selected_terms = selected_from_screening[:max_vars]\n'
    '    # Forca dummies _missing a acompanhar variaveis de taxa\n'
    '    selected_terms = force_missing_flags(selected_terms, df_proc.columns.tolist())\n'
    "    print(f'Parcimônia aplicada: {max_vars} mantidas')\n"
    'else:\n'
    '    selected_terms = selected_from_screening\n'
    '    # Forca dummies _missing a acompanhar variaveis de taxa\n'
    '    selected_terms = force_missing_flags(selected_terms, df_proc.columns.tolist())\n'
    "    print(f'Todas as {len(selected_from_screening)} mantidas')\n"
    "print('\\nSelecionadas:')\n"
    'for t in selected_terms:\n'
    '    try:\n'
    "        pval = screening[screening['covariate'] == t]['p_value_null'].values[0]\n"
    "        print(f'  {t} (p={pval:.4f})')\n"
    '    except IndexError:\n'
    "        print(f'  {t} (p=N/A, n\\u00e3o testada no screening)')\n"
    '\n'
    '# ---- bloco 16 ----\n'
    '# Sensibilidade: lista da triagem refeita com as linhas ordenadas no aareg\n'
    "sens_dir = AALEN_RES / 'triagem_corrigida'\n"
    'sens_dir.mkdir(parents=True, exist_ok=True)\n'
    "sens = subprocess.run(['Rscript', str(AALEN_SHARED / 'run_aalen_tied_inference.R'), str(csv_path), str(sens_dir), POSICAO, ','.join(selected_terms)], capture_output=True, text=True, timeout=600)\n"
    'if sens.returncode != 0:\n'
    "    raise RuntimeError(sens.stderr)\n"
    '# Modelo principal: lista fixada\n'
    "selected_terms = json.loads((AALEN_SHARED / 'listas_aalen_fixadas.json').read_text(encoding='utf-8'))[POSICAO]\n"
    "cov_list = ','.join(selected_terms)\n"
    "cmd = ['Rscript', str(AALEN_SHARED / 'run_aalen_tied_inference.R'), str(csv_path), str(AALEN_RES), POSICAO, cov_list]\n"
    "print('Executando modelo Aalen 100% não-paramétrico...')\n"
    'result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)\n'
    'print(result.stdout)\n'
    'if result.stderr:\n'
    "    print('STDERR:', result.stderr)\n"
    'if result.returncode != 0:\n'
    "    raise RuntimeError(f'R failed: {result.returncode}')\n"
    "print('Modelo concluído.')\n"
    '\n'
    '# ---- bloco 18 ----\n'
    "summary_path = AALEN_RES / f'aalen_summary_{POSICAO}.json'\n"
    "with open(summary_path, 'r') as f:\n"
    '    aalen_summary = json.load(f)\n'
    "final_covs = aalen_summary['covariates_tested']\n"
    "if not isinstance(final_covs, list) or not set(final_covs).issubset(selected_terms):\n"
    "    raise RuntimeError('Covariáveis efetivas do R inconsistentes com a lista submetida')\n"
    "final_covs_path = AALEN_RES / f'aalen_final_covs_{POSICAO}.json'\n"
    "with open(final_covs_path, 'w') as f:\n"
    '    json.dump(final_covs, f, indent=2)\n'
    "print(f'Lista efetiva do R salva: {final_covs_path}')\n"
    'print(f\'Posição: {aalen_summary["position"]}\')\n'
    'print(f\'Subjects: {aalen_summary["n_subjects"]} | Periods: {aalen_summary["n_periods"]} | Events: {aalen_summary["n_events"]}\')\n'
    "tests_null = aalen_summary['tests_null_effect']\n"
    "print('\\nTeste de coeficiente medio temporal ponderado (aareg):')\n"
    'print(f\'{"Covariável":<45} {"p-value":>10} {"Signif.":>8}\')\n'
    "print('-' * 65)\n"
    'for term, vals in tests_null.items():\n'
    "    if term == '(Intercept)': continue\n"
    "    sig = '***' if vals['p_value'] < 0.05 else ('*' if vals['p_value'] < 0.10 else '')\n"
    '    print(f\'{term:<45} {vals["p_value"]:>10.4f} {sig:>8}\')\n'
    "sig_multi = [t for t, v in tests_null.items() if v['p_value'] < 0.05 and t != '(Intercept)']\n"
    "print(f'\\nSignificativas (p < 0.05): {len(sig_multi)}')\n"
    'for t in sig_multi:\n'
    "    print(f'  {t}')\n"
    '\n'
    '# ---- bloco 20 ----\n'
    "cum_path = AALEN_RES / f'aalen_cum_coefs_{POSICAO}.csv'\n"
    "var_path = AALEN_RES / f'aalen_robvar_cum_{POSICAO}.csv'\n"
    'cum_df = pd.read_csv(cum_path)\n'
    'var_df = pd.read_csv(var_path)\n'
    'time_col = cum_df.columns[0]\n'
    "terms_to_plot = [c for c in cum_df.columns if c not in [time_col, '(Intercept)']]\n"
    'n_terms = len(terms_to_plot)\n'
    'if n_terms > 0:\n'
    "    out_path = AALEN_FIG / f'aalen_cum_coefs_{POSICAO}.pdf'\n"
    '    plot_aalen_cum_coefs_thesis(\n'
    '        cum_df, var_df, POSICAO, out_path,\n'
    '        height_frac=aalen_height_frac(n_terms),\n'
    '    )\n'
    "    print(f'Figura salva: {out_path}')\n"
    '\n'
    '# ---- bloco 22 ----\n'
    "cox_hr_path = INFER_COX_DIR / 'resultados' / f'cox_hr_{POSICAO}.csv'\n"
    "cox_final_covs = pd.read_csv(cox_hr_path)['covariate'].tolist()\n"
    'from lifelines import CoxTimeVaryingFitter\n'
    "cox_cols_fit = ['player_id', 'start', 'stop', 'event'] + cox_final_covs\n"
    'df_cox_fit = df_proc[cox_cols_fit].dropna().copy()\n'
    'ctv = CoxTimeVaryingFitter(penalizer=0.0)\n'
    "ctv.fit(df_cox_fit, id_col='player_id', start_col='start', stop_col='stop', event_col='event', show_progress=False)\n"
    'cox_coefs = ctv.summary.copy().reset_index()\n'
    "print('Cox TDC reajustado.')\n"
    '\n'
    '# ---- bloco 23 ----\n'
    'common_covs = [c for c in final_covs if c in cox_final_covs]\n'
    'compare_covs = common_covs if len(common_covs) > 0 else list(set(final_covs + cox_final_covs))\n'
    "aalen_null_p = {k: v['p_value'] for k, v in tests_null.items() if k != '(Intercept)'}\n"
    "cox_hr = pd.read_csv(INFER_COX_DIR / 'resultados' / f'cox_hr_{POSICAO}.csv')\n"
    "comp_path = AALEN_FIG / f'comparacao_cox_aalen_{POSICAO}.pdf'\n"
    'plot_comparacao_cox_aalen_thesis(\n'
    '    compare_covs,\n'
    '    cox_hr,\n'
    '    aalen_null_p,\n'
    '    cum_df,\n'
    '    var_df,\n'
    '    comp_path,\n'
    ')\n'
    "print(f'Figura comparativa salva: {comp_path}')\n"
    '\n'
    '# ---- bloco 25 ----\n'
    "print(f'=== RESUMO AALEN — {POSICAO} ===')\n"
    "print(f'Covariáveis testadas (pré-Cox): {len(cov_cols)}')\n"
    "print(f'Após filtragem de colinearidade: {len(cov_cols_filtered)}')\n"
    "print(f'Significativas no screening (p<0.05): {len(selected_from_screening)}')\n"
    "print(f'Covariáveis candidatas do Aalen: {len(selected_terms)}')\n"
    "print(f'Covariáveis efetivas do Aalen (multivariado): {len(final_covs)}')\n"
    "print(f'Covariáveis finais do Cox TDC: {len(cox_final_covs)}')\n"
    "print(f'Covariáveis em comum: {len(common_covs)}')\n"
    'comparison = {\n'
    "    'position': POSICAO,\n"
    "    'covariates_pre_cox': cov_cols,\n"
    "    'covariates_after_collinearity': cov_cols_filtered,\n"
    "    'aalen_significant_univariate': selected_from_screening,\n"
    "    'aalen_candidates_before_R_input_filter': selected_terms,\n"
    "    'aalen_final_multivariate': final_covs,\n"
    "    'cox_final': cox_final_covs,\n"
    "    'common_significant': common_covs,\n"
    "    'aalen_only': [c for c in final_covs if c not in cox_final_covs],\n"
    "    'cox_only': [c for c in cox_final_covs if c not in final_covs]\n"
    '}\n'
    "comp_json_path = AALEN_RES / f'comparacao_cox_aalen_{POSICAO}.json'\n"
    "with open(comp_json_path, 'w') as f:\n"
    '    json.dump(comparison, f, indent=2)\n'
    "print(f'\\nResumo comparativo salvo: {comp_json_path}')\n"
)


def main() -> None:
    namespace = globals()
    namespace["__name__"] = "__passo__"
    namespace["__file__"] = str(SCRIPT_PATH)

    import sys as _sys
    _sys.path.insert(0, str(ROOT / "Predição/shared"))
    from thesis_figure_style import (
        apply_thesis_rcparams,
        save_thesis_figure,
        figsize_full,
        plot_aalen_cum_coefs_thesis,
        plot_comparacao_cox_aalen_thesis,
        aalen_height_frac,
    )
    apply_thesis_rcparams()
    namespace["save_thesis_figure"] = save_thesis_figure
    namespace["figsize_full"] = figsize_full
    namespace["plot_aalen_cum_coefs_thesis"] = plot_aalen_cum_coefs_thesis
    namespace["plot_comparacao_cox_aalen_thesis"] = plot_comparacao_cox_aalen_thesis
    namespace["aalen_height_frac"] = aalen_height_frac
    exec(compile(CODIGO_PASSO, str(SCRIPT_PATH), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
