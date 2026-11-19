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
    'import warnings\n'
    "warnings.filterwarnings('ignore')\n"
    "pd.set_option('display.max_columns', None)\n"
    "plt.rcParams['font.family'] = 'sans-serif'\n"
    '\n'
    'from lifelines import CoxTimeVaryingFitter, CoxPHFitter\n'
    'from lifelines.statistics import proportional_hazard_test\n'
    'from lifelines.utils import concordance_index\n'
    'from sklearn.preprocessing import StandardScaler\n'
    'import scipy.stats as scipy_stats\n'
    'import sys\n'
    'import subprocess\n'
    'from pathlib import Path\n'
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
    'INFER_COX_DIR = COX_DIR\n'
    '\n'
    "sys.path.append(str((INFER_DIR / 'shared').resolve()))\n"
    '\n'
    'from survival_utils import fit_imputation_stats, apply_imputation_stats, fit_scaler, apply_scaler, temporal_player_splits, force_missing_flags, filter_collinear_covariates, covariaveis_identificaveis_cox\n'
    '\n'
    "POSICAO = 'RB'\n"
    "CORES = {'risco': '#D85A30', 'protetor': '#1D9E75', 'neutro': '#2E4057',\n"
    "         'destaque': '#F4A261', 'fundo': '#F8F9FA'}\n"
    '\n'
    "df_cox = pd.read_pickle(PREP_DIR / 'resultados' / f'df_cox_{POSICAO}.pkl')\n"
    "print(f'Painel carregado: {df_cox.shape}')\n"
    'print(f\'Jogadores: {df_cox["player_id"].nunique()} | Eventos: {df_cox["event"].sum()}\')\n'
    'print(f\'Taxa de censura: {1 - df_cox["event"].mean():.1%}\')\n'
    '\n'
    '# ---- bloco 4 ----\n'
    "ID_COLS   = ['player_id', 'player_name', 'season']\n"
    "TIME_COLS = ['start', 'stop', 'event', 'entry']\n"
    'cov_cols  = [c for c in df_cox.columns if c not in ID_COLS + TIME_COLS]\n'
    "cov_cols = [c for c in cov_cols if c != 'draft_round']\n"
    "print(f'Covariáveis ({len(cov_cols)}):')\n"
    'for c in cov_cols:\n'
    "    print(f'  {c}')\n"
    '\n'
    '# ---- bloco 6 ----\n'
    'imputer_stats = fit_imputation_stats(df_cox, cov_cols)\n'
    '\n'
    'df_model = apply_imputation_stats(df_cox, cov_cols, imputer_stats)\n'
    'scaler = StandardScaler()\n'
    'df_model[cov_cols] = scaler.fit_transform(df_model[cov_cols])\n'
    '\n'
    "cols_fit = ['player_id', 'start', 'stop', 'event'] + cov_cols\n"
    'df_fit   = df_model[cols_fit].copy()\n'
    '\n'
    '\n'
    '# ---- bloco 8 ----\n'
    '# Filtro de colinearidade (|r| > 0.85) antes do stepwise backward\n'
    '# Colunas nao identificaveis (years_exp = eixo do tempo; first_observed_season; duplicatas _missing)\n'
    '# saem ANTES do filtro de colinearidade, como no Aalen (passo 14). No filtro, years_exp\n'
    '# (r = 0,97 com age no QB) tiraria age e depois seria removida por ser o eixo do tempo.\n'
    'cov_cols = covariaveis_identificaveis_cox(df_fit, cov_cols)\n'
    'cov_cols_filtered = filter_collinear_covariates(df_fit, cov_cols, threshold=0.85, verbose=True)\n'
    'cov_cols = force_missing_flags(cov_cols_filtered, df_fit.columns.tolist())\n'
    'cov_cols = covariaveis_identificaveis_cox(df_fit, cov_cols)\n'
    'print(f"Covariaveis apos filtro de colinearidade: {len(cov_cols)}")\n'
    '# Modelo saturado = conjunto POS-filtro de colinearidade (o que a escrita descreve).\n'
    'ctv_full = CoxTimeVaryingFitter(penalizer=0.0)\n'
    "ctv_full.fit(df_fit[['player_id', 'start', 'stop', 'event'] + cov_cols],\n"
    "             id_col='player_id', start_col='start',\n"
    "             stop_col='stop', event_col='event', show_progress=False)\n"
    "print(f'Log-likelihood parcial (modelo saturado pos-filtro): {ctv_full.log_likelihood_:.4f}')\n"
    '\n'
    '\n'
    'def backward_cox_tdv(df_fit, id_col, start_col, stop_col, event_col,\n'
    '                     covariates, penalizer=0.0, threshold=0.05, verbose=True):\n'
    '    current_covs = list(covariates)\n'
    '    iteration = 0\n'
    '    while True:\n'
    '        cols = [id_col, start_col, stop_col, event_col] + current_covs\n'
    '        model = CoxTimeVaryingFitter(penalizer=penalizer)\n'
    '        model.fit(df_fit[cols], id_col=id_col, start_col=start_col,\n'
    '                  stop_col=stop_col, event_col=event_col, show_progress=False)\n'
    "        pvals = model.summary['p'].copy()\n"
    '        worst_var = pvals.idxmax()\n'
    '        worst_p   = pvals.max()\n'
    '        if verbose:\n'
    "            print(f'Iter {iteration}: {len(current_covs)} vars | '\n"
    "                  f'Pior: {worst_var} (p={worst_p:.4f})')\n"
    '        if worst_p > threshold:\n'
    '            current_covs.remove(worst_var)\n'
    '            iteration += 1\n'
    '        else:\n'
    '            if verbose:\n'
    "                print(f'\\n✓ Convergido em {iteration} iterações.')\n"
    "                print(f'Variáveis finais ({len(current_covs)}): {current_covs}')\n"
    '            return model, current_covs\n'
    '    return model, current_covs\n'
    '\n'
    "n_events = df_cox['event'].sum()\n"
    'max_vars = max(2, int(n_events / 10))\n'
    "print(f'Eventos: {n_events} → Máximo recomendado de variáveis: {max_vars}')\n"
    '\n'
    'threshold = 0.05 if n_events >= 50 else 0.10\n'
    "print(f'Threshold de p-value: {threshold}')\n"
    '\n'
    'ctv_final, final_covs = backward_cox_tdv(\n'
    "    df_fit, id_col='player_id', start_col='start',\n"
    "    stop_col='stop', event_col='event', covariates=cov_cols,\n"
    '    threshold=threshold\n'
    ')\n'
    '\n'
    'if len(final_covs) > max_vars:\n'
    "    print(f'\\n⚠️ Muitas variáveis ({len(final_covs)}) para {n_events} eventos.')\n"
    "    print(f'Reduzindo para top {max_vars} por menor p-value...')\n"
    "    top_by_p = ctv_final.summary['p'].nsmallest(max_vars).index.tolist()\n"
    '    final_covs = top_by_p\n'
    "    cols_refit = ['player_id', 'start', 'stop', 'event'] + final_covs\n"
    '    ctv_final = CoxTimeVaryingFitter(penalizer=0.0)\n'
    "    ctv_final.fit(df_fit[cols_refit], id_col='player_id', start_col='start',\n"
    "                  stop_col='stop', event_col='event', show_progress=False)\n"
    '\n'
    "print(f'\\nModelo parcimonioso — {POSICAO}')\n"
    'ctv_final.print_summary()\n'
    "hr_df = ctv_final.summary[['coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%', 'p']]\n"
    "hr_df.columns = ['coef', 'HR', 'se', 'CI_lower', 'CI_upper', 'p_value']\n"
    "hr_df.index.name = 'covariate'\n"
    "hr_df.to_csv(COX_DIR / 'resultados' / f'cox_hr_{POSICAO}.csv')\n"
    "print(f'HRs exportados para cox_hr_{POSICAO}.csv')\n"
    '\n'
    '# ---- bloco 10 ----\n'
    "smry = ctv_final.summary.copy()\n"
    "plot_cox_forest_thesis(smry, POSICAO, COX_DIR / 'figuras' / f'forest_plot_{POSICAO}.pdf', cores=CORES)\n"
    '\n'
    '# ---- bloco 12 ----\n'
    '# Schoenfeld deve usar o modelo EXATO dos HRs (sem flags _missing que caíram no backward)\n'
    'final_covs_schoenfeld = list(final_covs)\n'
    'final_covs = force_missing_flags(final_covs, df_model.columns.tolist())\n'
    "schoenfeld_input = df_model[['player_id', 'start', 'stop', 'event'] + final_covs_schoenfeld].copy()\n"
    "schoenfeld_csv = COX_DIR / 'resultados' / f'schoenfeld_{POSICAO}.csv'\n"
    "schoenfeld_input_csv = COX_DIR / 'resultados' / f'schoenfeld_input_{POSICAO}.csv'\n"
    "schoenfeld_plot = COX_DIR / 'figuras' / f'schoenfeld_{POSICAO}.pdf'\n"
    "schoenfeld_script = INFER_DIR / 'shared' / 'run_schoenfeld_diagnostics.R'\n"
    "schoenfeld_unpen_csv = COX_DIR / 'resultados' / f'coxph_unpenalized_{POSICAO}.csv'\n"
    '\n'
    'schoenfeld_input.to_csv(schoenfeld_input_csv, index=False)\n'
    'cmd = [\n'
    "    'Rscript',\n"
    '    str(schoenfeld_script),\n'
    '    str(schoenfeld_input_csv),\n'
    '    str(schoenfeld_csv),\n'
    '    str(schoenfeld_plot),\n'
    "    ','.join(final_covs_schoenfeld),\n"
    '    POSICAO,\n'
    '    str(schoenfeld_unpen_csv),\n'
    ']\n'
    '\n'
    'completed = subprocess.run(cmd, capture_output=True, text=True, check=False)\n'
    'if completed.stdout.strip():\n'
    '    print(completed.stdout)\n'
    'if completed.stderr.strip():\n'
    '    print(completed.stderr)\n'
    '\n'
    'ph_smry = pd.read_csv(schoenfeld_csv)\n'
    "print('Resumo de proporcionalidade (cox.zph):')\n"
    "print(ph_smry[['term', 'chisq', 'p', 'interpretacao']].sort_values('p').to_string(index=False))\n"
    '\n'
    "violacoes = ph_smry.loc[(ph_smry['term'] != 'GLOBAL') & (ph_smry['violacao']), 'term'].tolist()\n"
    'if violacoes:\n'
    "    print(f'\\nVariáveis com violação: {violacoes}')\n"
    'else:\n'
    "    print('\\nNenhuma violação detectada entre as covariáveis individuais.')\n"
    '\n'
    "if (ph_smry['term'] == 'GLOBAL').any():\n"
    "    global_row = ph_smry.loc[ph_smry['term'] == 'GLOBAL'].iloc[0]\n"
    '    print(f"Teste global: chisq={global_row[\'chisq\']:.3f} | p={global_row[\'p\']:.4f}")\n'
    '\n'
    '# ---- bloco 14 ----\n'
    'll_full = ctv_full.log_likelihood_\n'
    'll_pars = ctv_final.log_likelihood_\n'
    '# Graus de liberdade pelo numero REAL de coeficientes de cada ajuste.\n'
    '# (antes: contagem pos-filtro com modelo pre-filtro -> df subestimado.)\n'
    'df_full = int(len(ctv_full.params_))\n'
    'df_pars = int(len(ctv_final.params_))\n'
    'df_diff = df_full - df_pars\n'
    '\n'
    'if df_diff > 0:\n'
    '    lr_stat = -2 * (ll_pars - ll_full)\n'
    '    p_lr    = scipy_stats.chi2.sf(lr_stat, df=df_diff)\n'
    "    print(f'LR test — χ²({df_diff}) = {lr_stat:.2f}, p = {p_lr:.4f}')\n"
    '    if p_lr > 0.05:\n'
    "        print('→ Modelo parcimonioso NÃO piora significativamente (p > 0.05)')\n"
    '    else:\n'
    "        print('→ Modelo parcimonioso piora significativamente — considerar revisar corte de p-value')\n"
    'else:\n'
    "    print('Modelo parcimonioso tem mesmo número de variáveis que o completo.')\n"
    '\n'
    "lr_stat_out = -2.0 * (ll_pars - ll_full) if df_diff > 0 else float('nan')\n"
    "lr_p_out = float(scipy_stats.chi2.sf(lr_stat_out, df=df_diff)) if df_diff > 0 else float('nan')\n"
    'lr_tbl = pd.DataFrame([{\n'
    "    'posicao': POSICAO, 'll_saturado': float(ll_full), 'll_final': float(ll_pars),\n"
    "    'n_coef_saturado': df_full, 'n_coef_final': df_pars, 'df_diff': int(df_diff),\n"
    "    'lr_chisq': float(lr_stat_out), 'lr_p': lr_p_out,\n"
    '}])\n'
    "lr_tbl.to_csv(COX_DIR / 'resultados' / f'lr_test_{POSICAO}.csv', index=False)\n"
    "print(f'lr_test_{POSICAO}.csv gravado')\n"
    '\n'
    '# ---- bloco 16 ----\n'
    "print(f'=== RESUMO DO MODELO FINAL — {POSICAO} ===')\n"
    "print(f'Variáveis selecionadas: {len(final_covs)}')\n"
    "print(f'Log-likelihood: {ctv_final.log_likelihood_:.4f}')\n"
    "print(f'\\nVariáveis finais: {final_covs}')\n"
    '\n'
    'import json\n'
    "with open(COX_DIR / 'resultados' / f'final_covs_{POSICAO}.json', 'w') as f:\n"
    '    # Grava o conjunto EFETIVAMENTE ajustado (sem as dummies _missing, que o\n'
    '    # stepwise descartou). Antes o arquivo recebia as flags forcadas e\n'
    '    # contradizia cox_hr_{POS}.csv e a descricao da tese.\n'
    '    json.dump(list(final_covs_schoenfeld), f)\n'
    '    f.flush()\n'
    "print(f'\\nVariáveis salvas em final_covs_{POSICAO}.json')\n"
    '\n'
    'if n_events < 30:\n'
    "    print(f'\\n⚠️ Poucos eventos ({n_events}) — resultados devem ser interpretados com cautela')\n"
)


def main() -> None:
    namespace = globals()
    namespace["__name__"] = "__passo__"
    namespace["__file__"] = str(SCRIPT_PATH)

    import sys as _sys
    _sys.path.insert(0, str(ROOT / "Predição/shared"))
    from thesis_figure_style import apply_thesis_rcparams, plot_cox_forest_thesis, save_thesis_figure, figsize_full
    apply_thesis_rcparams()
    namespace["plot_cox_forest_thesis"] = plot_cox_forest_thesis
    namespace["save_thesis_figure"] = save_thesis_figure
    namespace["figsize_full"] = figsize_full
    exec(compile(CODIGO_PASSO, str(SCRIPT_PATH), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
