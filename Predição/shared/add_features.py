#!/usr/bin/env python3
"""
Adiciona features lag-safe ao df_cox sem data leakage.
Todas as features usam informação disponível até t-1.

Features criadas:
- depth_worsened_lag: depth_team piorou em t-2→t-1? (Binária)
- offense_pct_declined_lag: offense_pct caiu em t-2→t-1? (Binária)
- games_missed_increased_lag: games_missed aumentou em t-2→t-1? (Binária)
- rookie_flag: Primeira temporada observada? (Binária)
- contract_year_flag: years_exp entre 2 e 4? (Binária)
- efficiency_declined_lag: Métrica de eficiência posição-específica caiu? (Binária)
- seasons_since_gap: Temporadas desde o último hiato (Inteira)
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PREP_DIR = ROOT / 'Inferência' / 'Preparação Modelagem' / 'resultados'
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


def lag_temporada(df, col, k):
    """Valor de `col` na temporada t-k do mesmo jogador.
    Defasagem por temporada (não por linha): NaN se o jogador não tem linha em t-k
    (hiato), o que segue a mesma convenção do primeiro ano (fillna(0) nas flags)."""
    chave = pd.MultiIndex.from_arrays([df['player_id'], df['season'] - k])
    return pd.Series(df.set_index(['player_id', 'season'])[col].reindex(chave).values, index=df.index)

for POS in ['QB', 'RB', 'WR']:
    print(f'=== {POS} ===')
    df = pd.read_pickle(PREP_DIR / f'df_cox_{POS}.pkl')
    
    # Sort for lag operations
    df = df.sort_values(['player_id', 'season']).reset_index(drop=True)
    
    # --- Feature 1: depth_worsened_lag ---
    # depth_team piorou da temporada t-2 para t-1? (shift(1) = valor em t-1)
    df['depth_lag1'] = lag_temporada(df, 'depth_team', 1)
    df['depth_lag2'] = lag_temporada(df, 'depth_team', 2)
    df['depth_worsened_lag'] = ((df['depth_lag1'] > df['depth_lag2']).fillna(0)).astype(int)
    df.drop(['depth_lag1', 'depth_lag2'], axis=1, inplace=True)
    
    # --- Feature 2: offense_pct_declined_lag ---
    df['off_pct_lag1'] = lag_temporada(df, 'offense_pct', 1)
    df['off_pct_lag2'] = lag_temporada(df, 'offense_pct', 2)
    df['offense_pct_declined_lag'] = ((df['off_pct_lag1'] < df['off_pct_lag2']).fillna(0)).astype(int)
    df.drop(['off_pct_lag1', 'off_pct_lag2'], axis=1, inplace=True)
    
    # --- Feature 3: games_missed_increased_lag ---
    df['gm_lag1'] = lag_temporada(df, 'games_missed', 1)
    df['gm_lag2'] = lag_temporada(df, 'games_missed', 2)
    df['games_missed_increased_lag'] = ((df['gm_lag1'] > df['gm_lag2']).fillna(0)).astype(int)
    df.drop(['gm_lag1', 'gm_lag2'], axis=1, inplace=True)
    
    # --- Feature 4: rookie_flag ---
    df['rookie_flag'] = (df['first_observed_season'] == 1).astype(int)
    
    # --- Feature 5: contract_year_flag ---
    # Anos 2-4: fim típico do contrato de rookie
    # contract_year_flag removido (importance zero em todos os modelos, auditoria maio/2026)
    df['contract_year_flag'] = 0
    
    # --- Feature 6: declining_efficiency_lag ---
    # Posição-específica: métrica de eficiência que caiu
    if POS == 'QB' and 'passer_rating' in df.columns:
        df['eff_lag1'] = lag_temporada(df, 'passer_rating', 1)
        df['eff_lag2'] = lag_temporada(df, 'passer_rating', 2)
        df['efficiency_declined_lag'] = ((df['eff_lag1'] < df['eff_lag2']).fillna(0)).astype(int)
        df.drop(['eff_lag1', 'eff_lag2'], axis=1, inplace=True)
    elif POS == 'RB' and 'observed_career_lag_ypc' in df.columns:
        df['eff_lag1'] = lag_temporada(df, 'observed_career_lag_ypc', 1)
        df['eff_lag2'] = lag_temporada(df, 'observed_career_lag_ypc', 2)
        df['efficiency_declined_lag'] = ((df['eff_lag1'] < df['eff_lag2']).fillna(0)).astype(int)
        df.drop(['eff_lag1', 'eff_lag2'], axis=1, inplace=True)
    elif POS == 'WR' and 'observed_career_lag_catch_rate' in df.columns:
        df['eff_lag1'] = lag_temporada(df, 'observed_career_lag_catch_rate', 1)
        df['eff_lag2'] = lag_temporada(df, 'observed_career_lag_catch_rate', 2)
        df['efficiency_declined_lag'] = ((df['eff_lag1'] < df['eff_lag2']).fillna(0)).astype(int)
        df.drop(['eff_lag1', 'eff_lag2'], axis=1, inplace=True)
    else:
        df['efficiency_declined_lag'] = 0
    
    # --- Feature 7: seasons_since_gap ---
    # Quantas temporadas consecutivas desde o ultimo gap?
    df['seasons_since_gap'] = 0
    for pid, grp in df.groupby('player_id'):
        grp = grp.sort_values('season')
        counter = 0
        for idx in grp.index:
            if df.at[idx, 'had_prior_gap_flag'] == 1:
                counter = 0
            df.at[idx, 'seasons_since_gap'] = counter
            counter += 1
    
    # Summary
    new_features = ['depth_worsened_lag', 'offense_pct_declined_lag', 
                    'games_missed_increased_lag', 'rookie_flag', 
                    'contract_year_flag', 'efficiency_declined_lag',
                    'seasons_since_gap']
    
    print(f'Novas features: {new_features}')
    for feat in new_features:
        if feat in df.columns:
            print(f'  {feat}: mean={df[feat].mean():.3f}, unique={df[feat].nunique()}')
    
    # Save
    out_path = OUT_DIR / f'df_cox_{POS}_features.pkl'
    df.to_pickle(out_path)
    print(f'Salvo: {out_path} ({df.shape})')
    print()

print('Feature engineering concluído!')
