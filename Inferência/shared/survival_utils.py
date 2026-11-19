import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler


RATE_KEYWORDS = (
    "comp_pct",
    "ypa",
    "int_pct",
    "td_pct",
    "ypc",
    "yptarget",
    "ypr",
    "catch_rate",
    "adot_acc",
    "rush_td_pct",
    "yptouch",
    "reception_rate",
)

REPRESENTATIVE_COLUMNS = {
    "player_name",
    "position",
    "birth_year",
    "draft_year",
    "draft_round",
    "draft_pick",
    "draft_ovr",
    "height",
    "weight",
    "college",
    "season_type",
    "team",
    "conference",
    "division",
    "depth_team",
    "games_missed",
    "offense_pct",
    "age",
    "years_exp",
    "games_played_season",
    "games_played_career",
    "passer_rating",
    "adot",
    "comp_pct",
    "int_pct",
    "pass_td_pct",
    "ypa",
    "rec_td_pct",
    "yptarget",
    "ayptarget",
    "ypr",
    "rush_td_pct",
    "ypc",
    "td_pct",
    "yptouch",
}

AGG_MAX_PREFIXES = ("season_", "career_", "delta_")
AGG_RATE_PREFIXES = ("team_",)
AGG_NUMERIC_EXCLUDE = {"player_id", "season"}
MODEL_EPSILON = 1e-3


def _compute_nfl_passer_rating(attempts, completions, yards, touchdowns, interceptions):
    attempts = pd.to_numeric(attempts, errors="coerce")
    completions = pd.to_numeric(completions, errors="coerce")
    yards = pd.to_numeric(yards, errors="coerce")
    touchdowns = pd.to_numeric(touchdowns, errors="coerce")
    interceptions = pd.to_numeric(interceptions, errors="coerce")

    rating = pd.Series(np.nan, index=attempts.index, dtype=float)
    valid = attempts > 0
    if not valid.any():
        return rating

    comp_term = ((completions[valid] / attempts[valid]) - 0.3) * 5
    ypa_term = ((yards[valid] / attempts[valid]) - 3) * 0.25
    td_term = (touchdowns[valid] / attempts[valid]) * 20
    int_term = 2.375 - (interceptions[valid] / attempts[valid]) * 25

    components = [comp_term, ypa_term, td_term, int_term]
    components = [component.clip(lower=0, upper=2.375) for component in components]
    rating.loc[valid] = ((components[0] + components[1] + components[2] + components[3]) / 6) * 100
    return rating.round(2)


def sanitize_derived_metrics(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    out = df.copy()

    if "games_missed" in out.columns:
        numeric = pd.to_numeric(out["games_missed"], errors="coerce")
        neg_count = (numeric < 0).sum()
        if neg_count > 0 and verbose:
            warnings.warn(
                f"sanitize_derived_metrics: {neg_count} linhas com games_missed < 0 "
                f"(inconsistencia de fonte); clipados para 0."
            )
        out["games_missed"] = numeric.clip(lower=0)

    if {
        "season_pass_attempts",
        "season_complete_pass",
        "season_passing_yards",
        "season_pass_touchdown",
        "season_interception",
    }.issubset(out.columns):
        old_rating = out.get("passer_rating", pd.Series(dtype=float))
        new_rating = _compute_nfl_passer_rating(
            out["season_pass_attempts"],
            out["season_complete_pass"],
            out["season_passing_yards"],
            out["season_pass_touchdown"],
            out["season_interception"],
        )
        if verbose and old_rating is not None and len(old_rating) > 0:
            diff = (old_rating - new_rating).abs()
            big_diff = (diff > 0.5).sum()
            if big_diff > 0:
                warnings.warn(
                    f"sanitize_derived_metrics: {big_diff} linhas com discrepancia "
                    f"> 0.5 no passer_rating; recalculado pela formula oficial da NFL."
                )
            became_nan = (old_rating.notna() & new_rating.isna()).sum()
            if became_nan > 0:
                warnings.warn(
                    f"sanitize_derived_metrics: {became_nan} linhas com passer_rating "
                    f"alterado para NaN (0 pass attempts); mais honesto que 0.0."
                )
        out["passer_rating"] = new_rating

    additive_components = {
        "touches": ("pass_attempts", "rush_attempts", "receptions"),
        "total_yards": ("passing_yards", "rushing_yards", "receiving_yards"),
    }
    for metric, components in additive_components.items():
        for level in ("season", "career"):
            target = f"{level}_{metric}"
            if target not in out.columns:
                continue
            component_columns = [f"{level}_{component}" for component in components]
            if not set(component_columns).issubset(out.columns):
                out[target] = np.nan
                continue
            numeric_components = out[component_columns].apply(pd.to_numeric, errors="coerce")
            complete = np.isfinite(numeric_components.to_numpy(dtype=float)).all(axis=1)
            reconstructed = numeric_components.sum(axis=1, min_count=len(components))
            out[target] = reconstructed.where(complete)

    if "offense_pct" in out.columns:
        out["offense_pct"] = pd.to_numeric(out["offense_pct"], errors="coerce").clip(lower=0, upper=1)

    # Team shares: clip to [0, 1] (negative shares from sack yardage are real but conceptually 0)
    for col in out.columns:
        if col.startswith("team_") and ("_share" in col or "_pct" in col):
            out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=0, upper=1)

    return out


def get_rate_like_columns(columns):
    return [c for c in columns if any(key in c for key in RATE_KEYWORDS)]


def _first_non_null(series):
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else np.nan


def _representative_row(group: pd.DataFrame) -> pd.Series:
    sort_cols = []
    ascending = []
    for col in ["offense_snaps", "team_offense_snaps", "games_played_season"]:
        if col in group.columns:
            sort_cols.append(col)
            ascending.append(False)
    if sort_cols:
        return group.sort_values(sort_cols, ascending=ascending, na_position="last").iloc[0]
    return group.iloc[0]


def aggregate_player_season(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out_rows = []
    group_cols = ["player_id", "season"]

    for (_, _), group in df.groupby(group_cols, sort=False, observed=True):
        rep = _representative_row(group).copy()
        row = {"player_id": rep["player_id"], "season": rep["season"]}

        if "team" in group.columns:
            row["season_stints"] = int(group["team"].fillna("__nan__").nunique())
            row["multi_team_season_flag"] = int(row["season_stints"] > 1)
        else:
            row["season_stints"] = int(len(group))
            row["multi_team_season_flag"] = int(len(group) > 1)

        for col in df.columns:
            if col in group_cols:
                continue

            series = group[col]

            if col in REPRESENTATIVE_COLUMNS or col.startswith(AGG_RATE_PREFIXES) or any(
                key in col for key in RATE_KEYWORDS
            ):
                row[col] = rep[col] if pd.notna(rep[col]) else _first_non_null(series)
                continue

            if col.startswith(AGG_MAX_PREFIXES):
                # season_*, career_*, delta_*: todos usam max().
                # O CSV fonte ja preenche season_* com o total completo da temporada
                # em todas as linhas do jogador (inclusive multi-time).
                # Verificado empiricamente: 100% das colunas season_* tem valores
                # identicos entre times (auditoria maio/2026).
                if pd.api.types.is_numeric_dtype(series):
                    numeric = pd.to_numeric(series, errors="coerce")
                    row[col] = numeric.max(skipna=True)
                else:
                    row[col] = _first_non_null(series)
                continue

            if pd.api.types.is_numeric_dtype(series) and col not in AGG_NUMERIC_EXCLUDE:
                numeric = pd.to_numeric(series, errors="coerce")
                row[col] = numeric.sum(min_count=1)
            else:
                row[col] = rep[col] if pd.notna(rep[col]) else _first_non_null(series)

        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    if "multi_team_season_flag" in out.columns:
        n_multi = (out["multi_team_season_flag"] == 1).sum()
        if n_multi > 0:
            print(f"aggregate_player_season: {n_multi} jogador-temporadas multi-time detectados "
                  f"(entre {out['player_id'].nunique()} jogadores). Usando max() para season_*/career_*/delta_*.")
    out = out.sort_values(["player_id", "season"]).reset_index(drop=True)
    return out


def imputa_years_exp(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche `years_exp` ausente pela sequencia do proprio jogador.

    Na fonte, o incremento de `years_exp` entre duas linhas do mesmo jogador e igual ao
    incremento da temporada em 100% dos casos, de modo que `years_exp - season` e constante
    por jogador. A linha sem `years_exp` (16 linhas da temporada regular de 2016) recebe
    `season` mais essa constante; sem nenhuma outra linha do jogador, usa `season - draft_year`.
    Descartar essas linhas criaria saidas falsas (o jogador pareceria sair na temporada anterior).
    """
    out = df.copy()
    base = (out["years_exp"] - out["season"]).groupby(out["player_id"]).transform("median")
    falta = out["years_exp"].isna() & base.notna()
    out.loc[falta, "years_exp"] = out.loc[falta, "season"] + base[falta]
    if "draft_year" in out.columns:
        sem_ref = out["years_exp"].isna() & out["draft_year"].notna()
        out.loc[sem_ref, "years_exp"] = (out.loc[sem_ref, "season"] - out.loc[sem_ref, "draft_year"]).clip(lower=0)
    return out


def load_clean_position_data(root, position, season_type="REG"):
    df_raw = pd.read_csv(root / "DataBase" / "yearly_player_stats_offense.csv")
    reg = imputa_years_exp(df_raw[df_raw["season_type"] == season_type])
    # Ultima temporada do jogador na liga, em QUALQUER posicao: quem muda de posicao (por exemplo,
    # de QB para TE) nao sai da liga e deve ser censurado, nao contado como saida.
    ultima_na_liga = reg.groupby("player_id", observed=True)["season"].max()
    df = reg[reg["position"] == position].copy()
    df = aggregate_player_season(df)
    df = sanitize_derived_metrics(df)
    df["_ultima_temporada_liga"] = df["player_id"].map(ultima_na_liga)
    return df


def add_gap_aware_survival_structure(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.sort_values(["player_id", "season"]).reset_index(drop=True).copy()
    n_before = len(out)
    out = out.dropna(subset=["years_exp"]).copy()
    n_lost = n_before - len(out)
    if n_lost > 0:
        lost_players = df.loc[df["years_exp"].isna(), "player_id"].nunique()
        print(f"add_gap_aware_survival_structure: {n_lost} linhas removidas por years_exp NaN "
              f"({lost_players} jogadores).")
    out["prev_season"] = out.groupby("player_id", observed=True)["season"].shift(1)
    out["next_season"] = out.groupby("player_id", observed=True)["season"].shift(-1)
    out["gap_years_prev"] = (
        pd.to_numeric(out["season"], errors="coerce")
        - pd.to_numeric(out["prev_season"], errors="coerce")
        - 1
    )
    out["gap_years_prev"] = out["gap_years_prev"].clip(lower=0).fillna(0).astype(int)
    out["within_window_gap_years_before_t"] = out.groupby("player_id", observed=True)["gap_years_prev"].cumsum()
    out["observed_seasons_before_t"] = out.groupby("player_id", observed=True).cumcount()
    out["first_observed_season"] = (out["observed_seasons_before_t"] == 0).astype(int)

    # Anos de carreira que antecedem a 1a linha observada e nao tem estatistica no painel
    # (years_exp conta as temporadas desde o inicio da carreira; subtrai-se as linhas ja vistas
    # e os hiatos internos). Nao e um indicador de "antes de 2012": o painel comeca em 2012, e um
    # jogador draftado em 2012 que so aparece em 2013 tem 1 ano sem linha (a temporada de estreia).
    out["unobs_career_years"] = (
        pd.to_numeric(out["years_exp"], errors="coerce")
        - out["observed_seasons_before_t"]
        - out["within_window_gap_years_before_t"]
    ).clip(lower=0)
    # Veterano pre-2012: a carreira comecou antes de 2012. O inicio da carreira e
    # season - years_exp (years_exp e a experiencia na temporada, e a diferenca e constante por
    # jogador) e tambem o ano do draft, quando existe. Usa-se o menor dos dois.
    inicio_carreira = pd.to_numeric(out["season"], errors="coerce") - pd.to_numeric(out["years_exp"], errors="coerce")
    if "draft_year" in out.columns:
        inicio_carreira = np.fmin(inicio_carreira, pd.to_numeric(out["draft_year"], errors="coerce"))
    out["pre_window_veteran"] = (pd.Series(inicio_carreira, index=out.index) < 2012).astype(int)
    out["unobserved_history_flag"] = (
        (out["first_observed_season"] == 1) & (out["unobs_career_years"] > 0)
    ).astype(int)
    out["had_prior_gap_flag"] = (out["within_window_gap_years_before_t"] > 0).astype(int)

    out["event"] = out.groupby("player_id", observed=True)["season"].shift(-1).isna().astype(int)
    if "_ultima_temporada_liga" in out.columns:
        # Ultima linha na posicao, mas o jogador seguiu na liga em outra posicao: censura.
        out.loc[out["season"] < out["_ultima_temporada_liga"], "event"] = 0
        out = out.drop(columns=["_ultima_temporada_liga"])
    out.loc[out["season"] == 2024, "event"] = 0

    out["start"] = pd.to_numeric(out["years_exp"], errors="coerce").clip(lower=0)
    out["stop"] = out["start"] + 1
    out["entry"] = out.groupby("player_id", observed=True)["start"].transform("min")

    out = out.drop(columns=["prev_season", "next_season"])
    return out


def build_residual_snapshot_panel(df_cox: pd.DataFrame, covariates: list[str] | None = None, meta_cols: list[str] | None = None):
    covariates = covariates or []
    meta_cols = meta_cols or []

    panel = df_cox.sort_values(["player_id", "season", "stop"]).copy()
    final_state = (
        panel.groupby("player_id", observed=True)
        .agg(
            final_stop=("stop", "last"),
            final_event=("event", "last"),
            final_season=("season", "last"),
        )
        .reset_index()
    )
    panel = panel.merge(final_state, on="player_id", how="left")
    panel["residual_duration"] = (
        pd.to_numeric(panel["final_stop"], errors="coerce") - pd.to_numeric(panel["stop"], errors="coerce")
    )
    panel["residual_event"] = pd.to_numeric(panel["final_event"], errors="coerce").fillna(0).astype(int)
    panel["residual_duration_model"] = panel["residual_duration"]
    event_zero_mask = (panel["residual_event"] == 1) & (panel["residual_duration_model"] <= 0)
    panel.loc[event_zero_mask, "residual_duration_model"] = MODEL_EPSILON

    keep_cols = [
        "player_id",
        "player_name",
        "season",
        "stop",
        "event",
        "residual_duration",
        "residual_duration_model",
        "residual_event",
    ]
    keep_cols.extend(covariates)
    keep_cols.extend(meta_cols)
    keep_cols = list(dict.fromkeys([col for col in keep_cols if col in panel.columns]))
    out = panel[keep_cols].copy()
    out = out.rename(columns={"season": "snapshot_season", "stop": "snapshot_stop"})
    return out.sort_values(["player_id", "snapshot_season"]).reset_index(drop=True)


def fit_imputation_stats(df, covariates, group_col="years_exp"):
    stats = {
        "group_col": group_col,
        "has_group_col": group_col in df.columns,
        "rate_cols": [],
        "global_medians": {},
        "group_medians": {},
    }

    for col in covariates:
        if col not in df.columns or col.endswith("_missing"):
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        # Para draft_ovr, usar mediana apenas dos draftados se undrafted_flag existir
        if col == "draft_ovr" and "undrafted_flag" in df.columns:
            drafted_only = series[df["undrafted_flag"] == 0]
            stats["global_medians"][col] = drafted_only.median()
        else:
            stats["global_medians"][col] = series.median()

        if any(key in col for key in RATE_KEYWORDS):
            stats["rate_cols"].append(col)
            if stats["has_group_col"]:
                grouped = df[[group_col, col]].copy()
                grouped[col] = pd.to_numeric(grouped[col], errors="coerce")
                stats["group_medians"][col] = grouped.groupby(group_col)[col].median()

    return stats


def apply_imputation_stats(df, covariates, stats, preserve_missing_flags=True):
    out = df.copy()
    group_col = stats["group_col"]

    for col in covariates:
        if col not in out.columns or col.endswith("_missing"):
            continue

        if preserve_missing_flags:
            miss_col = f"{col}_missing"
            if miss_col in out.columns:
                out[miss_col] = out[miss_col].fillna(out[col].isna().astype(int))

        if col in stats["rate_cols"] and stats.get("has_group_col") and group_col in out.columns:
            mapped = out[group_col].map(stats["group_medians"].get(col, pd.Series(dtype=float)))
            out[col] = out[col].fillna(mapped)

        out[col] = out[col].fillna(stats["global_medians"].get(col))

    return out


def fit_scaler(train_df, covariates):
    scaler = StandardScaler()
    scaler.fit(train_df[covariates])
    return scaler


def apply_scaler(df, scaler, covariates):
    out = df.copy()
    out[covariates] = scaler.transform(out[covariates])
    return out


def temporal_player_splits(df, season_col, min_train_players=80, min_test_players=20):
    seasons = sorted(df[season_col].dropna().unique())
    splits = []

    for season in seasons[1:]:
        train_idx = df.index[df[season_col] < season].to_numpy()
        test_idx = df.index[df[season_col] == season].to_numpy()
        if len(train_idx) >= min_train_players and len(test_idx) >= min_test_players:
            splits.append((int(season), train_idx, test_idx))

    return splits


def filter_collinear_covariates(df, covariates, threshold=0.85, verbose=True):
    """Remove covariáveis com correlação absoluta acima do threshold.

    Para cada par de covariáveis com |correlação| > threshold, remove a que tem
    maior número de correlações acima do threshold (ou seja, a mais colinear).
    Mantém covariáveis com menor número de correlações extremas.
    
    Colunas _missing (flags de imputação) nunca são removidas — são metadados,
    não covariáveis substantivas, e sua preservação é essencial para transparência
    da imputação.
    """
    covs = [c for c in covariates if c in df.columns]
    if len(covs) < 2:
        return covs

    # Separar flags _missing — nunca podem ser removidas
    missing_flags = {c for c in covs if c.endswith("_missing")}
    substantive = [c for c in covs if c not in missing_flags]
    
    if len(substantive) < 2:
        return covs

    numeric = df[substantive].apply(pd.to_numeric, errors='coerce')
    corr_matrix = numeric.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)

    removed = []
    remaining = set(substantive)

    while True:
        high_corr_counts = {}
        for c in remaining:
            high_corr_counts[c] = sum(
                1 for other in remaining if other != c and corr_matrix.loc[c, other] > threshold
            )
        if max(high_corr_counts.values(), default=0) == 0:
            break
        to_remove = max(sorted(high_corr_counts), key=high_corr_counts.get)
        remaining.remove(to_remove)
        removed.append(to_remove)

    if verbose and removed:
        print(f'Removidas {len(removed)} covariáveis por colinearidade (|r| > {threshold}):')
        for c in removed:
            print(f'  {c}')
        print(f'Covariáveis restantes: {len(remaining) + len(missing_flags)}')

    return sorted(remaining) + sorted(missing_flags)


def force_missing_flags(selected_covariates, available_columns):
    """Garante que dummies _missing acompanhem suas respectivas variaveis de taxa.
    
    Se uma variavel como observed_career_lag_comp_pct foi selecionada e existe
    uma coluna observed_career_lag_comp_pct_missing nos dados, essa dummy eh
    automaticamente incluida no modelo — independentemente de ter passado
    pelo filtro de colinearidade ou stepwise.
    
    Isso garante transparencia: o modelo sabe quando um valor foi imputado.
    """
    result = list(selected_covariates)
    for cov in selected_covariates:
        miss_col = f"{cov}_missing"
        if miss_col in available_columns and miss_col not in result:
            result.append(miss_col)
    return result


def adaptive_rank_filter(df, covariates, start_col='start', stop_col='stop',
                         event_col='event', tol=1e-9, verbose=True):
    """Garante posto completo da matriz de covariáveis em todo tempo de evento.

    O estimador de Aalen requer que X_j tenha posto completo em cada t_j.
    Esta função verifica iterativamente o posto da matriz de covariáveis no
    conjunto de risco de cada tempo de evento. Se houver deficiência,
    remove a covariável cuja exclusão mais frequentemente restaura o posto
    completo. A identificação é feita por dois mecanismos:

    1. Colunas constantes no conjunto de risco (nunique <= 1) são
       penalizadas proporcionalmente.
    2. Se não houver colunas estritamente constantes, testa-se a remoção
       temporária de cada coluna; a que mais frequentemente restaura o
       posto quando removida é escolhida.

    Em caso de empate, desempata-se pela menor variância global da covariável.

    Referências:
    - Martinussen & Scheike (2006), Dynamic Regression Models for Survival
      Data, Cap. 5 (suposições regulares R1-R5).
    - Huffer & McKeague (1991), JASA 86(413), 114-129.
    """
    covs = [c for c in covariates if c in df.columns]
    event_times = df.loc[df[event_col] == 1, stop_col].unique()
    event_times = np.sort(event_times)
    removed = []
    overall_vars = df[covs].var().to_dict()

    while True:
        constant_score = {c: 0 for c in covs}
        removal_score = {c: 0 for c in covs}
        any_deficiency = False
        n_deficient_times = 0

        for t in event_times:
            in_risk = df[(df[start_col] < t) & (df[stop_col] >= t)]
            if len(in_risk) == 0:
                continue
            X = in_risk[covs].values.astype(float)
            p = len(covs)
            n_obs = len(in_risk)
            # Se há menos observações que covariáveis, o posto naturalmente é < p.
            # Isso não é culpa de nenhuma covariável em particular; ignoramos.
            if n_obs < p:
                continue
            rank = np.linalg.matrix_rank(X, tol=tol)
            if rank < p:
                any_deficiency = True
                n_deficient_times += 1

                # Mecanismo 1: colunas constantes
                constant_here = []
                for i, c in enumerate(covs):
                    if np.allclose(X[:, i], X[0, i], atol=tol):
                        constant_here.append(c)
                if constant_here:
                    for c in constant_here:
                        constant_score[c] += 1
                else:
                    # Mecanismo 2: testar remoção temporária de cada coluna
                    for i, c in enumerate(covs):
                        X_minus = np.delete(X, i, axis=1)
                        rank_minus = np.linalg.matrix_rank(X_minus, tol=tol)
                        if rank_minus == p - 1:
                            # A remoção desta coluna restaurou posto completo
                            removal_score[c] += 1

        if not any_deficiency:
            break

        # Combinar scores: constant_score tem peso maior
        combined_score = {}
        for c in covs:
            # constant_score é mais informativo; removal_score é fallback
            combined_score[c] = constant_score[c] * 2 + removal_score[c]

        max_score = max(combined_score.values(), default=0)
        candidates = [c for c, s in combined_score.items() if s == max_score]

        if len(candidates) == 1:
            to_remove = candidates[0]
        else:
            # Desempate pela menor variância global
            to_remove = min(candidates, key=lambda c: overall_vars.get(c, np.inf))

        covs.remove(to_remove)
        removed.append(to_remove)
        del overall_vars[to_remove]

        if len(covs) == 0:
            raise ValueError(
                "Todas as covariáveis foram removidas pela filtragem de posto adaptativa."
            )

    if verbose and removed:
        print(f'Removidas {len(removed)} covariáveis por deficiência de posto '
              f'em {n_deficient_times} tempo(s) de evento:')
        for c in removed:
            print(f'  {c}')
        print(f'Covariáveis restantes: {len(covs)}')

    return covs, removed



class ResultadoLogrank:
    """Resultado no formato usado pelo pipeline (test_statistic, p_value, degrees_of_freedom)."""

    def __init__(self, estatistica: float, gl: int, p: float):
        self.test_statistic = estatistica
        self.degrees_of_freedom = gl
        self.p_value = p


def logrank_entrada_tardia(duration, groups, event_observed, entry=None) -> ResultadoLogrank:
    """Log-rank de k grupos com entrada tardia (truncamento a esquerda).

    O jogador so entra no conjunto de risco depois da propria entrada: em cada tempo de
    evento t, estao em risco os que tem entry < t <= duration. E o teste que corresponde
    as curvas de Kaplan-Meier desenhadas com `entry=` (Klein e Moeschberger, 2003, cap. 4 e
    sec. 7.3). Sem `entry` (ou com entry = 0 para todos) reproduz o `multivariate_logrank_test`
    do lifelines.
    """
    from scipy.stats import chi2

    t = np.asarray(duration, dtype=float)
    e = np.asarray(event_observed, dtype=int)
    g = np.asarray(pd.Series(groups).astype(str))
    a = np.zeros_like(t) if entry is None else np.asarray(entry, dtype=float)
    rotulos = np.unique(g)
    k = len(rotulos)
    if k < 2:
        raise ValueError("log-rank exige pelo menos dois grupos")
    indicador = np.stack([g == r for r in rotulos], axis=1).astype(float)
    U = np.zeros(k)
    V = np.zeros((k, k))
    for tempo in np.unique(t[e == 1]):
        em_risco = (a < tempo) & (t >= tempo)
        n = em_risco.sum()
        if n == 0:
            continue
        mortos = em_risco & (t == tempo) & (e == 1)
        d = mortos.sum()
        n_g = indicador[em_risco].sum(axis=0)
        d_g = indicador[mortos].sum(axis=0)
        U += d_g - d * n_g / n
        if n > 1:
            fator = d * (n - d) / (n - 1)
            prop = n_g / n
            V += fator * (np.diag(prop) - np.outer(prop, prop))
    Ur, Vr = U[:-1], V[:-1, :-1]
    estatistica = float(Ur @ np.linalg.solve(Vr, Ur))
    gl = k - 1
    return ResultadoLogrank(estatistica, gl, float(chi2.sf(estatistica, gl)))


# Colunas que nao sao identificaveis no Cox com tempo em anos de carreira (start = years_exp):
# - years_exp e o proprio eixo do tempo, constante dentro de cada conjunto de risco;
# - first_observed_season e duplicata de unobserved_history_flag DENTRO de cada conjunto de risco:
#   para tempos > 1 as duas colunas sao iguais em todas as linhas (FOS = flag = 1 so na entrada com
#   anos de carreira sem linha), e no tempo 1 FOS e constante (todas as linhas tem start = 0).
#   A correlacao maxima de first_observed_season com as demais colunas e 0,80 no QB (flags _missing),
#   abaixo de 0,85; ainda assim o Cox sem penalizacao com as duas colunas e singular
#   (LinAlgError em CoxTimeVaryingFitter). Mantem-se a flag, que tem definicao.
COLUNAS_NAO_IDENTIFICAVEIS_COX = ("years_exp", "first_observed_season")
MOTIVOS_NAO_IDENTIFICAVEIS_COX = {
    "years_exp": "eixo do tempo (years_exp)",
    "first_observed_season": "duplicata de unobserved_history_flag no conjunto de risco",
}


def covariaveis_identificaveis_cox(df, covariates, verbose=True):
    """Remove do conjunto candidato do Cox as colunas sem efeito identificavel.

    Alem das colunas de COLUNAS_NAO_IDENTIFICAVEIS_COX, remove colunas identicas a outra ja
    mantida (por exemplo, indicadores `_missing` com o mesmo denominador). Sem essa limpeza a
    matriz de informacao do Cox sem penalizacao e singular.
    """
    mantidas, removidas = [], []
    for col in covariates:
        if col in COLUNAS_NAO_IDENTIFICAVEIS_COX:
            removidas.append((col, MOTIVOS_NAO_IDENTIFICAVEIS_COX[col]))
            continue
        serie = pd.to_numeric(df[col], errors="coerce").to_numpy()
        duplicada = next((m for m in mantidas
                          if np.array_equal(pd.to_numeric(df[m], errors="coerce").to_numpy(), serie, equal_nan=True)), None)
        if duplicada is not None:
            removidas.append((col, f"identica a {duplicada}"))
            continue
        mantidas.append(col)
    if verbose and removidas:
        for col, motivo in removidas:
            print(f"covariaveis_identificaveis_cox: removida {col} ({motivo})")
    return mantidas
