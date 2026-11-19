#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript run_aalen_tied_inference.R <input_csv> <output_dir> <position> <covariates_comma_sep>")
}

input_csv <- args[[1]]
output_dir <- args[[2]]
position <- args[[3]]
covariates <- strsplit(args[[4]], ",", fixed = TRUE)[[1]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)
covariates <- sort(covariates[covariates %in% names(df)])
if (!length(covariates)) stop("Nenhuma covariavel valida.")
for (column in c("start", "stop", "event", covariates)) df[[column]] <- as.numeric(df[[column]])
df <- df[complete.cases(df[, c("player_id", "start", "stop", "event", covariates), drop = FALSE]), , drop = FALSE]
if (!nrow(df)) stop("Nenhuma linha completa para o ajuste.")
if (any(!is.finite(df$start)) || any(!is.finite(df$stop)) || any(df$start >= df$stop)) stop("Intervalos start-stop invalidos.")
if (any(!df$event %in% c(0, 1))) stop("event deve conter somente 0/1.")
if (any(abs(c(df$start, df$stop) - round(c(df$start, df$stop))) > 1e-8)) stop("O estimador sem jitter requer limites anuais inteiros.")

rank_at_time <- function(data, time, covs) {
  risk <- data$start < time & data$stop >= time
  if (!any(risk)) return(0L)
  design <- cbind(Intercept = 1, as.matrix(data[risk, covs, drop = FALSE]))
  qr(design, tol = 1e-10)$rank
}

select_supported_window <- function(data, covs) {
  nmin_default <- 3L * length(covs)
  event_times <- sort(unique(data$stop[data$event == 1]))
  if (!length(event_times)) stop("Nenhum evento no input.")
  grid_times <- seq(min(data$start) + 1, max(data$stop), by = 1)
  valid <- vapply(grid_times, function(time) {
    risk_n <- sum(data$start < time & data$stop >= time)
    risk_n > 0 && risk_n >= nmin_default && rank_at_time(data, time, covs) == length(covs) + 1L
  }, logical(1))
  runs <- list()
  current <- numeric()
  for (i in seq_along(grid_times)) {
    time <- grid_times[[i]]
    if (!valid[[i]]) {
      if (length(current)) runs[[length(runs) + 1L]] <- current
      current <- numeric()
      next
    }
    current <- c(current, time)
  }
  if (length(current)) runs[[length(runs) + 1L]] <- current
  if (!length(runs)) stop("Nenhum tempo de evento tem posto completo e suporte nmin padrao.")
  longest <- which.max(vapply(runs, length, integer(1)))
  chosen <- runs[[longest]]
  start <- chosen[[1]] - 1
  end <- tail(chosen, 1)
  selected_events <- event_times[event_times > start & event_times <= end]
  if (length(selected_events) < 2L) stop("A janela de posto/suporte contem menos de dois tempos de evento.")
  list(start = start, end = end, support_times = chosen, event_times = selected_events,
       nmin_default = nmin_default, selection = "longest consecutive integer risk-time grid run with full [Intercept + X] rank and risk_n >= 3*nvar; zero-risk/unsupported years break runs; earliest run wins ties")
}

rank_only_end <- function(data, covs, start) {
  grid_times <- seq(start + 1, max(data$stop), by = 1)
  valid <- vapply(grid_times, function(time) {
    risk_n <- sum(data$start < time & data$stop >= time)
    risk_n > 0 && rank_at_time(data, time, covs) == length(covs) + 1L
  }, logical(1))
  if (!length(valid) || !valid[[1]]) stop("A janela rank-only nao inicia no limite inferior da janela principal.")
  first_invalid <- which(!valid)[1]
  last_index <- if (is.na(first_invalid)) length(grid_times) else first_invalid - 1L
  end <- grid_times[[last_index]]
  event_times <- sort(unique(data$stop[data$event == 1 & data$stop > start & data$stop <= end]))
  if (length(event_times) < 2L) stop("A janela rank-only tem menos de dois tempos de evento.")
  list(start = start, end = end, support_times = grid_times[seq_len(last_index)],
       event_times = event_times,
       selection = "same lower boundary as primary; extend over consecutive integer years while [Intercept + X] risk rank remains full; no nmin requirement")
}

gap_fixture <- do.call(rbind, lapply(c(1, 2, 4, 5), function(start_year) {
  data.frame(player_id = paste0("g", start_year, "_", 1:3), start = start_year,
             stop = start_year + 1, event = 1, x = 0:2)
}))
gap_window <- select_supported_window(gap_fixture, "x")
if (gap_window$start != 1 || gap_window$end != 3 || !identical(gap_window$support_times, c(2, 3))) {
  stop("A grade anual nao separou corretamente a lacuna de risco sintetica.")
}

clip_window <- function(data, start, end) {
  original_stop <- data$stop
  overlaps <- data$start < end & data$stop > start
  clipped <- data[overlaps, , drop = FALSE]
  stop_before_clip <- original_stop[overlaps]
  clipped$start <- pmax(clipped$start, start)
  clipped$event <- ifelse(stop_before_clip > end, 0, clipped$event)
  clipped$stop <- pmin(stop_before_clip, end)
  clipped <- clipped[clipped$start < clipped$stop, , drop = FALSE]
  if (any(clipped$event == 1 & clipped$stop <= start)) stop("Evento na fronteira inferior entrou no ajuste.")
  if (any(clipped$stop > end)) stop("Intervalo ultrapassa a fronteira superior.")
  clipped
}

validate_risk_membership <- function(full, window_data, event_times) {
  for (time in event_times) {
    full_risk <- rownames(full)[full$start < time & full$stop >= time]
    window_risk <- rownames(window_data)[window_data$start < time & window_data$stop >= time]
    if (!identical(sort(full_risk), sort(window_risk))) stop(sprintf("Conjunto de risco mudou no tempo %g.", time))
    full_events <- rownames(full)[full$event == 1 & full$stop == time]
    window_events <- rownames(window_data)[window_data$event == 1 & window_data$stop == time]
    if (!identical(sort(full_events), sort(window_events))) stop(sprintf("Eventos empatados mudaram no tempo %g.", time))
  }
  invisible(TRUE)
}

json_write <- function(value, path) {
  encoded <- jsonlite::toJSON(value, pretty = TRUE, auto_unbox = TRUE, null = "null", na = "null", digits = 16)
  writeLines(as.character(encoded), path, useBytes = TRUE)
}

test_metadata <- list(
  test_family = "weighted_mean_aareg",
  test_scope = "weighted temporal mean",
  confidence_band_type = "pointwise_conditional",
  selection_conditioning = "conditional_X_selected_by_univariate_screening",
  covariate_selection_status = "X vem da triagem univariada (run_aalen_univariate_screening.R, semente 42, sem jitter); sem reselecao aqui; claims condicionais a este X fixo",
  test_description = "Teste robusto de coeficiente medio ponderado de survival::aareg restrito a janela; nao e teste supremo nem teste de constancia. Empates tratados nativamente pelo aareg (sem sorteio nem jitter); X condicional a triagem univariada."
)

fit_export <- function(data, covs, pos, start, end, nmin_policy, result_dir) {
  dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)
  # aareg agrupa as influencias do teste com o cluster fora da ordem interna (survival 3.5.8):
  # ordenar por (stop, -event) alinha os dois, e o p deixa de depender da ordem das linhas.
  data <- data[order(data$stop, -data$event), , drop = FALSE]
  formula <- as.formula(paste("Surv(start, stop, event) ~", paste(sprintf("`%s`", covs), collapse = " + ")))
  if (nmin_policy == "default") {
    fit <- survival::aareg(formula, data = data, cluster = player_id, dfbeta = TRUE, taper = 1)
    nmin_value <- 3L * length(covs)
  } else {
    fit <- survival::aareg(formula, data = data, cluster = player_id, dfbeta = TRUE, taper = 1, nmin = 1)
    nmin_value <- 1L
  }
  event_times <- sort(unique(as.numeric(fit$times)))
  requested_times <- sort(unique(data$stop[data$event == 1]))
  if (!identical(event_times, requested_times)) {
    stop(sprintf("aareg nao manteve todos os eventos da janela (%s; nmin=%d).", pos, nmin_value))
  }
  if (length(event_times) < 2L) stop("aareg requer pelo menos dois tempos de evento apoiados.")

  term_names <- c("Intercept", covs)
  n_terms <- length(term_names)
  beta_running <- setNames(numeric(n_terms), term_names)
  model_variance_running <- matrix(0, n_terms, n_terms, dimnames = list(term_names, term_names))
  player_ids <- unique(as.character(data$player_id))
  player_influence_running <- matrix(0, nrow = length(player_ids), ncol = n_terms,
                                     dimnames = list(player_ids, term_names))
  cumulative_rows <- list(c(start, beta_running))
  model_variance_rows <- list(c(start, rep(0, n_terms)))
  robust_variance_rows <- list(c(start, rep(0, n_terms)))
  event_audit <- list()
  max_beta_error <- 0
  max_robust_variance_error <- 0

  for (time in seq(start + 1, end, by = 1)) {
    risk <- data$start < time & data$stop >= time
    risk_data <- data[risk, , drop = FALSE]
    design <- cbind(Intercept = 1, as.matrix(risk_data[, covs, drop = FALSE]))
    event_vector <- as.numeric(risk_data$event == 1 & risk_data$stop == time)
    xtx <- crossprod(design)
    if (qr(design, tol = 1e-10)$rank != n_terms) stop(sprintf("Posto incompleto no tempo %g.", time))
    tied_events <- sum(event_vector)
    if (tied_events) {
      manual_delta <- solve(xtx, crossprod(design, event_vector))[, 1]
      names(manual_delta) <- term_names
      fit_rows <- which(as.numeric(fit$times) == time)
      if (length(fit_rows) != tied_events) stop(sprintf("Contagem de eventos empatados difere no tempo %g.", time))
      fit_risk <- unique(as.numeric(fit$nrisk[fit_rows]))
      if (length(fit_risk) != 1L || fit_risk != sum(risk)) stop(sprintf("Contagem de risco aareg difere no tempo %g.", time))
      fit_delta <- colSums(fit$coefficient[fit_rows, , drop = FALSE])
      beta_error <- max(abs(fit_delta[term_names] - manual_delta))
      max_beta_error <- max(max_beta_error, beta_error)
      if (!is.finite(beta_error) || beta_error > 1e-8) stop(sprintf("OLS conjunto nao reproduz aareg no tempo %g (erro %.3g).", time, beta_error))

      residual <- event_vector - drop(design %*% manual_delta)
      influence <- t(solve(xtx, t(design)))
      influence <- sweep(influence, 1, residual, `*`)
      dimnames(influence) <- list(NULL, term_names)
      model_variance_running <- model_variance_running + crossprod(influence)
      grouped_influence <- rowsum(influence, group = as.character(risk_data$player_id), reorder = FALSE)
      robust_year_manual <- crossprod(grouped_influence)
      player_influence_running[rownames(grouped_influence), ] <- player_influence_running[rownames(grouped_influence), , drop = FALSE] + grouped_influence
      robust_cumulative_manual <- crossprod(player_influence_running)

      time_index <- match(time, sort(unique(as.numeric(fit$times))))
      aareg_influence <- as.matrix(fit$dfbeta[, , time_index, drop = FALSE][, , 1])
      robust_year_aareg <- crossprod(aareg_influence)
      max_robust_variance_error <- max(max_robust_variance_error, max(abs(robust_year_manual - robust_year_aareg)))
      if (max(abs(robust_year_manual - robust_year_aareg)) > 1e-8) stop(sprintf("Variancia robusta anual diverge da influencia dfbeta no tempo %g.", time))
      beta_running <- beta_running + fit_delta[term_names]
    } else {
      beta_error <- 0
      robust_cumulative_manual <- crossprod(player_influence_running)
    }
    cumulative_rows[[length(cumulative_rows) + 1L]] <- c(time, beta_running)
    model_variance_rows[[length(model_variance_rows) + 1L]] <- c(time, diag(model_variance_running))
    robust_variance_rows[[length(robust_variance_rows) + 1L]] <- c(time, diag(robust_cumulative_manual))
    event_audit[[as.character(time)]] <- list(risk_n = sum(risk), tied_events = tied_events, risk_design_rank = n_terms,
                                                manual_ols_max_abs_difference = beta_error)
  }

  output_terms <- c("(Intercept)", covs)
  make_output_frame <- function(rows) {
    values <- do.call(rbind, lapply(rows, as.numeric))
    frame <- as.data.frame(values, check.names = FALSE)
    names(frame) <- c("time", output_terms)
    frame
  }
  cum_coefs <- make_output_frame(cumulative_rows)
  var_cum <- make_output_frame(model_variance_rows)
  robvar_cum <- make_output_frame(robust_variance_rows)
  write.csv(cum_coefs, file.path(result_dir, sprintf("aalen_cum_coefs_%s.csv", pos)), row.names = FALSE)
  write.csv(var_cum, file.path(result_dir, sprintf("aalen_var_cum_%s.csv", pos)), row.names = FALSE)
  write.csv(robvar_cum, file.path(result_dir, sprintf("aalen_robvar_cum_%s.csv", pos)), row.names = FALSE)
  saveRDS(fit, file.path(result_dir, sprintf("aalen_fit_%s.rds", pos)))

  tests <- summary(fit)$table
  tests_null_effect <- list()
  for (i in seq_len(nrow(tests))) {
    term <- rownames(tests)[[i]]
    output_term <- if (term == "Intercept") "(Intercept)" else term
    p_value <- as.numeric(tests[i, "p"])
    tests_null_effect[[output_term]] <- list(
      p_value = if (is.finite(p_value)) round(p_value, 6) else NA_real_,
      significant = is.finite(p_value) && p_value < 0.05
    )
  }
  primary_summary <- list(
    janela_inicio = start,
    janela_fim = end,
    position = pos,
    n_subjects = length(unique(data$player_id)),
    n_periods = nrow(data),
    n_events = sum(data$event == 1),
    covariates_tested = covs,
    model = "aalen_tied_survival_aareg",
    estimator = "survival::aareg(Surv(start,stop,event) ~ X, cluster=player_id, dfbeta=TRUE, taper=1)",
    nmin_policy = nmin_policy,
    nmin_value = nmin_value,
    tests_null_effect = tests_null_effect,
    tests_constant_effect = setNames(list(), character()),
    var_cum_interpretation = "Interval-row influence diagnostic only; not a formal martingale/model-based variance.",
    robvar_cum_interpretation = "Player-clustered cumulative OLS influence covariance; cross-checked against aareg dfbeta.",
    implementation_validation = list(
      rank_and_nmin_window = "PASS",
      window_risk_membership = "PASS",
      tied_OLS_increment = "PASS",
      robust_influence_variance = "PASS",
      maximum_OLS_increment_abs_difference = max_beta_error,
      maximum_robust_annual_variance_abs_difference = max_robust_variance_error,
      event_time_risk_checks = event_audit
    )
  )
  primary_summary <- c(primary_summary, test_metadata)
  list(fit = fit, summary = primary_summary, event_times = event_times, nmin = nmin_value)
}

window <- select_supported_window(df, covariates)
window_data <- clip_window(df, window$start, window$end)
validate_risk_membership(df, window_data, window$support_times)

primary <- fit_export(window_data, covariates, position, window$start, window$end,
                      "default", output_dir)
sensitivity_dir_name <- "nmin1_sensitivity"
sensitivity_dir <- file.path(output_dir, sensitivity_dir_name)
sensitivity_window <- rank_only_end(df, covariates, window$start)
sensitivity_data <- clip_window(df, sensitivity_window$start, sensitivity_window$end)
validate_risk_membership(df, sensitivity_data, sensitivity_window$support_times)
sensitivity <- fit_export(sensitivity_data, covariates, position, sensitivity_window$start,
                           sensitivity_window$end, "nmin1", sensitivity_dir)

input_counts <- list(n_subjects = length(unique(df$player_id)), n_periods = nrow(df), n_events = sum(df$event == 1))
primary$summary$n_subjects_input <- input_counts$n_subjects
primary$summary$n_periods_input <- input_counts$n_periods
primary$summary$n_events_input <- input_counts$n_events
primary$summary$n_events_in_window <- sum(window_data$event == 1)
primary$summary$n_event_times_supported <- length(primary$event_times)
primary$summary$supported_integer_risk_times <- window$support_times
primary$summary$effective_support <- list(
  policy = "nmin default = 3*nvar intersected with full risk-set rank",
  integer_risk_years = window$support_times,
  event_years = window$event_times,
  n_integer_risk_years = length(window$support_times),
  n_event_years = length(window$event_times)
)
primary$summary$input_counts_before_window <- input_counts
primary$summary$window_selection <- list(
  rule = window$selection,
  supported_integer_risk_times = window$support_times,
  start_stop_convention = "(a,b]",
  excluded_events_at_or_before_start = sum(df$event == 1 & df$stop <= window$start),
  intervals_truncated_or_censored_at_end = sum(df$stop > window$end & df$start < window$end),
  events_retained = sum(window_data$event == 1),
  event_times_selected_by_rank_and_default_nmin = window$event_times
)
primary$summary$sensitivity_nmin1_outputs <- list(
  directory = sensitivity_dir_name,
  files = c(
    sprintf("%s/aalen_cum_coefs_%s.csv", sensitivity_dir_name, position),
    sprintf("%s/aalen_var_cum_%s.csv", sensitivity_dir_name, position),
    sprintf("%s/aalen_robvar_cum_%s.csv", sensitivity_dir_name, position),
    sprintf("%s/aalen_fit_%s.rds", sensitivity_dir_name, position),
    sprintf("%s/aalen_summary_%s.json", sensitivity_dir_name, position)
  ),
  janela_inicio = sensitivity_window$start,
  janela_fim = sensitivity_window$end,
  nmin = 1L,
  supported_integer_risk_times = sensitivity_window$support_times,
  event_times_supported = sensitivity_window$event_times,
  selection = sensitivity_window$selection,
  comparison_endpoint_common = min(window$end, sensitivity_window$end)
)
primary$summary$var_cum_interpretation <- "Interval-row influence diagnostic only; not a formal martingale/model-based variance."
primary$summary$robvar_cum_interpretation <- "Player-clustered cumulative OLS influence covariance; cross-checked against aareg dfbeta."

primary_curve <- read.csv(file.path(output_dir, sprintf("aalen_cum_coefs_%s.csv", position)), check.names = FALSE)
sensitivity_curve <- read.csv(file.path(sensitivity_dir, sprintf("aalen_cum_coefs_%s.csv", position)), check.names = FALSE)
primary_robvar <- read.csv(file.path(output_dir, sprintf("aalen_robvar_cum_%s.csv", position)), check.names = FALSE)
sensitivity_robvar <- read.csv(file.path(sensitivity_dir, sprintf("aalen_robvar_cum_%s.csv", position)), check.names = FALSE)
comparison_endpoint <- min(window$end, sensitivity_window$end)
primary_curve_row <- max(which(primary_curve$time <= comparison_endpoint))
sensitivity_curve_row <- max(which(sensitivity_curve$time <= comparison_endpoint))
primary_robvar_row <- max(which(primary_robvar$time <= comparison_endpoint))
sensitivity_robvar_row <- max(which(sensitivity_robvar$time <= comparison_endpoint))
primary$summary$nmin1_common_endpoint_comparison <- list(
  endpoint = comparison_endpoint,
  primary_curve_time = primary_curve$time[primary_curve_row],
  sensitivity_curve_time = sensitivity_curve$time[sensitivity_curve_row],
  primary_robvar_time = primary_robvar$time[primary_robvar_row],
  sensitivity_robvar_time = sensitivity_robvar$time[sensitivity_robvar_row],
  max_abs_B_difference = max(abs(as.numeric(primary_curve[primary_curve_row, -1, drop = TRUE]) -
                                 as.numeric(sensitivity_curve[sensitivity_curve_row, -1, drop = TRUE]))),
  max_abs_robvar_difference = max(abs(as.numeric(primary_robvar[primary_robvar_row, -1, drop = TRUE]) -
                                      as.numeric(sensitivity_robvar[sensitivity_robvar_row, -1, drop = TRUE])))
)
json_write(primary$summary, file.path(output_dir, sprintf("aalen_summary_%s.json", position)))
 sensitivity$summary$n_subjects_input <- input_counts$n_subjects
 sensitivity$summary$n_periods_input <- input_counts$n_periods
 sensitivity$summary$n_events_input <- input_counts$n_events
 sensitivity$summary$n_events_in_window <- sum(sensitivity_data$event == 1)
 sensitivity$summary$n_event_times_supported <- length(sensitivity$event_times)
 sensitivity$summary$input_counts_before_window <- input_counts
sensitivity$summary$effective_support <- list(
  policy = "rank-only extension with nmin=1; same lower boundary as primary",
  integer_risk_years = sensitivity_window$support_times,
  event_years = sensitivity_window$event_times,
  n_integer_risk_years = length(sensitivity_window$support_times),
  n_event_years = length(sensitivity_window$event_times)
)
 sensitivity$summary$window_selection <- list(
  rule = sensitivity_window$selection,
  supported_integer_risk_times = sensitivity_window$support_times,
  start_stop_convention = "(a,b]",
  intervals_truncated_or_censored_at_end = sum(df$stop > sensitivity_window$end & df$start < sensitivity_window$end),
  events_retained = sum(sensitivity_data$event == 1),
  sensitivity_only = TRUE,
  never_fallback_for_primary = TRUE
)
 sensitivity$summary$var_cum_interpretation <- "Interval-row influence diagnostic only; not a formal martingale/model-based variance."
 sensitivity$summary$robvar_cum_interpretation <- "Player-clustered cumulative OLS influence covariance; cross-checked against aareg dfbeta."
 sensitivity$summary$common_endpoint_comparison <- primary$summary$nmin1_common_endpoint_comparison
json_write(sensitivity$summary, file.path(sensitivity_dir, sprintf("aalen_summary_%s.json", position)))

cat(sprintf("SUCCESS: aareg sem jitter para %s\n", position))
cat(sprintf("  Janela (%g, %g] | nmin default=%d | covariaveis efetivas=%d\n", window$start, window$end, window$nmin_default, length(covariates)))
cat(sprintf("  Subjects: %d | Periods: %d | Events: %d\n", length(unique(window_data$player_id)), nrow(window_data), sum(window_data$event == 1)))
