#!/usr/bin/env Rscript
# run_aalen_univariate_screening.R
# Triagem univariada do Aalen: ajusta survival::aareg para cada covariavel isolada e usa o
# p-value do teste de coeficiente medio ponderado (summary(aareg)$table, coluna p), o MESMO
# teste e a MESMA regra de empates do ajuste final (run_aalen_tied_inference.R).
#
# Empates: nativos. Todas as saidas de uma temporada dividem o conjunto de risco da propria
# temporada (survival::aareg). Nao ha sorteio, desempate nem semente: o resultado e determinista.
#
# Janela: a mesma regra de select_supported_window do ajuste final (maior trecho contínuo de anos
# com posto completo de [Intercepto + X] e risk_n >= 3*nvar), aplicada a CADA covariavel isolada
# (nvar = 1). A janela final depende das covariaveis selecionadas, que so existem depois desta
# triagem; aplicar a regra ao conjunto candidato inteiro (nvar = 17 a 21) estreitava a janela
# do QB para 2 anos (4,6], o que nao e a janela do ajuste.
#
# Uso:
#   Rscript run_aalen_univariate_screening.R <input_csv> <output_csv> <covariates_comma_sep> [position]

suppressPackageStartupMessages(library(survival))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_aalen_univariate_screening.R <input_csv> <output_csv> <covariates_comma_sep> [position]")
}

input_csv  <- args[[1]]
output_csv <- args[[2]]
covariates <- strsplit(args[[3]], ",", fixed = TRUE)[[1]]
position   <- if (length(args) >= 4) args[[4]] else ""

df <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)

covariates <- sort(covariates[covariates %in% names(df)])
if (length(covariates) == 0) {
  stop("Nenhuma covariável válida encontrada.")
}

for (col in c("start", "stop", "event", covariates)) {
  df[[col]] <- as.numeric(df[[col]])
}
df <- df[complete.cases(df[, c("player_id", "start", "stop", "event", covariates), drop = FALSE]), , drop = FALSE]
if (any(abs(c(df$start, df$stop) - round(c(df$start, df$stop))) > 1e-8)) {
  stop("O estimador requer limites anuais inteiros.")
}

# Janela de posto completo e suporte minimo (mesma regra do ajuste final).
rank_at_time <- function(data, time, covs) {
  risk <- data$start < time & data$stop >= time
  if (!any(risk)) return(0L)
  design <- cbind(Intercept = 1, as.matrix(data[risk, covs, drop = FALSE]))
  qr(design, tol = 1e-10)$rank
}

select_supported_window <- function(data, covs) {
  nmin_default <- 3L * length(covs)
  grid_times <- seq(min(data$start) + 1, max(data$stop), by = 1)
  valid <- vapply(grid_times, function(time) {
    risk_n <- sum(data$start < time & data$stop >= time)
    risk_n > 0 && risk_n >= nmin_default && rank_at_time(data, time, covs) == length(covs) + 1L
  }, logical(1))
  runs <- list()
  current <- numeric()
  for (i in seq_along(grid_times)) {
    if (!valid[[i]]) {
      if (length(current)) runs[[length(runs) + 1L]] <- current
      current <- numeric()
      next
    }
    current <- c(current, grid_times[[i]])
  }
  if (length(current)) runs[[length(runs) + 1L]] <- current
  if (!length(runs)) stop("Nenhum tempo de evento tem posto completo e suporte nmin padrao.")
  chosen <- runs[[which.max(vapply(runs, length, integer(1)))]]
  list(start = chosen[[1]] - 1, end = tail(chosen, 1))
}

# Recorta os intervalos a janela (identico ao clip_window do ajuste final).
clip_window <- function(data, start, end) {
  original_stop <- data$stop
  overlaps <- data$start < end & data$stop > start
  clipped <- data[overlaps, , drop = FALSE]
  stop_before_clip <- original_stop[overlaps]
  clipped$start <- pmax(clipped$start, start)
  clipped$event <- ifelse(stop_before_clip > end, 0, clipped$event)
  clipped$stop <- pmin(stop_before_clip, end)
  clipped <- clipped[clipped$start < clipped$stop, , drop = FALSE]
  clipped
}

results <- data.frame(
  covariate = character(),
  p_value_null = numeric(),
  significant_null = logical(),
  janela_inicio = numeric(),
  janela_fim = numeric(),
  n_subjects = integer(),
  n_events = integer(),
  converged = logical(),
  error_msg = character(),
  stringsAsFactors = FALSE
)

for (cov in covariates) {
  p_null <- NA_real_
  converged <- FALSE
  error_msg <- ""
  janela_cov <- tryCatch(select_supported_window(df, cov), error = function(e) {
    error_msg <<- conditionMessage(e)
    NULL
  })
  if (is.null(janela_cov)) {
    results <- rbind(results, data.frame(
      covariate = cov, p_value_null = NA, significant_null = FALSE,
      janela_inicio = NA, janela_fim = NA, n_subjects = NA, n_events = NA,
      converged = FALSE, error_msg = error_msg, stringsAsFactors = FALSE
    ))
    next
  }
  dados <- clip_window(df, janela_cov$start, janela_cov$end)
  # aareg agrupa as influencias do teste com o cluster fora da ordem interna (survival 3.5.8):
  # ordenar por (stop, -event) alinha os dois, e o p deixa de depender da ordem das linhas.
  dados <- dados[order(dados$stop, -dados$event), , drop = FALSE]
  n_subjects <- length(unique(dados$player_id))
  n_events   <- sum(dados$event == 1)

  if (var(dados[[cov]], na.rm = TRUE) < 1e-12) {
    error_msg <- "zero variance"
  } else {
    tryCatch({
      fit <- survival::aareg(as.formula(paste0("Surv(start, stop, event) ~ `", cov, "`")),
                             data = dados, cluster = player_id, dfbeta = TRUE, taper = 1)
      tab <- summary(fit)$table
      p_null <- as.numeric(tab[nrow(tab), "p"])
      converged <- TRUE
    }, error = function(e) {
      error_msg <<- conditionMessage(e)
    })
  }

  results <- rbind(results, data.frame(
    covariate = cov,
    p_value_null = p_null,
    significant_null = !is.na(p_null) && p_null < 0.05,
    janela_inicio = janela_cov$start,
    janela_fim = janela_cov$end,
    n_subjects = n_subjects,
    n_events = n_events,
    converged = converged,
    error_msg = error_msg,
    stringsAsFactors = FALSE
  ))
}

results <- results[order(results$p_value_null, results$covariate, na.last = TRUE), ]
write.csv(results, output_csv, row.names = FALSE)
cat(sprintf("Screening univariado (aareg, empates nativos, janela por covariavel) %s: %d covariáveis testadas.\n", position, nrow(results)))
cat(sprintf("Significativas (p < 0.05): %d\n", sum(results$significant_null, na.rm = TRUE)))
