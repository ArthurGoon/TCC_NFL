#!/usr/bin/env Rscript
# run_cox_aalen_inference.R
# Ajusta o modelo Cox-Aalen teorico (Scheike & Zhang 2002) via timereg::cox.aalen().
# Formula: lambda(t) = Y(t) * {alpha0(t) + X' alpha(t)} * exp(Z' beta)
# Termos sem prop() = aditivos (Aalen, efeito no tempo).
# Termos com prop() = multiplicativos (Cox, HR constante).
# Estimacao: algoritmo iterativo MMQ (aditivo) + verossimilhanca parcial (multiplicativo).

suppressPackageStartupMessages(library(timereg))
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))

set.seed(42)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop(paste0(
    "Usage: Rscript run_cox_aalen_inference.R ",
    "<input_csv> <output_dir> <position> <additive_comma_sep_or_-> <prop_comma_sep> [inicio fim]"
  ))
}

input_csv  <- args[[1]]
output_dir <- args[[2]]
position   <- args[[3]]
additive_raw <- args[[4]]
prop_raw     <- args[[5]]
# Janela de posto completo do Aalen da mesma posicao (start.time, max.time). Sem ela, o cox.aalen
# usaria todos os anos, inclusive os de matriz singular, onde B(t) nao e identificado.
janela_ini <- if (length(args) >= 7) as.numeric(args[[6]]) else NA_real_
janela_fim <- if (length(args) >= 7) as.numeric(args[[7]]) else NA_real_

parse_covs <- function(raw) {
  raw <- trimws(raw)
  if (is.na(raw) || raw == "" || raw == "-") {
    return(character(0))
  }
  covs <- trimws(unlist(strsplit(raw, ",", fixed = TRUE)))
  covs[covs != "" & covs != "-"]
}

additive <- parse_covs(additive_raw)
prop_covs <- parse_covs(prop_raw)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)

needed <- unique(c("player_id", "start", "stop", "event", additive, prop_covs))
missing_cols <- setdiff(needed, names(df))
if (length(missing_cols) > 0) {
  stop(sprintf("Colunas ausentes no CSV: %s", paste(missing_cols, collapse = ", ")))
}
if (length(prop_covs) == 0) {
  stop("A parte multiplicativa (prop) nao pode ser vazia neste teste.")
}

for (col in c("start", "stop", "event", additive, prop_covs)) {
  df[[col]] <- as.numeric(df[[col]])
}

df <- df[complete.cases(df[, needed]), ]

# Desempate dos tempos de saida PARA DENTRO do proprio intervalo. Com o tempo em anos inteiros,
# quase todas as saidas empatam, e o timereg (read.surv) desempata somando ruido POSITIVO ao
# tempo, o que joga a saida para dentro da temporada seguinte e tira do conjunto de risco quem
# terminou a mesma temporada sem sair. Deslocando cada saida para (stop - 0,5; stop), nao ha
# empate entre saidas e o conjunto de risco de cada saida e o da propria temporada.
desempata_saidas <- function(d) {
  estado <- if (exists(".Random.seed", envir = globalenv())) get(".Random.seed", envir = globalenv()) else NULL
  set.seed(20241012L)
  # o passo 67 testa outras sementes por esta variavel; sem ela, vale a semente fixa acima
  if (nzchar(Sys.getenv("COX_AALEN_SEED"))) set.seed(as.integer(Sys.getenv("COX_AALEN_SEED")))
  ev <- d$event == 1
  d$stop[ev] <- d$stop[ev] - runif(sum(ev)) * 0.5
  if (!is.null(estado)) assign(".Random.seed", estado, envir = globalenv())
  d
}
# Eventos dentro da janela de posto do cox.aalen (start.time, max.time], contados ANTES do desempate
# (que so mexe no interior da temporada). Igual ao n_events_in_window do Aalen, quando a janela existe.
n_events_janela <- if (is.na(janela_ini)) sum(df$event == 1) else sum(df$event == 1 & df$stop > janela_ini & df$stop <= janela_fim)
df <- desempata_saidas(df)

n_subjects <- length(unique(df$player_id))
n_periods  <- nrow(df)
n_events   <- sum(df$event == 1)

prop_terms <- paste0("prop(", prop_covs, ")")
rhs_parts <- c(additive, prop_terms)
fml <- as.formula(paste0("Surv(start, stop, event) ~ ", paste(rhs_parts, collapse = " + ")))

cat(sprintf("Formula: %s\n", paste(deparse(fml), collapse = " ")))
cat(sprintf("  Aditivo X: %s\n", if (length(additive) == 0) "(somente intercepto)" else paste(additive, collapse = ", ")))
cat(sprintf("  Multiplicativo Z: %s\n", paste(prop_covs, collapse = ", ")))

fit <- NULL
janela_args <- if (is.na(janela_ini)) list() else list(start.time = janela_ini, max.time = janela_fim)
tryCatch({
  fit <- do.call(cox.aalen, c(list(
    fml,
    data = df,
    id = df$player_id,
    robust = 1,
    n.sim = 1000,
    residuals = 0,
    silent = 1,
    basesim = 1,
    max.clust = 10000
  ), janela_args))
}, error = function(e) {
  stop(sprintf("cox.aalen failed: %s", conditionMessage(e)))
})

if (is.null(fit) || is.null(fit$gamma)) {
  stop("cox.aalen nao retornou coeficientes multiplicativos (gamma).")
}

strip_prop <- function(nm) {
  sub("^prop\\((.*)\\)$", "\\1", nm)
}

gamma <- as.numeric(fit$gamma)
gamma_names <- strip_prop(rownames(fit$gamma))
if (is.null(gamma_names) || length(gamma_names) != length(gamma)) {
  gamma_names <- prop_covs
}

se_model <- if (!is.null(fit$var.gamma)) sqrt(diag(as.matrix(fit$var.gamma))) else rep(NA_real_, length(gamma))
se_robust <- if (!is.null(fit$robvar.gamma)) sqrt(diag(as.matrix(fit$robvar.gamma))) else se_model
se_use <- se_robust
z <- gamma / se_use
p_wald <- 2 * pnorm(-abs(z))
hr <- exp(gamma)
ci_lo <- exp(gamma - 1.96 * se_use)
ci_hi <- exp(gamma + 1.96 * se_use)

p_prop <- rep(NA_real_, length(gamma))
if (!is.null(fit$pval.Prop)) {
  prop_map <- fit$pval.Prop
  names(prop_map) <- strip_prop(names(prop_map))
  p_prop <- as.numeric(prop_map[gamma_names])
}

beta_df <- data.frame(
  covariate = gamma_names,
  coef = gamma,
  se = se_model,
  se_robust = se_robust,
  HR = hr,
  CI_lower = ci_lo,
  CI_upper = ci_hi,
  p_value = p_wald,
  p_proportionality = p_prop,
  stringsAsFactors = FALSE
)
write.csv(
  beta_df,
  file.path(output_dir, sprintf("cox_aalen_beta_%s.csv", position)),
  row.names = FALSE
)

cum_coefs <- as.data.frame(fit$cum)
write.csv(
  cum_coefs,
  file.path(output_dir, sprintf("cox_aalen_cum_%s.csv", position)),
  row.names = FALSE
)

if (!is.null(fit$var.cum)) {
  write.csv(
    as.data.frame(fit$var.cum),
    file.path(output_dir, sprintf("cox_aalen_var_cum_%s.csv", position)),
    row.names = FALSE
  )
}
if (!is.null(fit$robvar.cum)) {
  write.csv(
    as.data.frame(fit$robvar.cum),
    file.path(output_dir, sprintf("cox_aalen_robvar_cum_%s.csv", position)),
    row.names = FALSE
  )
}

safe_named_list <- function(x) {
  if (is.null(x) || length(x) == 0) {
    return(list())
  }
  nm <- names(x)
  if (is.null(nm)) {
    nm <- paste0("term_", seq_along(x))
  }
  out <- list()
  for (i in seq_along(x)) {
    key <- nm[[i]]
    if (is.null(key) || key == "") key <- paste0("term_", i)
    out[[key]] <- unname(as.numeric(x[[i]]))
  }
  out
}

json_list <- list(
  position = position,
  model = "cox_aalen_scheike_zhang_2002",
  janela_inicio = if (is.na(janela_ini)) NULL else janela_ini,
  janela_fim = if (is.na(janela_fim)) NULL else janela_fim,
  formula = paste(deparse(fml), collapse = " "),
  n_subjects = n_subjects,
  n_periods = n_periods,
  n_events = n_events,
  n_events_janela = n_events_janela,
  additive_covariates = as.list(additive),
  multiplicative_covariates = as.list(prop_covs),
  converged = TRUE,
  tests_null_effect = list(),
  tests_constant_effect = list(),
  tests_proportionality = list()
)

p0 <- fit$pval.testBeq0
pc <- fit$pval.testBeqC
if (!is.null(p0)) {
  nm <- names(p0)
  if (is.null(nm)) nm <- paste0("term_", seq_along(p0))
  for (i in seq_along(p0)) {
    term <- nm[[i]]
    json_list$tests_null_effect[[term]] <- list(
      p_value = unname(round(as.numeric(p0[[i]]), 6)),
      significant = unname(as.numeric(p0[[i]]) < 0.05)
    )
  }
}
if (!is.null(pc)) {
  nm <- names(pc)
  if (is.null(nm)) nm <- paste0("term_", seq_along(pc))
  for (i in seq_along(pc)) {
    term <- nm[[i]]
    json_list$tests_constant_effect[[term]] <- list(
      p_value = unname(round(as.numeric(pc[[i]]), 6)),
      is_constant = unname(as.numeric(pc[[i]]) >= 0.05)
    )
  }
}
if (!is.null(fit$pval.Prop)) {
  pp <- fit$pval.Prop
  names(pp) <- strip_prop(names(pp))
  for (i in seq_along(pp)) {
    term <- names(pp)[[i]]
    json_list$tests_proportionality[[term]] <- list(
      p_value = unname(round(as.numeric(pp[[i]]), 6)),
      proportional = unname(as.numeric(pp[[i]]) >= 0.05)
    )
  }
}

write(
  jsonlite::toJSON(json_list, pretty = TRUE, auto_unbox = TRUE, null = "null"),
  file.path(output_dir, sprintf("cox_aalen_summary_%s.json", position))
)

fig_dir <- file.path(output_dir, "..", "figuras")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
fig_path <- file.path(fig_dir, sprintf("cox_aalen_cum_%s.png", position))
n_add_plots <- max(1L, ncol(cum_coefs) - 1L)
png(fig_path, width = 1600, height = max(900, 280 * n_add_plots), res = 140)
par(mfrow = c(ceiling(n_add_plots / 2), min(2, n_add_plots)))
plot(fit)
dev.off()

saveRDS(fit, file.path(output_dir, sprintf("cox_aalen_fit_%s.rds", position)))

cat(sprintf("SUCCESS: Cox-Aalen para %s\n", position))
cat(sprintf("  Subjects: %d | Periods: %d | Events: %d\n", n_subjects, n_periods, n_events))
cat(sprintf("  Aditivo: %d covs + intercepto | Multiplicativo: %d covs\n", length(additive), length(prop_covs)))
