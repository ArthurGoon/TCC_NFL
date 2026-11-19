#!/usr/bin/env Rscript
# Contraste preditivo: Aalen com a especificacao curta. n.sim baixo porque so precisamos de B(t).
suppressPackageStartupMessages(library(timereg))
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))

set.seed(42)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript run_aalen_fit.R <train_csv> <test_csv> <covs_comma> <output_csv> [model_rds]")
}

train_csv  <- args[[1]]
test_csv   <- args[[2]]
covariates <- strsplit(args[[3]], ",", fixed = TRUE)[[1]]
output_csv <- args[[4]]
model_rds  <- ifelse(length(args) >= 5, args[[5]], NA)

df_train <- read.csv(train_csv, stringsAsFactors = FALSE, check.names = FALSE)
df_test  <- read.csv(test_csv,  stringsAsFactors = FALSE, check.names = FALSE)

for (col in c("start", "stop", "event")) {
  df_train[[col]] <- as.numeric(df_train[[col]])
  if (col %in% names(df_test)) df_test[[col]] <- as.numeric(df_test[[col]])
}
for (col in covariates) {
  df_train[[col]] <- as.numeric(df_train[[col]])
  if (col %in% names(df_test)) df_test[[col]] <- as.numeric(df_test[[col]])
}

covariates <- covariates[covariates %in% names(df_train) & covariates %in% names(df_test)]
if (length(covariates) == 0) stop("Nenhuma covariavel valida.")

train_cols <- c("player_id", "start", "stop", "event", covariates)
test_cols  <- c("player_id", "start", covariates)
df_train <- df_train[complete.cases(df_train[, train_cols]), ]
df_test  <- df_test[complete.cases(df_test[, test_cols]), ]

# Desempate dos tempos de saida PARA DENTRO do proprio intervalo. Com o tempo em anos inteiros,
# quase todas as saidas empatam, e o timereg (read.surv) desempata somando ruido POSITIVO ao
# tempo, o que joga a saida para dentro da temporada seguinte e tira do conjunto de risco quem
# terminou a mesma temporada sem sair. Deslocando cada saida para (stop - 0,5; stop), nao ha
# empate entre saidas e o conjunto de risco de cada saida e o da propria temporada.
desempata_saidas <- function(d) {
  estado <- if (exists(".Random.seed", envir = globalenv())) get(".Random.seed", envir = globalenv()) else NULL
  set.seed(20241012L)
  ev <- d$event == 1
  d$stop[ev] <- d$stop[ev] - runif(sum(ev)) * 0.5
  if (!is.null(estado)) assign(".Random.seed", estado, envir = globalenv())
  d
}
df_train <- desempata_saidas(df_train)

rhs <- paste(covariates, collapse = " + ")
fml <- as.formula(paste0("Surv(start, stop, event) ~ ", rhs))

fit <- aalen(fml, data = df_train, id = df_train$player_id,
             robust = 1, n.sim = 50, residuals = 0, silent = 1)

cum_df <- as.data.frame(fit$cum)
time_vec <- cum_df[, 1]
B_mat <- as.matrix(cum_df[, -1])
term_names <- colnames(B_mat)
max_abs_B <- max(abs(B_mat), na.rm = TRUE)
if (!length(B_mat) || !any(is.finite(B_mat)) || max_abs_B == 0) {
  stop(sprintf("aalen degenerado: B(t) identicamente 0 (%d eventos, %d covs).",
               sum(df_train$event == 1), length(covariates)))
}

X_test <- as.matrix(df_test[, covariates, drop = FALSE])
n_times <- length(time_vec)
n_test  <- nrow(X_test)

Lambda <- matrix(B_mat[, 1], nrow = n_test, ncol = n_times, byrow = TRUE)
for (j in seq_along(covariates)) {
  cov_name <- covariates[j]
  col_idx <- which(term_names == cov_name)
  if (length(col_idx) == 0) next
  Lambda <- Lambda + outer(X_test[, j], B_mat[, col_idx], "*")
}

S_mat <- exp(-pmax(Lambda, 0))
out_df <- data.frame(player_id = df_test$player_id, start_test = df_test$start, stringsAsFactors = FALSE)

find_nearest_idx <- function(target_times, time_grid) {
  sapply(target_times, function(t) which.min(abs(time_grid - t)))
}

# Lambda(t) e uma funcao em degraus: vale o ultimo tempo de evento <= t (0 antes do primeiro).
lambda_em <- function(Lambda, t, time_grid) {
  idx <- findInterval(t, time_grid)
  out <- numeric(length(t))
  ok <- idx > 0
  out[ok] <- Lambda[cbind(which(ok), idx[ok])]
  out
}
# Tempo restante r conta do fim da temporada de teste (stop = start + 1): sair logo apos a
# temporada (r = 0) e um evento em stop. S(h) = P(r > h) exige nenhum evento em
# stop, ..., stop + h, isto e, o incremento de Lambda entre start e start + h + 1.
horizons <- c(1, 2, 3, 5)
start_vec <- df_test$start
for (h in horizons) {
  lambda_start <- lambda_em(Lambda, start_vec, time_vec)
  lambda_end   <- lambda_em(Lambda, start_vec + h + 1, time_vec)
  lambda_cond <- lambda_end - lambda_start
  out_df[[sprintf("S_cond_h%d", h)]] <- exp(-pmax(lambda_cond, 0))
  out_df[[sprintf("Lambda_cond_h%d", h)]] <- lambda_cond
}

write.csv(out_df, output_csv, row.names = FALSE)
if (!is.na(model_rds)) saveRDS(fit, model_rds)

summary_list <- list(
  n_train_periods = nrow(df_train),
  n_train_events = sum(df_train$event == 1),
  n_test_rows = nrow(df_test),
  covariates_used = covariates,
  max_abs_B = max_abs_B,
  n_times = n_times
)
write(toJSON(summary_list, pretty = TRUE, auto_unbox = TRUE),
      sub("\\.csv$", "_summary.json", output_csv))
cat(sprintf("SUCCESS max|B|=%.4f eventos=%d covs=%d\n",
            max_abs_B, sum(df_train$event == 1), length(covariates)))
