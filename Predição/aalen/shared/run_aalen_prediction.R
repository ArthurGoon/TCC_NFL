#!/usr/bin/env Rscript
# run_aalen_prediction.R
# Ajusta modelo de Aalen 100% nao-parametrico em dados de treino
# e prediz S(t) truncada CONDICIONAL ao tempo de inicio dos jogadores de teste.
#
# O Aalen opera em tempo absoluto (desde o inicio da carreira). Para predicao
# de carreira restante a partir de um snapshot atual, calculamos:
#   S_cond(h | start) = S(start + h) / S(start) = exp(-(Lambda(start+h) - Lambda(start)))
#
# Truncamento: Lambda_cond_trunc = max(0, Lambda_cond)
#
# Uso:
#   Rscript run_aalen_prediction.R <train_csv> <test_csv> <covs_comma> <output_csv> [model_rds]

suppressPackageStartupMessages(library(timereg))
suppressPackageStartupMessages(library(survival))

set.seed(42)  # Reproducibilidade do bootstrap (n.sim=1000)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript run_aalen_prediction.R <train_csv> <test_csv> <covs_comma> <output_csv> [model_rds]")
}

train_csv   <- args[[1]]
test_csv    <- args[[2]]
covariates  <- strsplit(args[[3]], ",", fixed = TRUE)[[1]]
output_csv  <- args[[4]]
model_rds   <- ifelse(length(args) >= 5, args[[5]], NA)

# Ler dados
df_train <- read.csv(train_csv, stringsAsFactors = FALSE, check.names = FALSE)
df_test  <- read.csv(test_csv,  stringsAsFactors = FALSE, check.names = FALSE)

# Garantir tipos numericos
for (col in c("start", "stop", "event")) {
  df_train[[col]] <- as.numeric(df_train[[col]])
  if (col %in% names(df_test)) df_test[[col]] <- as.numeric(df_test[[col]])
}
for (col in covariates) {
  df_train[[col]] <- as.numeric(df_train[[col]])
  if (col %in% names(df_test)) df_test[[col]] <- as.numeric(df_test[[col]])
}

# Filtrar covs que existem em ambos
covariates <- covariates[covariates %in% names(df_train) & covariates %in% names(df_test)]
if (length(covariates) == 0) stop("Nenhuma covariavel valida em ambos os datasets.")

# Remover linhas com NA nas colunas essenciais
train_cols <- c("player_id", "start", "stop", "event", covariates)
test_cols  <- c("player_id", covariates)
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

# Garantir que start existe no teste (necessario para sobrevivencia condicional)
if (!"start" %in% names(df_test)) {
  stop("Coluna 'start' nao encontrada no teste. Necessaria para sobrevivencia condicional.")
}

# Ajustar modelo Aalen 100% nao-parametrico no treino
rhs <- paste(covariates, collapse = " + ")
fml <- as.formula(paste0("Surv(start, stop, event) ~ ", rhs))

cat(sprintf("Ajustando Aalen em %d periodos, %d eventos...\n",
            nrow(df_train), sum(df_train$event == 1)))

fit <- NULL
tryCatch({
  fit <- aalen(fml, data = df_train, id = df_train$player_id,
               robust = 1, n.sim = 1000, residuals = 0, silent = 1)
}, error = function(e) {
  stop(sprintf("aalen failed: %s", conditionMessage(e)))
})

# Extrair cumulative coefficients B(t)
cum_df <- as.data.frame(fit$cum)
time_vec <- cum_df[, 1]  # primeira coluna eh o tempo
B_mat <- as.matrix(cum_df[, -1])  # demais colunas sao B(t)
term_names <- colnames(B_mat)
if (!length(B_mat) || !any(is.finite(B_mat)) || max(abs(B_mat), na.rm = TRUE) == 0) {
  stop(sprintf(
    "aalen degenerado: B(t) identicamente 0 (%d eventos, %d covs). Nao escrever S=1.",
    sum(df_train$event == 1), length(covariates)
  ))
}

# Montar matriz de covariaveis do teste
X_test <- as.matrix(df_test[, covariates, drop = FALSE])

n_times <- length(time_vec)
n_test  <- nrow(X_test)

# Lambda: matriz n_test x n_times (NAO TRUNCADO — usamos para diferencas)
Lambda <- matrix(B_mat[, 1], nrow = n_test, ncol = n_times, byrow = TRUE)

# Adicionar contribuicao de cada covariavel
for (j in seq_along(covariates)) {
  cov_name <- covariates[j]
  col_idx <- which(term_names == cov_name)
  if (length(col_idx) == 0) {
    col_idx <- grep(paste0("^", cov_name, "$"), term_names)
    if (length(col_idx) == 0) {
      warning(sprintf("Covariavel '%s' nao encontrada nos termos do modelo. Pulando.", cov_name))
      next
    }
  }
  Lambda <- Lambda + outer(X_test[, j], B_mat[, col_idx], "*")
}

# Truncamento do Lambda absoluto (mantido para diagnostico)
Lambda_trunc <- pmax(Lambda, 0)
S_mat <- exp(-Lambda_trunc)

# Data frame de saida
out_df <- data.frame(
  player_id = df_test$player_id,
  start_test = df_test$start,
  stringsAsFactors = FALSE
)

# Adicionar colunas S(t) e Lambda(t) absolutos para diagnostico
for (i in seq_len(n_times)) {
  col_name <- sprintf("S_t%.4f", time_vec[i])
  out_df[[col_name]] <- S_mat[, i]
}
for (i in seq_len(n_times)) {
  col_name <- sprintf("Lambda_t%.4f", time_vec[i])
  out_df[[col_name]] <- Lambda[, i]
}

# Sobrevivencia CONDICIONAL do tempo restante r (contado do fim da temporada):
# S(h) = P(r > h) = exp(-(Lambda(start+h+1) - Lambda(start)))
# NAO truncamos Lambda antes de subtrair — truncamos apenas a diferenca final

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
  lambda_cond_trunc <- pmax(lambda_cond, 0)
  s_cond <- exp(-lambda_cond_trunc)
  
  out_df[[sprintf("S_cond_h%d", h)]] <- s_cond
  out_df[[sprintf("Lambda_cond_h%d", h)]] <- lambda_cond
  out_df[[sprintf("Lambda_cond_trunc_h%d", h)]] <- lambda_cond_trunc
  out_df[[sprintf("t_eff_start_h%d", h)]] <- start_vec
  out_df[[sprintf("t_eff_end_h%d", h)]] <- start_vec + h + 1
}

# Salvar predicoes
write.csv(out_df, output_csv, row.names = FALSE)
cat(sprintf("Predicoes salvas: %s (%d jogadores x %d tempos)\n", output_csv, n_test, n_times))

# Salvar modelo RDS se solicitado
if (!is.na(model_rds)) {
  saveRDS(fit, model_rds)
  cat(sprintf("Modelo RDS salvo: %s\n", model_rds))
}

# Salvar tambem um resumo do modelo
n_trunc_abs  <- sum(Lambda < 0)
n_trunc_cond <- sum(out_df$Lambda_cond_trunc_h1 < 0 | out_df$Lambda_cond_h1 < 0)

summary_list <- list(
  n_train_periods = nrow(df_train),
  n_train_events = sum(df_train$event == 1),
  n_test_subjects = length(unique(df_test$player_id)),
  n_test_rows = nrow(df_test),
  covariates_used = covariates,
  time_grid = as.numeric(time_vec),
  n_truncated_absolute = n_trunc_abs,
  pct_truncated_absolute = 100 * n_trunc_abs / length(Lambda)
)

summary_json <- jsonlite::toJSON(summary_list, pretty = TRUE, auto_unbox = TRUE)
summary_path <- sub("\\.csv$", "_summary.json", output_csv)
write(summary_json, summary_path)
cat(sprintf("Resumo salvo: %s\n", summary_path))

cat("SUCCESS\n")
