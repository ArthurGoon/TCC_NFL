#!/usr/bin/env Rscript
# run_aalen_inference.R
# Ajusta modelo de Aalen 100% NAO-PARAMETRICO
# O timereg lida com singularidade de X'X internamente via pseudo-inversa

suppressPackageStartupMessages(library(timereg))
suppressPackageStartupMessages(library(survival))

set.seed(42)  # Semente da reamostragem por multiplicadores do erro-padrao (n.sim=1000)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript run_aalen_inference.R <input_csv> <output_dir> <position> <covariates_comma_sep>")
}

input_csv   <- args[[1]]
output_dir  <- args[[2]]
position    <- args[[3]]
covariates  <- strsplit(args[[4]], ",", fixed = TRUE)[[1]]

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)
covariates <- sort(covariates[covariates %in% names(df)])
if (length(covariates) == 0) stop("Nenhuma covariavel valida.")

for (col in c("start", "stop", "event")) df[[col]] <- as.numeric(df[[col]])
for (col in covariates) df[[col]] <- as.numeric(df[[col]])

df <- df[complete.cases(df[, c("player_id", "start", "stop", "event", covariates)]), ]

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
df <- desempata_saidas(df)

# Janela de estimacao do Aalen: o estimador so e definido onde a matriz (intercepto + covariaveis)
# dos jogadores em risco tem posto completo (Martinussen e Scheike, 2006). Com tempo em anos de
# carreira e intervalos de uma temporada, os jogadores em risco no ano k sao as linhas com
# start = k - 1. A janela e o trecho continuo mais longo de anos com posto completo; antes dele
# (primeiro ano, em que as variaveis defasadas valem zero para todos) e depois dele (carreiras
# longas, com poucos jogadores em risco) o modelo nao e estimavel.
janela_posto <- function(d, covs) {
  anos <- sort(unique(d$start))
  ok <- sapply(anos, function(a) {
    linhas <- d[d$start == a, covs, drop = FALSE]
    if (nrow(linhas) <= length(covs)) return(FALSE)
    qr(cbind(1, as.matrix(linhas)))$rank == length(covs) + 1
  })
  if (!any(ok)) stop("Nenhum ano de carreira com posto completo.")
  r <- rle(ok)
  fim_run <- cumsum(r$lengths)
  ini_run <- fim_run - r$lengths + 1
  melhores <- which(r$values)
  k <- melhores[which.max(r$lengths[melhores])]
  list(inicio = anos[ini_run[k]], fim = anos[fim_run[k]] + 1)
}
janela <- janela_posto(df, covariates)
cat(sprintf("Janela de posto completo: anos de carreira (%g, %g]\n", janela$inicio, janela$fim))

n_subjects <- length(unique(df$player_id))
n_periods  <- nrow(df)
n_events   <- sum(df$event == 1)

# Montar formula 100% nao-parametrica
rhs <- paste(covariates, collapse = " + ")
fml <- as.formula(paste0("Surv(start, stop, event) ~ ", rhs))

cat(sprintf("Formula: %s\n", deparse(fml)))

# Ajustar 100% nao-parametrico
fit <- NULL
tryCatch({
  fit <- aalen(fml, data=df, id=df$player_id, robust=1, n.sim=1000, residuals=0, silent=1,
               start.time=janela$inicio, max.time=janela$fim)
}, error = function(e) {
  stop(sprintf("aalen failed: %s", conditionMessage(e)))
})

# Extrair resultados
cum_coefs <- as.data.frame(fit$cum)
write.csv(cum_coefs, file.path(output_dir, sprintf("aalen_cum_coefs_%s.csv", position)), row.names = FALSE)

var_cum <- as.data.frame(fit$var.cum)
write.csv(var_cum, file.path(output_dir, sprintf("aalen_var_cum_%s.csv", position)), row.names = FALSE)

robvar_cum <- as.data.frame(fit$robvar.cum)
write.csv(robvar_cum, file.path(output_dir, sprintf("aalen_robvar_cum_%s.csv", position)), row.names = FALSE)

# Testes
n_terms <- length(fit$pval.testBeq0)
term_names <- names(fit$pval.testBeq0)
if (is.null(term_names)) term_names <- rep("", n_terms)

json_list <- list(
  janela_inicio = janela$inicio,
  janela_fim = janela$fim,
  position = position,
  n_subjects = n_subjects,
  n_periods = n_periods,
  n_events = n_events,
  covariates_tested = covariates,
  model = "aalen_nonparametric",
  tests_null_effect = list(),
  tests_constant_effect = list()
)

# Testes para todas as covs (np)
for (i in seq_len(n_terms)) {
  term <- term_names[i]
  if (is.null(term) || term == "") term <- paste0("term_", i)
  json_list$tests_null_effect[[term]] <- list(
    p_value = round(fit$pval.testBeq0[i], 6),
    significant = fit$pval.testBeq0[i] < 0.05
  )
  json_list$tests_constant_effect[[term]] <- list(
    p_value = round(fit$pval.testBeqC[i], 6),
    is_constant = fit$pval.testBeqC[i] >= 0.05
  )
}

json_str <- jsonlite::toJSON(json_list, pretty = TRUE, auto_unbox = TRUE)
write(json_str, file.path(output_dir, sprintf("aalen_summary_%s.json", position)))

# Figura
fig_path <- file.path(output_dir, "..", "figuras", sprintf("aalen_cum_coefs_%s.png", position))
dir.create(dirname(fig_path), recursive = TRUE, showWarnings = FALSE)
png(fig_path, width = 1800, height = max(1200, 300 * ncol(cum_coefs)), res = 150)
par(mfrow = c(ceiling(ncol(cum_coefs) / 2), 2))
plot(fit)
dev.off()

# RDS
saveRDS(fit, file.path(output_dir, sprintf("aalen_fit_%s.rds", position)))

cat(sprintf("SUCCESS: Aalen nao-parametrico para %s\n", position))
cat(sprintf("  Subjects: %d | Periods: %d | Events: %d\n", n_subjects, n_periods, n_events))
cat(sprintf("  Termos: %d\n", n_terms))
