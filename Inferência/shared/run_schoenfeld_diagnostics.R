args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 5) {
  stop("Usage: Rscript run_schoenfeld_diagnostics.R <input_csv> <output_csv> <output_plot> <cov1,cov2,...> <position> [unpenalized_csv]")
}

input_csv <- args[[1]]
output_csv <- args[[2]]
output_plot <- args[[3]]
covariates <- strsplit(args[[4]], ",", fixed = TRUE)[[1]]
position <- args[[5]]
# Opcional: coeficientes do MESMO coxph sem penalizacao usado no cox.zph.
# E a unica fonte correta para comparar o Cox oficial (L2=0.1) com o nao
# penalizado: mesmo ajuste, mesmas linhas completas, mesmo ties.
unpenalized_csv <- if (length(args) >= 6) args[[6]] else NA_character_

suppressPackageStartupMessages(library(survival))

df <- read.csv(input_csv, stringsAsFactors = FALSE)
required_cols <- c("start", "stop", "event", covariates)
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(paste("Missing columns:", paste(missing_cols, collapse = ", ")))
}

df <- df[complete.cases(df[, required_cols]), required_cols, drop = FALSE]
if (nrow(df) == 0) {
  stop("No complete rows available for Schoenfeld diagnostics.")
}

rhs <- paste(covariates, collapse = " + ")
fml <- as.formula(paste0("Surv(start, stop, event) ~ ", rhs))

fit <- coxph(fml, data = df, ties = "efron")
zph <- cox.zph(fit, transform = "rank")

# Coeficientes do coxph SEM penalizacao (mesmo ajuste que sustenta o cox.zph).
if (!is.na(unpenalized_csv)) {
  cf <- summary(fit)$coefficients
  unp <- data.frame(
    covariate = rownames(cf),
    coxph_coef = cf[, "coef"],
    coxph_HR = exp(cf[, "coef"]),
    coxph_se = cf[, "se(coef)"],
    coxph_z = cf[, "z"],
    coxph_p = cf[, "Pr(>|z|)"],
    row.names = NULL,
    stringsAsFactors = FALSE
  )
  write.csv(unp, unpenalized_csv, row.names = FALSE)
}

tbl <- as.data.frame(zph$table)
tbl$term <- rownames(tbl)
rownames(tbl) <- NULL
names(tbl) <- c("chisq", "df", "p", "term")
tbl <- tbl[, c("term", "chisq", "df", "p")]
tbl$violacao <- tbl$p < 0.05
tbl$interpretacao <- ifelse(tbl$violacao, "Violação (p<0.05)", "OK (p>=0.05)")
tbl$neg_log10_p <- -log10(pmax(tbl$p, 1e-12))
write.csv(tbl, output_csv, row.names = FALSE)

plot_tbl <- subset(tbl, term != "GLOBAL")
plot_tbl <- plot_tbl[order(plot_tbl$p, decreasing = TRUE), ]
global_tbl <- subset(tbl, term == "GLOBAL")

# Rotulos do glossario da tese (fonte unica, a mesma das figuras em Python).
# O R (Rscript) grava espacos do caminho do --file como "~+~" (ex.: "TCC - 11_10"),
# por isso o caminho e desfeito antes de normalizar.
script_dir <- tryCatch(
  {
    arq <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
    dirname(normalizePath(gsub("~+~", " ", arq, fixed = TRUE)))
  },
  error = function(e) "."
)
glossario_csv <- file.path(script_dir, "..", "..", "Predição", "shared", "glossario_variaveis.csv")
glossario <- read.csv(glossario_csv, stringsAsFactors = FALSE, encoding = "UTF-8")
rotulos <- setNames(glossario$rotulo, glossario$variavel)
# Indicador de ausencia de uma taxa imputada: rotulo da taxa + "(ausente)", como no Python.
rotulos <- c(rotulos, setNames(paste0(glossario$rotulo, " (ausente)"), paste0(glossario$variavel, "_missing")))
sem_rotulo <- setdiff(plot_tbl$term, names(rotulos))
if (length(sem_rotulo) > 0) {
  stop(paste("Covariavel sem rotulo no glossario:", paste(sem_rotulo, collapse = ", ")))
}
plot_tbl$label <- unname(rotulos[plot_tbl$term])

# Calibrado para \thesisfigfit no LaTeX (textwidth × 0.44 textheight, A4 UFSCar)
TEXTWIDTH_IN <- 160 / 25.4
TEXTHEIGHT_IN <- (29.7 - 2.5 - 2.0) / 2.54
SLOT_H_IN <- TEXTHEIGHT_IN * 0.44
plot_w_in <- TEXTWIDTH_IN
plot_h_in <- SLOT_H_IN
cex_names <- max(0.95, min(1.2, 4.8 / nrow(plot_tbl)))
cex_p <- max(0.85, min(1.0, 4.5 / nrow(plot_tbl)))
if (grepl("\\.pdf$", output_plot, ignore.case = TRUE)) {
  pdf(output_plot, width = plot_w_in, height = plot_h_in)
} else {
  png(output_plot, width = round(plot_w_in * 200), height = round(plot_h_in * 200), res = 200)
}
layout(matrix(c(1, 2), nrow = 2), heights = c(6.2, 1.0))
# Margem esquerda fixa (maior rótulo entre QB/RB/WR) para alinhar pares na mesma página.
GLOBAL_MAX_LABEL_LEN <- 35L  # observed_career_lag_targets_redzone
left_margin <- max(12L, ceiling(GLOBAL_MAX_LABEL_LEN * 0.42))
xmax <- max(plot_tbl$neg_log10_p, na.rm = TRUE)
x_right <- xmax + 2.4
par(mar = c(4.5, left_margin, 3.5, 7))
cols <- ifelse(plot_tbl$violacao, "#D85A30", "#1D9E75")
bar_centers <- barplot(
  plot_tbl$neg_log10_p,
  horiz = TRUE,
  names.arg = plot_tbl$label,
  las = 1,
  col = cols,
  border = NA,
  main = "",
  xlab = expression(-log[10](p)~"(maior = evidência mais forte contra PH)"),
  cex.names = cex_names,
  cex.axis = 1.05,
  cex.lab = 1.1,
  xlim = c(0, x_right),
  xaxs = "i"
)
abline(v = -log10(0.05), lty = 2, lwd = 2, col = "#444444")
label_x <- xmax + 0.35
for (i in seq_len(nrow(plot_tbl))) {
  text(
    x = label_x,
    y = bar_centers[i],
    labels = sprintf("p = %.4f", plot_tbl$p[i]),
    pos = 4,
    cex = cex_p,
    xpd = NA
  )
}

par(mar = c(0, 1, 0, 1))
plot.new()
if (nrow(global_tbl) == 1) {
  global_p <- global_tbl$p[1]
  global_chisq <- global_tbl$chisq[1]
  global_status <- if (global_tbl$violacao[1]) "Violação global de PH" else "Sem evidência global de violação"
  global_col <- if (global_tbl$violacao[1]) "#D85A30" else "#1D9E75"
  text(0.02, 0.82, labels = sprintf("Teste global: chisq = %.2f | p = %.4f", global_chisq, global_p),
       adj = c(0, 0.5), cex = 1.05, col = "#2E4057")
  text(0.02, 0.52, labels = global_status,
       adj = c(0, 0.5), cex = 1.15, col = global_col, font = 2)
  text(0.02, 0.18,
       labels = "Verde = OK (p>=0.05)   |   Laranja = Violação (p<0.05)   |   Tracejado = limiar 0.05",
       adj = c(0, 0.5), cex = 0.95, col = "#444444")
}
dev.off()
