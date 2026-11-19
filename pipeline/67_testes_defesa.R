# Forma funcional dos termos finais e selecao retroativa no modelo discreto (passo 67).
library(survival); library(splines)
a <- commandArgs(trailingOnly = TRUE); dir <- a[[1]]; pos <- a[[2]]
d <- read.csv(file.path(dir, sprintf("cand_%s.csv", pos)), check.names = FALSE)
final <- readLines(file.path(dir, sprintf("final_%s.txt", pos)))
cand <- setdiff(names(d), c("player_id", "start", "stop", "event"))
q <- function(x) sprintf("`%s`", x)
fm <- function(v) as.formula(paste("Surv(start, stop, event) ~", paste(v, collapse = " + ")))
base <- coxph(fm(q(final)), data = d, ties = "efron")
esp <- do.call(rbind, lapply(final, function(x) {
  if (length(unique(d[[x]])) <= 5) return(NULL)
  alt <- coxph(fm(c(q(setdiff(final, x)), sprintf("ns(%s, 3)", q(x)))), data = d, ties = "efron")
  data.frame(posicao = pos, covariavel = x, p_nao_linear = anova(base, alt)$`Pr(>|Chi|)`[2])
}))
write.csv(esp, file.path(dir, sprintf("splines_%s.csv", pos)), row.names = FALSE)
ev <- tapply(d$event, d$stop, sum); anos <- as.numeric(names(ev))
corte <- max(anos[cumsum(rev(ev))[length(ev):1] >= 3])
d$ano <- factor(pmin(d$stop, corte))
vivas <- cand
repeat {
  g <- glm(as.formula(paste("event ~ ano +", paste(q(vivas), collapse = " + "))),
           data = d, family = binomial(link = "cloglog"), control = glm.control(maxit = 100))
  s <- summary(g)$coefficients; s <- s[!grepl("^\\(Intercept\\)|^ano", rownames(s)), , drop = FALSE]
  rownames(s) <- gsub("`", "", rownames(s))
  if (max(s[, 4]) < 0.05 || length(vivas) == 1) break
  vivas <- setdiff(vivas, rownames(s)[which.max(s[, 4])])
}
writeLines(vivas, file.path(dir, sprintf("glm_lista_%s.txt", pos)))
