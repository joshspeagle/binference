# Golden values for issue #77: NPMLE mixture weights from mixsqp.
#
# Writes mixsqp.json next to this script. Two problems, each stored as its inputs (so the
# test rebuilds the likelihood matrix) and mixsqp's solution:
#   "scale": mixsqp's own example, simulatemixdata(n = 400, m = 20) with set.seed(1):
#            x_i ~ a normal scale mixture, L_ij = N(x_i; 0, 1 + s_j^2).
#   "glmix": the Gaussian-location NPMLE that REBayes::GLmix solves (with Mosek, not
#            available here), on REBayes's baseball data as in Brown (2008): 2005 first
#            half, non-pitchers with at least 11 at-bats, y_i = asin(sqrt((H + 1/4) /
#            (AB + 1/2))), se_i = 1 / (2 sqrt(AB)), and GLmix's default grid of 300
#            equally spaced points over the range of y; L_ij = N(y_i; v_j, se_i^2).
# Run: Rscript make_mixsqp.R  (needs mixsqp, REBayes for the data, jsonlite)
library(mixsqp)
library(jsonlite)
data(bball, package = "REBayes")

solve <- function(L) {
  fit <- mixsqp(L, control = list(convtol.sqp = 1e-10, verbose = FALSE))
  w <- fit$x / sum(fit$x)
  list(weights = w, loglik = sum(log(L %*% w)), status = fit$status)
}

set.seed(1)
sim <- simulatemixdata(400, 20)
L <- outer(sim$x, sim$s, function(x, s) dnorm(x, sd = sqrt(1 + s^2)))  # not row-normalised
scale <- c(list(x = sim$x, s = sim$s), solve(L))

b <- subset(bball, year == 2005 & season == 1 & pitcher == 0 & AB >= 11)
y <- asin(sqrt((b$H + 0.25) / (b$AB + 0.5)))
se <- 1 / (2 * sqrt(b$AB))
v <- seq(min(y), max(y), length.out = 300)
L <- outer(seq_along(y), seq_along(v), function(i, j) dnorm(y[i], v[j], se[i]))
glmix <- c(list(y = y, se = se, grid = v), solve(L))

out <- list(
  scale = scale, glmix = glmix,
  software = list(R = R.version.string, mixsqp = as.character(packageVersion("mixsqp")),
                  REBayes = as.character(packageVersion("REBayes")),
                  convtol.sqp = 1e-10)
)
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
write_json(out, file.path(here, "mixsqp.json"), digits = I(17), auto_unbox = TRUE, pretty = FALSE)
