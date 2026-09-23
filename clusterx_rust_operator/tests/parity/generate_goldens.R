#!/usr/bin/env Rscript
# Golden fixtures for the Rust port's parity tests (create-rust-operator §8.2).
#
# Runs the **reference implementation** — the ClusterX package the R operator
# depends on (JinmiaoChenLab/ClusterX 0.99.1, sourced from ./reference/) — on
# the R operator's own fixture and on synthetic data, and dumps every stage the
# port checks: the gathered matrix, the dc estimate, rho, delta, higherID,
# peakID and the labels. Numeric values are written with sprintf("%.17g")
# because write.csv's 15 digits make an exact port look wrong on ties.
#
# Run from this directory:  Rscript generate_goldens.R
# Pinned reference version: ClusterX 0.99.1, R 4.3.3 (the fixtures were
# generated with that combination; they are committed, so regenerating them
# is only needed when the reference changes).

setwd(dirname(commandArgs(trailingOnly = FALSE)[grep("^--file=", commandArgs(trailingOnly = FALSE))]))

suppressPackageStartupMessages(library(plyr))
suppressPackageStartupMessages(library(Rtsne))
for (f in c("reference/ClusterX.R", "reference/cytof_dimensionReduction.R")) {
  source(f)
}

# %.17g for every numeric, "" for NA, so a dump round-trips exactly.
dump <- function(df, path) {
  if (is.null(dim(df))) df <- as.data.frame(as.list(df)) # a named vector: one row
  lines <- c(
    paste(names(df), collapse = ","),
    apply(df, 1, function(row) {
      paste(vapply(row, function(x) {
        if (is.na(x)) "" else sprintf("%.17g", as.numeric(x))
      }, ""), collapse = ",")
    })
  )
  writeLines(lines, path)
}
dump_int <- function(v, path) writeLines(vapply(v, function(x) if (is.na(x)) "" else sprintf("%d", as.integer(x)), ""), path)

# One label per point, exactly as main.R writes it: paste0("cluster", cluster).
labels <- function(cluster) ifelse(is.na(cluster), "clusterNA", paste0("cluster", cluster))

dump_labels <- function(cluster, path) writeLines(labels(cluster), path)

ari <- function(a, b) { # adjusted Rand index, base R, for the reseed envelope
  ta <- table(a, b)
  if (length(ta) == 0) return(0)
  choose2 <- function(x) x * (x - 1) / 2
  sum_ij <- sum(choose2(ta))
  sum_a <- sum(choose2(rowSums(ta)))
  sum_b <- sum(choose2(colSums(ta)))
  n <- choose2(sum(ta))
  expected <- sum_a * sum_b / n
  max_index <- (sum_a + sum_b) / 2
  (sum_ij - expected) / (max_index - expected)
}

## ---- 1. the R operator's own fixture, dimReduction = NULL -------------------
# tests/lucas_measurement_exp.tsv, 10 variables x 100 observations; the matrix
# is built exactly as main.R builds it: acast(.ci ~ .ri, mean), rows = .ci.
cat("lucas / NULL\n")
x <- read.table("../../../tests/lucas_measurement_exp.tsv", header = TRUE)
data <- tapply(x$Measurement, list(x$Observation, x$Variable), mean) # acast(.ci ~ .ri, mean)
data[is.na(data)] <- NaN
mat <- as.matrix(data)
dump(as.data.frame(mat), "lucas_matrix.csv")

r <- ClusterX(mat, dimReduction = NULL)
dump_int(r$cluster, "lucas_null_cluster.csv")
dump(c(dc = r$dc), "lucas_null_dc.csv")
dump(as.data.frame(cbind(rho = r$rho, delta = r$delta, higherID = r$higherID)), "lucas_null_rho_delta.csv")
dump_int(r$peakID, "lucas_null_peakID.csv")

## ---- 2. the same fixture, dimReduction = pca, outDim = 2 --------------------
cat("lucas / pca\n")
rp <- ClusterX(mat, dimReduction = "pca", outDim = 2)
mapped <- cytof_dimReduction(mat, method = "pca", out_dim = 2)
dump(as.data.frame(mapped), "lucas_pca_mapped.csv")
dump_int(rp$cluster, "lucas_pca_cluster.csv")
dump(c(dc = rp$dc), "lucas_pca_dc.csv")
dump(as.data.frame(cbind(rho = rp$rho, delta = rp$delta)), "lucas_pca_rho_delta.csv")

## ---- 3. tsne: seed 42 (the R operator's own test), plus a reseed envelope ---
# t-SNE is stochastic (Rtsne draws its random start from R's RNG), so parity
# for this path is a reseed envelope, not bitwise: the port's labels must sit
# inside the spread the reference itself produces across seeds.
cat("lucas / tsne (5 seeds)\n")
tsne_seeds <- c(1, 7, 42, 123, 777)
tsne_lab <- list()
for (s in tsne_seeds) {
  set.seed(s)
  tsne_lab[[as.character(s)]] <- ClusterX(mat, dimReduction = "tsne", outDim = 2)$cluster
  cat("  seed", s, "done\n")
}
dump_labels(tsne_lab[["42"]], "lucas_tsne_42_cluster.csv")
# mapped data of the seed-42 run, for a shape check of the reduction output
set.seed(42)
mapped42 <- cytof_dimReduction(mat, method = "tsne", out_dim = 2)
dump(as.data.frame(mapped42), "lucas_tsne_42_mapped.csv")

env <- expand.grid(a = tsne_seeds, b = tsne_seeds)
env$ari <- mapply(function(a, b) ari(tsne_lab[[as.character(a)]], tsne_lab[[as.character(b)]]), env$a, env$b)
env <- env[env$a < env$b, ]
dump(env, "lucas_tsne_envelope.csv")

## ---- 4. the dc sampling path (n > 10000 uses R's sample()) ------------------
# 12000 points, 8 variables, three shifted groups; seed 100 makes the sample
# inside estimateDc reproducible, which is what lets the port be compared.
cat("synthetic 12000 / NULL, seed 100\n")
set.seed(7)
n <- 12000
syn <- cbind(
  matrix(rnorm(n * 5), n),
  c(rnorm(n / 3, mean = 4), rnorm(n / 3, mean = -3), rnorm(n / 3)),
  c(rnorm(n / 3), rnorm(n / 3, mean = 5), rnorm(n / 3, mean = -2)),
  matrix(rnorm(n * 1), n)
)
dump(as.data.frame(syn), "synthetic12k_matrix.csv")
set.seed(100)
sampled <- sample(1:n, 10000, replace = FALSE)
dump_int(sampled, "synthetic12k_sample.csv")
set.seed(100)
rs <- ClusterX(syn, dimReduction = NULL)
dump(c(dc = rs$dc), "synthetic12k_dc.csv")
dump_int(rs$cluster, "synthetic12k_cluster.csv")
dump(as.data.frame(cbind(rho = rs$rho, delta = rs$delta, higherID = rs$higherID)), "synthetic12k_rho_delta.csv")
dump_int(rs$peakID, "synthetic12k_peakID.csv")

cat("done\n")
