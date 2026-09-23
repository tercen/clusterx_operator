# clusterx_rust_operator — maintenance notes

Rust port of the R `clusterx` operator (this repository) — the ClusterX density-peak
clustering of Rodriguez & Laio 2014 as packaged by JinmiaoChenLab/ClusterX 0.99.1.
Built 2026-09-23 following the `create-rust-operator` skill. The R operator at the
repository root is the parity reference and was not changed.

## What it is

Per-column clustering: the crosstab is gathered into the matrix R builds with
`acast(.ci ~ .ri, mean)` — one point per observation column, one dimension per
variable row factor, duplicate cells averaged — and clustered. Output is one
`<namespace>.cluster` string per observation, joined on `.ci` (the manifest's
`Observation` pair, copied from the R operator's `outputSpecsV2`).

Properties (identical to the R operator): `dimReduction` (`tsne`/`pca`/`NULL`),
`outDim` (2), `seed` (< 0 = random). `alpha` is the R operator's constant 0.001,
`gaussian = TRUE`, halos off — none of them are exposed by the R operator either.

| module | holds |
|---|---|
| `props` | the three properties, parsed as f64 then cast |
| `chunked` | the crosstab reader (copied from `asinh_rust_operator`; chunk decode with `rustson`, schema-bounded reads) |
| `input` | the `acast` gather: sums + counts into a dense `n_points × n_vars` matrix |
| `rng` | R's Mersenne-Twister + `set.seed` scramble + `sample()` rejection, transliterated from `flowsom-rs` (which validated it against R 4.0.4) |
| `special` | Student-t quantile for the ESD test (regularised incomplete beta + bisection) |
| `algorithm` | `estimateDc`, `localDensity`, `minDistToHigher`, `peakDetect` (generalised ESD), `clusterAssign` — pure functions over slices |
| `dimred` | PCA (`prcomp(scale=TRUE)` semantics, cyclic Jacobi) and Barnes–Hut t-SNE (transliterated from Rtsne 0.17's `tsne.cpp`/`sptree.cpp`, vp-tree replaced by a brute-force partial sort) |
| `output` | the per-column result table (`.ci` + `<namespace>.cluster`) |
| `context`, `upload`, `pagecache`, `progress`, `tson` | copied from `asinh_rust_operator`/`read_fcs_rust_operator` (the shared-plumbing debt the skill documents) |

## Parity — what is claimed and how it is proved

`tests/parity/generate_goldens.R` runs the **reference** (the ClusterX package,
vendored under `tests/parity/reference/`) on the R operator's own fixture and on
synthetic data, and dumps every stage at `%.17g`. `tests/parity.rs` compares the
port against those dumps. Proven, per path:

* **NULL (deterministic)** — dc **bitwise** (it comes from base R's double
  `dist`), and rho, delta, higherID, peakID and the labels exact, on 100 points
  (`lucas_*`) and 12000 points with `set.seed(100)` (`synthetic12k_*`, which
  exercises the `sample(1:n, 10000)` down-sample: the fixture contains R's
  dumped sample and the port reproduces it draw for draw).
* **pca** — labels exact. The mapped values agree to ~1e-12 up to a per-column
  sign (LAPACK's arbitrary singular-vector signs); a sign flip is a reflection,
  and the clustering works on Euclidean distances, so nothing downstream can
  change. dc agrees to ~1e-14, not bitwise, because the binary search can land
  one step apart on a 1e-15 perturbation of the mapped matrix.
* **tsne (stochastic)** — reseed envelope per the skill: the reference was run
  at seeds {1, 7, 42, 123, 777}, its pairwise label ARIs are committed
  (`lucas_tsne_envelope.csv`, spread −0.06…0.47), and the port must land
  inside that spread against the reference's seed-42 labels. The **clustering
  core** is separately checked label-exactly on the reference's own seed-42
  t-SNE mapping — the reduction is what is stochastic, not the clustering.
  Bitwise agreement with Rtsne is not achievable without LAPACK's exact PCA,
  so no stronger claim is made.

Two reference behaviours had to be **reproduced** (not fixed) for this:

1. **`pdist` computes in C float.** `localDensity` and `minDistToHigher` run on
   distances whose inputs were coerced to f32, accumulated in f32 and
   square-rooted in f32 (`pdist/src/pdist.c`: `float dist(...)`). In f64 the
   rho values differ at 1e-7 — enough to move the ESD peak search. The port's
   `algorithm::pdist` is a float32 pipeline. `estimateDc` uses base R's
   `dist`, which is double, and does not round.
2. **`splitFactorGenerator`'s random folds.** The reference splits the rows
   into RNG-drawn folds and replaces the comparison-zero entries of each
   chunk's distance matrix with the chunk maximum. That maximum only ever wins
   for a maximum-density point (nothing strictly denser to point at), so one
   delta value per maximum-density point is fold-structured. For n ≤ 8096
   there is a single fold and the value is the global maximum distance;
   beyond that the folds are replicated draw for draw (`algorithm::split_folds`),
   which is why the synthetic fixture is generated with a seed. R's own
   unseeded runs are not reproducible in this one value even between R runs.

Deliberate deviations (guards where the reference loops or returns garbage):

* Missing measurements: R fills with `NaN` and produces `clusterNA` for every
  point; the port refuses with a message naming the shape.
* Degenerate shapes: R's `estimateDc` searches `while(TRUE)` and never returns
  when the neighbour rate cannot land in (0.01, 0.02) — every projection with
  fewer than ~6 points, for a start. The port gives up after 100000 steps with
  a message. (The committed generator script had its tiny-input section
  removed for exactly this reason: R never came back.)
* Zero-variance variables under `pca`: R hands LAPACK the NaN; the port
  refuses.

Known non-matching last bits, and why they are safe: R's `mean`/`rowSums`
accumulate in long double, this port in f64 (rho agrees to ~1e-14); `qt` is
computed by continued fraction against R's AS 91 (agrees to ~1e-11, margins in
the ESD test on real data are orders larger — the ESD replay in the fixture
analysis shows margins of 0.2–4.6 where decisions are taken).

## Memory model

`memory_model.json`: `4e-6 · n_cols² + 150 MB` (single product feature — the
platform's formula is a product, so no additive terms are possible).

What the operator actually holds, at P observation columns and V variable rows:

* the gathered matrix, `8 · P · V` bytes;
* the `estimateDc` upper triangle over the down-sample, `4 · min(P, 10000)²`
  bytes (R materialises the full square: `8 · min(P,10000)²`); this is the
  `n_cols²` term — exact at the 10000-point cap;
* per-point arrays (rho, delta, higherID, labels), ~25 · P bytes;
* the binary, tokio/tonic, and gRPC buffers, ~50 MB → `offset = 100` books
  1.5 × 100 = 150 MB.

The model books the gather only where `P ≥ 2V`; for few points and very many
variables (P < 2V with a huge V) it under-books — not a cytometry shape, and
the model should be refit from `stats_d_actual_ram_peak` after real runs.

Measured with `/usr/bin/time`-style VmHWM on the release build: the 12000 × 8
fixture peaks at **389 MB** (the triangle dominates: 4 · 10000² bytes ≈ 381
MiB, matching the booking); the 100 × 10 fixture peaks at **3 MB**.

## The t-SNE path

Ported from Rtsne 0.17 (the version CRAN ships): pca-center-only →
`normalize_input` → brute-force K = 3 × perplexity neighbours →
`computeProbabilities` beta search → `symmetrizeMatrix` → 1000 iterations of
the Barnes–Hut gradient with the quadtree (`sptree.cpp` transliterated;
QT_NODE_CAPACITY = 1, duplicates not inserted). The random start is R's
`randn` (polar Box–Muller, second variate discarded) on the ported R stream,
so `seed` makes runs repeatable. Substitutions vs Rtsne, and why they are
safe given the reseed-envelope claim: brute-force neighbour search (same K
nearest, deterministic order) instead of the vp-tree (whose pivot choices
consume R's RNG in Rtsne); no OpenMP (Rtsne runs `num_threads = 1` from the R
operator's call anyway).

## Tests

* `cargo test` (unit): hand-computed ESD/`qt`/Jacobi/PCA values, TSON round
  trip, label rendering, property defaults.
* `tests/parity.rs`: the golden comparisons above. Regenerate with
  `Rscript tests/parity/generate_goldens.R` (needs the plyr/pdist/Rtsne R
  packages) — only when the reference changes; the fixtures are committed.
* `tests/test.json` (OperatorUnitTest): the R operator's own fixture with
  `dimReduction = NULL`, golden generated from the reference's labels
  (`tests/table1_null.csv`). Provenance: generated by
  `generate_goldens.R` from the ClusterX reference, not exported from a
  Studio run — no Studio was available when the port was built; the
  `cargo test` parity suite is the stronger check and it compares every
  intermediate stage, not just the final labels.

## Release

Same rules as the skill: bump `operator.json`'s `container` tag, let CI go
green, tag `X.Y.Z`, push — one unbroken sequence. The image is built from
`clusterx_rust_operator/` as the docker context; CI lives in
`.github/workflows/rust-ci.yml` (path-filtered to `clusterx_rust_operator/**`
so the R operator's own CI is untouched).
