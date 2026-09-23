# clusterx operator (Rust)

#### Description
`clusterx` (Rust) performs fast clustering by automatic search and find of density peaks — a Rust port of the R `clusterx` operator in this repository (same algorithm: Rodriguez & Laio 2014 with the ClusterX package's automatic peak detection).

The Rust operator lives in `clusterx_rust_operator/`; the R operator at the repository root is **unchanged** and remains the parity reference: on identical inputs the port reproduces the reference's labels exactly on the deterministic paths (see `clusterx_rust_operator/tests/parity/`).

##### Usage
Input projection|.
---|---
`row`   | represents the variables (e.g. channels, markers)
`col`   | represents the observations (e.g. cells, samples, individuals)
`y-axis`| is the measurement value

Input parameters|.
---|---
`dimReduction`   | type of reduction to perform, `pca`, `tsne`, `NULL`, default is `NULL`
`outDim`   | number of dimensions to return, default 2 (`pca` only; `tsne` always reduces to 2, as in the R operator)
`seed`   | seed for random number generation. If less than 0, a random seed is set. Used by the dc down-sample (projections with more than 10000 observations) and the t-SNE random start

Output relations|.
---|---
`cluster`| character, returns a cluster id per column (e.g. per cell)

##### Details
The crosstab is gathered into the matrix the R operator builds with `acast(.ci ~ .ri, mean)` — one point per observation column — and clustered with the density-peak algorithm. Every observation × variable pair must have a value (the R reference fills missing cells with `NaN` and returns `clusterNA` for every point; this port refuses with a message instead).

Parity with the R reference, decided by golden-file comparison on identical inputs (`tests/parity/generate_goldens.R` runs the ClusterX package and dumps every stage at `%.17g`):

| path | claim | evidence |
|---|---|---|
| `dimReduction = NULL` | exact: dc (bitwise), rho, delta, peakID, labels | `lucas_*`, `synthetic12k_*` fixtures (100 and 12000 observations; the latter exercises the `sample(1:n, 10000)` down-sample with R's own RNG stream) |
| `dimReduction = pca` | exact labels; mapped values up to LAPACK's arbitrary singular-vector sign | `lucas_pca_*` fixtures |
| `dimReduction = tsne` | reseed envelope: the port's labels sit inside the spread the reference itself produces across seeds (ARI) | `lucas_tsne_envelope.csv`; the core is additionally checked label-exactly on the reference's own t-SNE mapping |

Deviations from the reference (all documented in `CLAUDE.md`): missing measurements are refused rather than producing all-`NA` labels; the distance-cutoff search gives up instead of looping forever on degenerate shapes; and the reference's `pdist` float32 arithmetic is reproduced, not "fixed".

#### References
see https://github.com/JinmiaoChenLab/ClusterX

##### See Also
`clusterx` (R, this repository)
