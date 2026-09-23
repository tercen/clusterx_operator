//! Dimension reduction, ported from the two methods the R operator exposes
//! through `reference/cytof_dimensionReduction.R`:
//!
//! * `pca` — `prcomp(data, scale. = TRUE)$x[, 1:outDim]`. R's `prcomp` uses
//!   LAPACK, whose singular-vector signs are arbitrary; a sign flip is a
//!   reflection, and the clustering downstream works on Euclidean distances,
//!   so it cannot change `dc`, `rho`, `delta` or the labels (asserted by the
//!   `lucas_pca_*` parity fixtures up to per-column sign).
//! * `tsne` — R's call is `Rtsne(data, initial_dims = ncol(data), dims = 2,
//!   check_duplicates = FALSE, pca = TRUE)`, the Barnes–Hut t-SNE of
//!   Rtsne 0.17. The algorithm below is a transliteration of that
//!   implementation (`tsne.cpp` + `sptree.cpp`), with two mechanical
//!   substitutions: the vp-tree neighbour search is done by a brute-force
//!   partial sort (same K nearest, deterministic order), and the random
//!   start draws from R's RNG (`rng::RRng`) so a `Seed` makes a run
//!   repeatable. Bitwise agreement with Rtsne is *not* claimed — the PCA
//!   stage's LAPACK signs already differ — so this path is verified as a
//!   reseed envelope, not goldens (see `tests/parity/README.md`).

use crate::rng::RRng;

/// One PCA map (`prcomp(..., scale. = TRUE)`): center and scale each column,
/// then project onto the leading eigenvectors of the scaled Gram matrix.
///
/// Returns an `n × k` row-major matrix, `k = min(out_dim, d)`.
pub fn pca_scaled(data: &[f64], n: usize, d: usize, out_dim: usize) -> anyhow::Result<Vec<f64>> {
    let xs = scale_columns(data, n, d)?;
    let mut gram = gram_matrix(&xs, n, d);
    let (eigenvalues, vectors) = jacobi_eigen(&mut gram, d);
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());
    let k = out_dim.min(d);

    let mut out = vec![0.0f64; n * k];
    for (kk, &col) in order.iter().take(k).enumerate() {
        for i in 0..n {
            let mut s = 0.0;
            for a in 0..d {
                s += xs[i * d + a] * vectors[a * d + col];
            }
            out[i * k + kk] = s;
        }
    }
    Ok(out)
}

/// `scale()`: subtract the column mean, divide by the sample sd (n − 1).
/// R's `prcomp(scale. = TRUE)` divides by the same sd; a zero-variance
/// column would divide by zero (R hands LAPACK the NaN and fails later), so
/// it is refused here with a message instead.
fn scale_columns(data: &[f64], n: usize, d: usize) -> anyhow::Result<Vec<f64>> {
    let mut xs = vec![0.0f64; n * d];
    for k in 0..d {
        let mut mean = 0.0;
        for i in 0..n {
            mean += data[i * d + k];
        }
        mean /= n as f64;
        let mut ss = 0.0;
        for i in 0..n {
            let dx = data[i * d + k] - mean;
            ss += dx * dx;
        }
        let sd = (ss / (n - 1) as f64).sqrt();
        if sd == 0.0 {
            anyhow::bail!(
                "variable (row factor value) {k} has zero variance; \
                 PCA cannot scale it — remove the constant variable"
            );
        }
        for i in 0..n {
            xs[i * d + k] = (data[i * d + k] - mean) / sd;
        }
    }
    Ok(xs)
}

fn gram_matrix(xs: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut gram = vec![0.0f64; d * d];
    for a in 0..d {
        for b in a..d {
            let mut s = 0.0;
            for i in 0..n {
                s += xs[i * d + a] * xs[i * d + b];
            }
            gram[a * d + b] = s;
            gram[b * d + a] = s;
        }
    }
    gram
}

/// One t-SNE map, Rtsne's defaults as the R operator calls them:
/// `initial_dims = ncol(data)`, `dims = 2`, `pca = TRUE` (center only, no
/// scale), perplexity 30, theta 0.5, 1000 iterations.
///
/// Returns an `n × 2` row-major matrix.
pub fn tsne(data: &[f64], n: usize, d: usize, rng: &mut RRng) -> anyhow::Result<Vec<f64>> {
    const PERPLEXITY: f64 = 30.0;
    const THETA: f64 = 0.5;
    const MAX_ITER: usize = 1000;
    const STOP_LYING_ITER: usize = 250;
    const MOM_SWITCH_ITER: usize = 250;
    const MOMENTUM: f64 = 0.5;
    const FINAL_MOMENTUM: f64 = 0.8;
    const ETA: f64 = 200.0;
    const EXAGGERATION: f64 = 12.0;

    if (n as f64) - 1.0 < 3.0 * PERPLEXITY {
        anyhow::bail!(
            "perplexity {PERPLEXITY} is too large for {n} observations (t-SNE stops when \
             n - 1 < 3 × perplexity)"
        );
    }

    // Rtsne with pca = TRUE: prcomp(X, center = TRUE, scale. = FALSE) —
    // center only, all components (initial_dims = ncol(data)).
    let mut centered = data.to_vec();
    for k in 0..d {
        let mut mean = 0.0;
        for i in 0..n {
            mean += centered[i * d + k];
        }
        mean /= n as f64;
        for i in 0..n {
            centered[i * d + k] -= mean;
        }
    }
    let mut gram = gram_matrix(&centered, n, d);
    let (eigenvalues, vectors) = jacobi_eigen(&mut gram, d);
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());
    let mut mapped = vec![0.0f64; n * d];
    for (kk, &col) in order.iter().enumerate() {
        for i in 0..n {
            let mut s = 0.0;
            for a in 0..d {
                s += centered[i * d + a] * vectors[a * d + col];
            }
            mapped[i * d + kk] = s;
        }
    }

    // normalize_input: center each column, then divide by the max |entry|.
    for k in 0..d {
        let mut mean = 0.0;
        for i in 0..n {
            mean += mapped[i * d + k];
        }
        mean /= n as f64;
        for i in 0..n {
            mapped[i * d + k] -= mean;
        }
    }
    let max_abs = mapped.iter().copied().fold(0.0f64, f64::max);
    if max_abs == 0.0 {
        anyhow::bail!("the data collapses to a single point after PCA; t-SNE is undefined");
    }
    for v in mapped.iter_mut() {
        *v /= max_abs;
    }

    // K nearest neighbours per point, brute force (deterministic order:
    // distance, then index). The vp-tree in Rtsne finds the same set.
    let k_nn = (3.0 * PERPLEXITY) as usize;
    let mut row_p = vec![0usize; n + 1];
    let mut col_p: Vec<u32> = Vec::with_capacity(n * k_nn);
    let mut val_p: Vec<f64> = Vec::with_capacity(n * k_nn);
    let mut row: Vec<(f64, u32)> = Vec::with_capacity(n);
    let mut cur_p = vec![0.0f64; k_nn];
    for i in 0..n {
        row.clear();
        for j in 0..n {
            if j == i {
                continue;
            }
            let mut ss = 0.0;
            for a in 0..d {
                let dx = mapped[i * d + a] - mapped[j * d + a];
                ss += dx * dx;
            }
            row.push((ss, j as u32));
        }
        row.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        // Gaussian calibration of the row to the target perplexity
        // (`computeProbabilities`): the distances are the K neighbours', and
        // the beta search returns a row of probabilities summing to 1.
        let distances: Vec<f64> = row.iter().take(k_nn).map(|e| e.0.sqrt()).collect();
        compute_probabilities(PERPLEXITY, &distances, &mut cur_p);
        for (m, &(_, j)) in row.iter().enumerate().take(k_nn) {
            col_p.push(j);
            val_p.push(cur_p[m]);
        }
        row_p[i + 1] = col_p.len();
    }

    // Symmetrize (`symmetrizeMatrix`), then normalise to sum 1.
    let (row_p, col_p, val_p) = symmetrize(&row_p, &col_p, &val_p, n);
    let sum: f64 = val_p.iter().sum();
    let val_p: Vec<f64> = val_p.iter().map(|&v| v / sum).collect();

    train_iterations(
        &val_p,
        &row_p,
        &col_p,
        n,
        rng,
        THETA,
        MAX_ITER,
        STOP_LYING_ITER,
        MOM_SWITCH_ITER,
        MOMENTUM,
        FINAL_MOMENTUM,
        ETA,
        EXAGGERATION,
    )
}

/// The gradient-descent loop (`trainIterations`), early exaggeration 12 for
/// the first 250 iterations, momentum 0.5 → 0.8.
#[allow(clippy::too_many_arguments)]
fn train_iterations(
    val_p: &[f64],
    row_p: &[usize],
    col_p: &[u32],
    n: usize,
    rng: &mut RRng,
    theta: f64,
    max_iter: usize,
    stop_lying_iter: usize,
    mom_switch_iter: usize,
    momentum_start: f64,
    final_momentum: f64,
    eta: f64,
    exaggeration: f64,
) -> anyhow::Result<Vec<f64>> {
    const D: usize = 2;
    let mut p = val_p.to_vec();
    let mut y = vec![0.0f64; n * D];
    // Rtsne: Y[i] = randn() * .0001 (polar Box–Muller on R's runif)
    for v in y.iter_mut() {
        *v = randn(rng) * 0.0001;
    }
    let mut dy = vec![0.0f64; n * D];
    let mut uy = vec![0.0f64; n * D];
    let mut gains = vec![1.0f64; n * D];
    let mut momentum = momentum_start;

    for v in p.iter_mut() {
        *v *= exaggeration;
    }

    for iter in 0..max_iter {
        if iter == stop_lying_iter {
            for v in p.iter_mut() {
                *v /= exaggeration;
            }
        }
        if iter == mom_switch_iter {
            momentum = final_momentum;
        }

        // Barnes–Hut gradient (`computeGradient`)
        let tree = SPTree::new_root(&y, n);
        let mut pos_f = vec![0.0f64; n * D];
        tree.compute_edge_forces(&y, row_p, col_p, &p, n, &mut pos_f);
        let mut neg_f = vec![0.0f64; n * D];
        let mut sum_q = 0.0;
        // Each point's forces go to its OWN slice of neg_f (the C++ passes
        // `neg_f + n * D`); a shared slice would let every point overwrite
        // the previous one's repulsion and the embedding would collapse into
        // the all-coincident state.
        for i in 0..n {
            sum_q += tree.compute_non_edge_forces(&y, i, theta, &mut neg_f[i * D..(i + 1) * D]);
        }
        for i in 0..n * D {
            dy[i] = pos_f[i] - neg_f[i] / sum_q;
        }
        for i in 0..n * D {
            gains[i] = if sign_tsne(dy[i]) != sign_tsne(uy[i]) {
                gains[i] + 0.2
            } else {
                gains[i] * 0.8
            };
            if gains[i] < 0.01 {
                gains[i] = 0.01;
            }
        }
        for i in 0..n * D {
            uy[i] = momentum * uy[i] - eta * gains[i] * dy[i];
        }
        for i in 0..n * D {
            y[i] += uy[i];
        }
        zero_mean(&mut y, n, D);
    }
    Ok(y)
}

fn sign_tsne(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else if x < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// `computeProbabilities` (tsne.cpp): calibrate one row of the input
/// affinities to the target perplexity. A binary search on the Gaussian
/// precision beta makes the row's Shannon entropy equal log(perplexity) to
/// within 1e-5 (at most 200 iterations, as the C++ does); the row is then
/// normalised to sum 1. The distances are euclidean (not squared).
fn compute_probabilities(perplexity: f64, distances: &[f64], cur_p: &mut [f64]) {
    let k = distances.len();
    let mut beta = 1.0f64;
    let mut min_beta = f64::MIN;
    let mut max_beta = f64::MAX;
    let tol = 1e-5;
    // DBL_MIN in the C++ is the smallest POSITIVE double, not the most negative
    let mut sum_p = f64::MIN_POSITIVE;
    let mut iter = 0;
    while iter < 200 {
        // Gaussian kernel row: exp(-beta * d²)
        for (cur, d) in cur_p.iter_mut().zip(distances.iter()) {
            *cur = (-beta * d * d).exp();
        }
        sum_p = f64::MIN_POSITIVE;
        for v in cur_p.iter().take(k) {
            sum_p += *v;
        }
        let mut h = 0.0;
        for (d, v) in distances.iter().zip(cur_p.iter()) {
            h += beta * (d * d * v);
        }
        h = h / sum_p + sum_p.ln();
        let h_diff = h - perplexity.ln();
        if h_diff < tol && -h_diff < tol {
            break;
        } else if h_diff > 0.0 {
            min_beta = beta;
            beta = if max_beta == f64::MAX || max_beta == f64::MIN {
                beta * 2.0
            } else {
                (beta + max_beta) / 2.0
            };
        } else {
            max_beta = beta;
            beta = if min_beta == f64::MIN || min_beta == f64::MAX {
                beta / 2.0
            } else {
                (beta + min_beta) / 2.0
            };
        }
        iter += 1;
    }
    // Row-normalise with the sum from the last beta
    for v in cur_p.iter_mut().take(k) {
        *v /= sum_p;
    }
}

#[allow(clippy::needless_range_loop)] // small fixed dimension, index math is clearer
fn zero_mean(x: &mut [f64], n: usize, d: usize) {
    let mut mean = vec![0.0f64; d];
    for i in 0..n {
        for k in 0..d {
            mean[k] += x[i * d + k];
        }
    }
    for k in 0..d {
        mean[k] /= n as f64;
    }
    for i in 0..n {
        for k in 0..d {
            x[i * d + k] -= mean[k];
        }
    }
}

/// R's `TSNE::randn`: polar Box–Muller, discarding the second variate. The
/// draws come off R's stream exactly as `R::runif(0, 1)` does.
fn randn(rng: &mut RRng) -> f64 {
    loop {
        let x = 2.0 * rng.unif_rand() - 1.0;
        let y = 2.0 * rng.unif_rand() - 1.0;
        let radius = x * x + y * y;
        if radius >= 1.0 || radius == 0.0 {
            continue;
        }
        let scale = (-2.0 * radius.ln() / radius).sqrt();
        return x * scale;
    }
}

/// `symmetrizeMatrix`, transliterated: build the symmetric CSR graph, then
/// halve the values.
fn symmetrize(
    row_p: &[usize],
    col_p: &[u32],
    val_p: &[f64],
    n: usize,
) -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    let mut row_counts = vec![0usize; n];
    for i in 0..n {
        for &c in &col_p[row_p[i]..row_p[i + 1]] {
            let c = c as usize;
            let present = col_p[row_p[c]..row_p[c + 1]]
                .iter()
                .any(|&m| m as usize == i);
            row_counts[i] += 1;
            if !present {
                row_counts[c] += 1;
            }
        }
    }
    let no_elem: usize = row_counts.iter().sum();
    let mut sym_row = vec![0usize; n + 1];
    for i in 0..n {
        sym_row[i + 1] = sym_row[i] + row_counts[i];
    }
    let mut sym_col = vec![0u32; no_elem];
    let mut sym_val = vec![0.0f64; no_elem];
    let mut offset = vec![0usize; n];
    for i in 0..n {
        for e in row_p[i]..row_p[i + 1] {
            let c = col_p[e] as usize;
            let mut present = false;
            for m in row_p[c]..row_p[c + 1] {
                if col_p[m] as usize == i {
                    present = true;
                    if i <= c {
                        sym_col[sym_row[i] + offset[i]] = col_p[e];
                        sym_col[sym_row[c] + offset[c]] = i as u32;
                        sym_val[sym_row[i] + offset[i]] = val_p[e] + val_p[m];
                        sym_val[sym_row[c] + offset[c]] = val_p[e] + val_p[m];
                    }
                }
            }
            if !present {
                sym_col[sym_row[i] + offset[i]] = col_p[e];
                sym_col[sym_row[c] + offset[c]] = i as u32;
                sym_val[sym_row[i] + offset[i]] = val_p[e];
                sym_val[sym_row[c] + offset[c]] = val_p[e];
            }
            if !present || i <= c {
                offset[i] += 1;
                if c != i {
                    offset[c] += 1;
                }
            }
        }
    }
    for v in sym_val.iter_mut() {
        *v /= 2.0;
    }
    (sym_row, sym_col, sym_val)
}

// ---------------------------------------------------------------------------
// SPTree — transliteration of Rtsne's sptree.cpp (van der Maaten), fixed at
// two dimensions and QT_NODE_CAPACITY = 1.
// ---------------------------------------------------------------------------
pub struct SPTree {
    corner: [f64; 2],
    width: [f64; 2],
    center_of_mass: [f64; 2],
    cum_size: usize,
    size: usize,
    index: [usize; 1],
    is_leaf: bool,
    children: Vec<SPTree>,
}

const NO_CHILDREN: usize = 4;

impl SPTree {
    /// The root constructor: mean and extent of the map, `width = max(max -
    /// mean, mean - min) + 1e-5` per axis, then all points inserted.
    #[allow(clippy::needless_range_loop)] // fixed 2-d indexing over interleaved coordinates
    pub fn new_root(data: &[f64], n: usize) -> SPTree {
        let mut mean = [0.0f64; 2];
        let mut min = [f64::MAX; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        for i in 0..n {
            for k in 0..2 {
                let v = data[i * 2 + k];
                mean[k] += v;
                if v < min[k] {
                    min[k] = v;
                }
                if v > max[k] {
                    max[k] = v;
                }
            }
        }
        for k in 0..2 {
            mean[k] /= n as f64;
        }
        let mut tree = SPTree {
            corner: mean,
            width: [
                f64::max(max[0] - mean[0], mean[0] - min[0]) + 1e-5,
                f64::max(max[1] - mean[1], mean[1] - min[1]) + 1e-5,
            ],
            center_of_mass: [0.0; 2],
            cum_size: 0,
            size: 0,
            index: [0],
            is_leaf: true,
            children: Vec::new(),
        };
        for i in 0..n {
            tree.insert(data, i);
        }
        tree
    }

    fn new_child(corner: [f64; 2], width: [f64; 2]) -> SPTree {
        SPTree {
            corner,
            width,
            center_of_mass: [0.0; 2],
            cum_size: 0,
            size: 0,
            index: [0],
            is_leaf: true,
            children: Vec::new(),
        }
    }

    #[allow(clippy::needless_range_loop)] // fixed 2-d axis indexing
    fn contains_point(&self, point: &[f64; 2]) -> bool {
        for k in 0..2 {
            if self.corner[k] - self.width[k] > point[k]
                || self.corner[k] + self.width[k] < point[k]
            {
                return false;
            }
        }
        true
    }

    #[allow(clippy::needless_range_loop)] // fixed 2-d axis indexing
    fn insert(&mut self, data: &[f64], new_index: usize) -> bool {
        let point = [data[new_index * 2], data[new_index * 2 + 1]];
        if !self.contains_point(&point) {
            return false;
        }

        // Online update of cumulative size and center of mass
        self.cum_size += 1;
        let mult1 = (self.cum_size - 1) as f64 / self.cum_size as f64;
        let mult2 = 1.0 / self.cum_size as f64;
        for k in 0..2 {
            self.center_of_mass[k] = self.center_of_mass[k] * mult1 + mult2 * point[k];
        }

        // Space in this leaf?
        if self.is_leaf && self.size < 1 {
            self.index[0] = new_index;
            self.size += 1;
            return true;
        }

        // Don't add duplicates
        for n in 0..self.size {
            if data[self.index[n] * 2] == point[0] && data[self.index[n] * 2 + 1] == point[1] {
                return true;
            }
        }

        if self.is_leaf {
            self.subdivide(data);
        }
        for child in self.children.iter_mut() {
            if child.insert(data, new_index) {
                return true;
            }
        }
        // Should never happen; the C++ returns false here as well, with the
        // point still counted in `cum_size`.
        false
    }

    fn subdivide(&mut self, data: &[f64]) {
        let (corner, width) = (self.corner, self.width);
        let mut children = Vec::with_capacity(NO_CHILDREN);
        for i in 0..NO_CHILDREN {
            let mut div = 1usize;
            let mut new_corner = [0.0f64; 2];
            let mut new_width = [0.0f64; 2];
            for k in 0..2 {
                new_width[k] = 0.5 * width[k];
                new_corner[k] = if (i / div) % 2 == 1 {
                    corner[k] - 0.5 * width[k]
                } else {
                    corner[k] + 0.5 * width[k]
                };
                div *= 2;
            }
            children.push(SPTree::new_child(new_corner, new_width));
        }
        // Move existing points down; empty the parent afterwards, as the C++
        // does (`index[i] = -1; size = 0; is_leaf = false`).
        let existing: Vec<usize> = (0..self.size).map(|i| self.index[i]).collect();
        self.size = 0;
        self.is_leaf = false;
        self.children = children;
        for idx in existing {
            for child in self.children.iter_mut() {
                if child.insert(data, idx) {
                    break;
                }
            }
        }
    }

    /// `computeNonEdgeForces`: the Barnes–Hut repulsion for one point.
    pub fn compute_non_edge_forces(
        &self,
        data: &[f64],
        point_index: usize,
        theta: f64,
        neg_f: &mut [f64],
    ) -> f64 {
        let mut result_sum = 0.0;
        if self.cum_size == 0 || (self.is_leaf && self.size == 1 && self.index[0] == point_index) {
            return result_sum;
        }
        let mut buff = [0.0f64; 2];
        let mut sqdist = 0.0;
        for k in 0..2 {
            buff[k] = data[point_index * 2 + k] - self.center_of_mass[k];
            sqdist += buff[k] * buff[k];
        }
        let max_width = self.width.iter().copied().fold(0.0f64, f64::max);
        if self.is_leaf || max_width / sqdist.sqrt() < theta {
            let q = 1.0 / (1.0 + sqdist);
            let mult = self.cum_size as f64 * q;
            result_sum += mult;
            let mult = mult * q;
            for k in 0..2 {
                neg_f[k] += mult * buff[k];
            }
        } else {
            for child in self.children.iter() {
                result_sum += child.compute_non_edge_forces(data, point_index, theta, neg_f);
            }
        }
        result_sum
    }

    /// `computeEdgeForces`: the attractive forces along the P graph.
    fn compute_edge_forces(
        &self,
        data: &[f64],
        row_p: &[usize],
        col_p: &[u32],
        val_p: &[f64],
        n: usize,
        pos_f: &mut [f64],
    ) {
        for i in 0..n {
            for e in row_p[i]..row_p[i + 1] {
                let ind2 = col_p[e] as usize * 2;
                let mut buff = [0.0f64; 2];
                let mut sqdist = 1.0;
                for k in 0..2 {
                    buff[k] = data[i * 2 + k] - data[ind2 + k];
                    sqdist += buff[k] * buff[k];
                }
                let q = val_p[e] / sqdist;
                for k in 0..2 {
                    pos_f[i * 2 + k] += q * buff[k];
                }
            }
        }
    }
}

/// Cyclic Jacobi eigen-decomposition of a symmetric d×d matrix (row-major,
/// destroyed on input). Returns (eigenvalues, eigenvectors as columns of the
/// d×d matrix V, with A = V Λ Vᵀ).
pub fn jacobi_eigen(a: &mut [f64], d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0f64; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..d {
            for q in (p + 1)..d {
                off += a[p * d + q].abs();
            }
        }
        if off == 0.0 {
            break;
        }
        for p in 0..d {
            for q in (p + 1)..d {
                let apq = a[p * d + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * d + q] - a[p * d + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // rows p and q: M ← Jᵀ M
                for k in 0..d {
                    let apk = a[p * d + k];
                    let aqk = a[q * d + k];
                    a[p * d + k] = c * apk - s * aqk;
                    a[q * d + k] = s * apk + c * aqk;
                }
                // columns p and q: M ← M J
                for k in 0..d {
                    let akp = a[k * d + p];
                    let akq = a[k * d + q];
                    a[k * d + p] = c * akp - s * akq;
                    a[k * d + q] = s * akp + c * akq;
                }
                // accumulate V ← V · J
                for k in 0..d {
                    let vkp = v[k * d + p];
                    let vkq = v[k * d + q];
                    v[k * d + p] = c * vkp - s * vkq;
                    v[k * d + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let eigenvalues = (0..d).map(|i| a[i * d + i]).collect();
    (eigenvalues, v)
}

#[cfg(test)]
mod tests {
    use super::{compute_probabilities, jacobi_eigen, pca_scaled};

    #[test]
    fn probability_rows_hit_the_target_perplexity() {
        // A realistic row: 90 distances with a cluster of close neighbours
        // and a spread-out tail. After the beta search the Shannon entropy
        // of the row must equal log(perplexity) within the search's 1e-5.
        let distances: Vec<f64> = (0..90usize)
            .map(|i| 0.1 + (i as f64) * 0.08 + ((i * 7) % 5) as f64 * 0.02)
            .collect();
        let mut p = vec![0.0; 90];
        compute_probabilities(30.0, &distances, &mut p);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "row sums to {sum}");
        let h = -p
            .iter()
            .map(|&v| if v > 0.0 { v * v.ln() } else { 0.0 })
            .sum::<f64>();
        assert!(
            (h - 30.0f64.ln()).abs() < 1e-4,
            "entropy {h} vs log(30) = {}",
            30.0f64.ln()
        );
        // the probabilities decay with distance (a Gaussian, not raw values)
        let max_first = p.iter().take(10).cloned().fold(0.0, f64::max);
        let max_last = p.iter().rev().take(10).cloned().fold(0.0, f64::max);
        assert!(
            max_first > max_last * 10.0,
            "far neighbours must be down-weighted"
        );
    }

    #[test]
    fn jacobi_recovers_a_diagonal_and_the_eigen_equation() {
        // A = [[4, 1], [1, 3]], eigenvalues (7 ± √5)/2
        let mut a = vec![4.0, 1.0, 1.0, 3.0];
        let (lambda, v) = jacobi_eigen(&mut a, 2);
        let mut sorted = lambda.clone();
        sorted.sort_by(|x, y| y.partial_cmp(x).unwrap());
        assert!((sorted[0] - (7.0 + 5.0f64.sqrt()) / 2.0).abs() < 1e-12);
        assert!((sorted[1] - (7.0 - 5.0f64.sqrt()) / 2.0).abs() < 1e-12);
        // A v = λ v for the first eigenvector
        let lam = lambda[0];
        let av0 = 4.0 * v[0] + 1.0 * v[2];
        let av1 = 1.0 * v[0] + 3.0 * v[2];
        assert!((av0 - lam * v[0]).abs() < 1e-10);
        assert!((av1 - lam * v[2]).abs() < 1e-10);
    }

    #[test]
    fn pca_projection_preserves_distances_of_the_scaled_input() {
        // 5 points in 3d, all 3 components kept: the pairwise distances of
        // the projection equal those of the centered+scaled input (the PCA
        // rotation cannot change them).
        let data: Vec<f64> = (0..15).map(|x| ((x * 7) % 11) as f64 - 5.0).collect();
        let (n, d) = (5usize, 3usize);
        let mut xs = vec![0.0f64; n * d];
        for k in 0..d {
            let mut mean = 0.0;
            for i in 0..n {
                mean += data[i * d + k];
            }
            mean /= n as f64;
            let mut ss = 0.0;
            for i in 0..n {
                let dx = data[i * d + k] - mean;
                ss += dx * dx;
            }
            let sd = (ss / (n - 1) as f64).sqrt();
            for i in 0..n {
                xs[i * d + k] = (data[i * d + k] - mean) / sd;
            }
        }
        let out = pca_scaled(&data, n, d, 3).unwrap();
        for i in 0..n {
            for j in (i + 1)..n {
                let d1: f64 = (0..d)
                    .map(|k| {
                        let dx = xs[i * d + k] - xs[j * d + k];
                        dx * dx
                    })
                    .sum();
                let d2: f64 = (0..d)
                    .map(|k| {
                        let dx = out[i * 3 + k] - out[j * 3 + k];
                        dx * dx
                    })
                    .sum();
                assert!(
                    ((d1 - d2) / d1.max(1.0)).abs() < 1e-9,
                    "distance {i}-{j}: {d1} vs {d2}"
                );
            }
        }
    }
}

#[cfg(test)]
mod sptree_tests {
    use super::SPTree;

    #[test]
    fn non_edge_forces_match_a_brute_force_sum() {
        // 6 well-separated 2-d points; the Barnes-Hut summarisation must
        // agree with the exact per-point sum to within the approximation's
        // own tolerance when theta is small, and the exact answer at
        // theta = 0 (every leaf visited).
        let data: Vec<f64> = vec![
            -10.0, -10.0, 10.0, -10.0, -10.0, 10.0, 10.0, 10.0, 0.1, 0.05, -0.2, 0.3,
        ];
        let n = 6;
        let tree = SPTree::new_root(&data, n);
        let query = 4usize; // the point near the origin
        for theta in [0.0f64, 0.5] {
            let mut neg_f = [0.0f64; 2];
            let got = tree.compute_non_edge_forces(&data, query, theta, &mut neg_f);
            let mut want = 0.0;
            let mut want_f = [0.0f64; 2];
            for j in 0..n {
                if j == query {
                    continue;
                }
                let mut sqdist = 0.0;
                let mut buff = [0.0f64; 2];
                for k in 0..2 {
                    buff[k] = data[query * 2 + k] - data[j * 2 + k];
                    sqdist += buff[k] * buff[k];
                }
                let q = 1.0 / (1.0 + sqdist);
                want += q;
                for k in 0..2 {
                    want_f[k] += q * q * buff[k];
                }
            }
            assert!(
                (got - want).abs() < 1e-9,
                "theta={theta}: sum_q {got} vs exact {want}"
            );
            for k in 0..2 {
                assert!(
                    (neg_f[k] - want_f[k]).abs() < 1e-6,
                    "theta={theta}: neg_f[{k}] {} vs exact {}",
                    neg_f[k],
                    want_f[k]
                );
            }
        }
    }
}
