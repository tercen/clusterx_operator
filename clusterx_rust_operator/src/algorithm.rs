//! The ClusterX algorithm (Rodriguez & Laio 2014, with the automatic peak
//! detection of the ClusterX R package), transliterated from
//! `reference/ClusterX.R` (JinmiaoChenLab/ClusterX 0.99.1) — the reference the
//! R operator runs. Pure functions over a row-major `n × d` matrix; no Tercen
//! types. Parity with the reference is decided by `tests/parity/`.
//!
//! One deliberate departure, a guard where the R reference loops forever:
//! `estimate_dc` refuses to iterate past [`MAX_DC_ITER`] (R's `while(TRUE)`
//! can never converge for a few points, or for data whose neighbour rate
//! jumps over the target band). A matrix with non-finite values is rejected
//! by the caller; R computes with `NaN`s and produces all-`NA` labels.

use crate::rng::RRng;

/// Cap on the binary search for the distance cutoff.
const MAX_DC_ITER: usize = 100_000;

/// `estimateDc`'s `sampleSize` at the R call site (the function's own default
/// of 5000 is overridden by ClusterX).
pub const DC_SAMPLE_SIZE: usize = 10_000;

#[derive(Debug, Clone)]
pub struct ClusterXResult {
    /// Cluster label per point, **1-based** like R's `match(i, peakID)`;
    /// `None` where R leaves `NA` (a chain that runs into a density tie it
    /// loses).
    pub cluster: Vec<Option<usize>>,
    pub dc: f64,
    pub rho: Vec<f64>,
    pub delta: Vec<f64>,
    /// 0-based index of the nearest higher-density point; for a
    /// highest-density point R's scan returns the first index, and so does
    /// this.
    pub higher_id: Vec<usize>,
    /// 0-based peak indices, in peak order (the order defines the labels).
    pub peak_id: Vec<usize>,
}

/// `ClusterX(data, dimReduction=NULL, ...)` with the R operator's arguments:
/// `dc` estimated, `gaussian = TRUE`, `alpha = 0.001`, no halos, no SVM.
///
/// `rng` is R's RNG; the reference consumes it only when `n > DC_SAMPLE_SIZE`
/// (the down-sample in `estimateDc`) — the fold structure of the pairwise
/// chunks does not change any value this port keeps (see `min_dist_to_higher`).
pub fn clusterx(
    data: &[f64],
    n: usize,
    d: usize,
    alpha: f64,
    rng: &mut RRng,
) -> anyhow::Result<ClusterXResult> {
    let dc = estimate_dc(data, n, d, DC_SAMPLE_SIZE, rng)?;
    let rho = local_density(data, n, d, dc, rng);
    let (delta, higher_id) = min_dist_to_higher(data, n, d, &rho, rng);
    let mut peak_id = peak_detect(&rho, &delta, alpha);

    let mut cluster = cluster_assign(&peak_id, &higher_id, &rho, n);

    // Noise-cluster removal, with the R source's own two thresholds: the
    // *trigger* tests 0.0005 but the *removal* keeps clusters of at least
    // `n * alpha`. Reproduced as written.
    let n_labels = peak_id.len();
    let mut counts = vec![0usize; n_labels];
    for c in cluster.iter().flatten() {
        counts[c - 1] += 1;
    }
    let nf = n as f64;
    if counts.iter().any(|&c| (c as f64) < nf * 0.0005) {
        peak_id = peak_id
            .iter()
            .enumerate()
            .filter(|(k, _)| counts[*k] as f64 >= nf * alpha)
            .map(|(_, &p)| p)
            .collect();
        cluster = cluster_assign(&peak_id, &higher_id, &rho, n);
    }

    Ok(ClusterXResult {
        cluster,
        dc,
        rho,
        delta,
        higher_id,
        peak_id,
    })
}

/// `estimateDc`: binary search for the distance cutoff whose neighbour rate
/// (mean fraction of *other* points within `dc`) falls in (0.01, 0.02).
///
/// For `n > sample_size` the reference draws the sample with
/// `sample(1:n, sampleSize, replace = FALSE)` — R's partial Fisher–Yates —
/// which [`RRng::sample_int`] reproduces. R holds the m×m distance matrix
/// (800 MB at the 10 000-point cap) and re-runs `rowSums` over it per step;
/// this stores the upper triangle once (each off-diagonal distance appears
/// twice in the square, the diagonal is zero) and counts over it. Same
/// numbers, same summation order, half the memory.
pub fn estimate_dc(
    data: &[f64],
    n: usize,
    d: usize,
    sample_size: usize,
    rng: &mut RRng,
) -> anyhow::Result<f64> {
    let m = n.min(sample_size);
    anyhow::ensure!(
        m >= 2,
        "the projection has {} observation(s); at least 2 are needed",
        m
    );
    let sample: Vec<usize> = if n > sample_size {
        rng.sample_int(n, sample_size)
    } else {
        (0..n).collect()
    };

    // Upper triangle of the sampled pairwise distances.
    let t = m * (m - 1) / 2;
    let mut tri: Vec<f64> = Vec::with_capacity(t);
    for a in 0..m {
        for b in (a + 1)..m {
            tri.push(euclidean(data, sample[a], sample[b], d));
        }
    }

    // R: dcMod = median(comb) * 0.05 over all m² entries (diagonal zeros
    // included); dcL = min(comb) = 0 through the diagonal. The median of the
    // square's multiset {0 × m} ∪ {each triangle value × 2}: ranks below m
    // are zero, anything above is the (r − m)/2-th smallest triangle value.
    let total = m * m;
    let median = {
        let mut kth = |r: usize| -> f64 {
            if r < m {
                0.0
            } else {
                select_nth(&mut tri, (r - m) / 2)
            }
        };
        let (r1, r2) = if total.is_multiple_of(2) {
            (total / 2 - 1, total / 2)
        } else {
            (total / 2, total / 2)
        };
        (kth(r1) + kth(r2)) / 2.0
    };

    let mut dc_mod = median * 0.05;
    let mut dc = dc_mod;
    let mut dc_last = 0.0f64;

    for _ in 0..MAX_DC_ITER {
        let rate = neighbour_rate(&tri, m, dc);
        // R (ClusterX.R): `if(neighborRate > neighborRateLow && neighborRate < neighborRateHigh)`
        // — strict on both ends; a rate of exactly 0.01 or 0.02 keeps searching.
        if rate > 0.01 && rate < 0.02 {
            return Ok(dc);
        }
        if rate >= 0.02 {
            let next = (dc + dc_last) / 2.0;
            dc_last = dc;
            dc = next;
            dc_mod /= 2.0;
        } else {
            dc_last = dc;
            dc += dc_mod;
        }
    }
    anyhow::bail!(
        "the distance cutoff search did not converge in {MAX_DC_ITER} steps: with {m} points no \
         cutoff puts the neighbour rate inside (0.01, 0.02). The R reference loops forever on \
         this shape; project more observations."
    )
}

/// `mean((rowSums(comb < dc) - 1) / size)` over the sampled rows, in row
/// order, exactly as R computes it from the full distance matrix; the
/// diagonal (0 < dc) is the +1 that the −1 removes, so counting the triangle
/// once per unordered pair gives the same per-row counts.
fn neighbour_rate(tri: &[f64], m: usize, dc: f64) -> f64 {
    let mut count = vec![0u64; m];
    let mut e = 0usize;
    for a in 0..m {
        for b in (a + 1)..m {
            if tri[e] < dc {
                count[a] += 1;
                count[b] += 1;
            }
            e += 1;
        }
    }
    let mut sum = 0.0;
    for c in count.iter() {
        sum += *c as f64 / m as f64;
    }
    sum / m as f64
}

/// Iterative quickselect (in place), returning the k-th smallest (0-based).
fn select_nth(v: &mut [f64], k: usize) -> f64 {
    let (mut lo, mut hi) = (0usize, v.len() - 1);
    loop {
        if lo == hi {
            return v[lo];
        }
        let pivot = v[(lo + hi) / 2];
        let (mut i, mut j) = (lo, hi);
        while i <= j {
            while v[i] < pivot {
                i += 1;
            }
            while v[j] > pivot {
                j -= 1;
            }
            if i <= j {
                v.swap(i, j);
                i += 1;
                if j == 0 {
                    break;
                }
                j -= 1;
            }
        }
        if k <= j {
            hi = j;
        } else if k >= i {
            lo = i;
        } else {
            return v[k];
        }
    }
}

/// The `pdist` package computes entirely in C `float` — the matrix values
/// are coerced to f32, the squared differences accumulate in f32, and the
/// square root is f32 (1.3905853336294187 comes back as 1.3905854225158691).
/// The reference's density and delta run on those values, so a port that
/// wants its labels has to reproduce the arithmetic exactly as it is.
/// `estimateDc` uses base R's `dist`, which is double, and is left alone.
fn pdist(data: &[f64], a: usize, b: usize, d: usize) -> f64 {
    let mut distance = 0.0f32;
    for k in 0..d {
        let xi = data[a * d + k] as f32;
        let yi = data[b * d + k] as f32;
        let diff = xi - yi;
        distance += diff * diff;
    }
    distance.sqrt() as f64
}

/// Euclidean distance between rows `a` and `b`, the way R's `dist` and
/// `pdist` compute it: sum the squared component differences in component
/// order, then take one square root.
pub fn euclidean(data: &[f64], a: usize, b: usize, d: usize) -> f64 {
    let (x, y) = (&data[a * d..a * d + d], &data[b * d..b * d + d]);
    let mut ss = 0.0;
    for k in 0..d {
        let dx = x[k] - y[k];
        ss += dx * dx;
    }
    ss.sqrt()
}

/// `localDensity` with the gaussian kernel the R operator uses:
/// `rho_i = sum_j exp(-(d_ij / dc)^2) - 1`, summed over j in order (the `- 1`
/// removes the self term `exp(0)`). R's `x^2` is `x * x`, hence `q * q`.
///
/// R splits the rows into random folds first (`splitFactorGenerator`); the
/// fold never changes a row's value, but it consumes RNG draws, so the draw
/// is taken here to keep the stream aligned with the reference.
#[allow(clippy::needless_range_loop)] // row i is the sum over every column j
pub fn local_density(data: &[f64], n: usize, d: usize, dc: f64, rng: &mut RRng) -> Vec<f64> {
    let _folds = split_folds(n, rng);
    let mut rho = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            let dist = pdist(data, i, j, d);
            let q = dist / dc;
            sum += (-(q * q)).exp();
        }
        rho[i] = sum - 1.0;
    }
    rho
}

/// `splitFactorGenerator(rowNum)` (colNum missing → square): fold size
/// `round(65545326 / rowNum)`, fold count `ceiling(rowNum / foldSize)`, and
/// one fold per row drawn with `sample(1:foldNum, rowNum, replace = TRUE)`.
/// Returns the fold id of each row, in row order.
fn split_folds(n: usize, rng: &mut RRng) -> Vec<usize> {
    let fold_size = (65_545_326.0 / n as f64).round() as usize;
    let fold_num = n.div_ceil(fold_size);
    (0..n)
        .map(|_| rng.unif_index(fold_num as f64) as usize + 1)
        .collect()
}

/// `minDistToHigher`: distance to the nearest strictly-higher-density point.
///
/// R multiplies the distances by the density comparison, replaces the zeros
/// with its chunk maximum and takes the row minimum. The chunk maximum is at
/// least every distance in the row, so for a point with a strictly denser
/// neighbour the real distances win and the value is fold-independent. A
/// maximum-density point has no such neighbour, so its delta **is** the
/// chunk maximum — and the chunks are random folds (`splitFactorGenerator`,
/// RNG-drawn), which is why that one value per maximum-density point is
/// fold-structured. The folds are replicated here, so the value matches the
/// reference whenever the reference was seeded (the `n ≤ 8096` case has a
/// single fold and is deterministic regardless).
pub fn min_dist_to_higher(
    data: &[f64],
    n: usize,
    d: usize,
    rho: &[f64],
    rng: &mut RRng,
) -> (Vec<f64>, Vec<usize>) {
    let folds = split_folds(n, rng);
    // per-fold maximum over the chunk × all-points distance matrix
    let mut fold_max = vec![f64::MIN; n + 1];
    for (i, &f) in folds.iter().enumerate() {
        for j in 0..n {
            let dist = pdist(data, i, j, d);
            if dist > fold_max[f] {
                fold_max[f] = dist;
            }
        }
    }
    let mut delta = vec![0.0f64; n];
    let mut higher = vec![0usize; n];
    for i in 0..n {
        let chunk_max = fold_max[folds[i]];
        // R: drMix = dist * (rho_i < rho_j), zeros replaced by the chunk
        // maximum; min and which.min over the row, in column order.
        let mut best = chunk_max;
        let mut best_j = 0usize;
        for j in 0..n {
            let dist = pdist(data, i, j, d);
            let value = if rho[i] < rho[j] { dist } else { chunk_max };
            if value < best {
                best = value;
                best_j = j;
            }
        }
        delta[i] = best;
        higher[i] = best_j;
    }
    (delta, higher)
}

/// `peakDetect`: replace infinite deltas, scale rho to [0, 1], multiply by
/// delta, and keep the points the generalised ESD test flags as anomalous in
/// **both** the combined index and delta itself.
pub fn peak_detect(rho: &[f64], delta: &[f64], alpha: f64) -> Vec<usize> {
    let mut delta = delta.to_vec();
    let finite_max = delta
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    for x in delta.iter_mut() {
        if x.is_infinite() {
            *x = finite_max * finite_max;
        }
    }

    let rho_min = rho.iter().copied().fold(f64::INFINITY, f64::min);
    let rho_max = rho.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = rho_max - rho_min;
    let rd_index: Vec<f64> = rho
        .iter()
        .zip(delta.iter())
        .map(|(&r, &dl)| ((r - rho_min) / span) * dl)
        .collect();

    let peaks1 = detect_anoms_sd(&rd_index, alpha);
    let peaks2 = detect_anoms_sd(&delta, alpha);
    // intersect, preserving the first vector's order (R's `intersect`)
    let mut out = Vec::new();
    for &p in &peaks1 {
        if peaks2.contains(&p) && !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

/// `detect_anoms_sd`: generalised ESD, `direction = "pos"`, `max_anoms = 0.1`.
/// Returns the flagged original indices (0-based), in detection order, with
/// tied maxima all flagged as R's `which(ares == maxAres)` does.
pub fn detect_anoms_sd(data: &[f64], alpha: f64) -> Vec<usize> {
    let num_obs = data.len();
    let max_outliers = (num_obs as f64 * 0.1).trunc() as usize;
    let mut anoms: Vec<usize> = Vec::new();
    // (original index, value), with R's `data <- data[-maxAres_id]` removals
    let mut current: Vec<(usize, f64)> = data.iter().copied().enumerate().collect();

    for i in 1..=max_outliers {
        if current.is_empty() {
            break;
        }
        let mean = mean_of(current.iter().map(|e| e.1));
        let sigma = sd_of(current.iter().map(|e| e.1), mean);
        if sigma == 0.0 {
            break;
        }
        let mut max_ares = f64::NEG_INFINITY;
        let mut max_positions: Vec<usize> = Vec::new();
        for (pos, e) in current.iter().enumerate() {
            let ares = (e.1 - mean) / sigma;
            if ares > max_ares {
                max_ares = ares;
                max_positions.clear();
                max_positions.push(pos);
            } else if ares == max_ares {
                max_positions.push(pos);
            }
        }
        // p and df use the ORIGINAL observation count, as the R source does
        let p = 1.0 - alpha / (num_obs - i + 1) as f64;
        let df = (num_obs - i - 1) as f64;
        let t = crate::special::qt(p, df);
        let lam = t * (num_obs - i) as f64
            / (((num_obs - i - 1) as f64 + t * t) * (num_obs - i + 1) as f64).sqrt();
        if max_ares > lam {
            for &pos in &max_positions {
                anoms.push(current[pos].0);
            }
            let removed = max_positions.clone();
            let mut keep = Vec::with_capacity(current.len() - removed.len());
            let mut ri = 0usize;
            for (pos, e) in current.iter().enumerate() {
                if ri < removed.len() && removed[ri] == pos {
                    ri += 1;
                    continue;
                }
                keep.push(*e);
            }
            current = keep;
        } else {
            break;
        }
    }
    anoms
}

/// `clusterAssign`: walk the points in decreasing density; a peak takes its
/// label (its position among the peaks), everything else inherits the label
/// of its nearest denser neighbour. A chain that runs into a point that has
/// not been labelled yet — possible when equal densities break the order —
/// stays `None`, exactly as R's `NA` propagates.
pub fn cluster_assign(
    peak_id: &[usize],
    higher_id: &[usize],
    rho: &[f64],
    n: usize,
) -> Vec<Option<usize>> {
    let mut run_order: Vec<usize> = (0..n).collect();
    // R's `order(rho, decreasing = TRUE)` is stable: ties keep index order.
    run_order.sort_by(|&a, &b| {
        rho[b]
            .partial_cmp(&rho[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut label = vec![None; n];
    for i in run_order {
        if let Some(pos) = peak_id.iter().position(|&p| p == i) {
            label[i] = Some(pos + 1);
        } else {
            label[i] = label[higher_id[i]];
        }
    }
    label
}

/// Arithmetic mean, summed in order. R's `mean` accumulates in long double,
/// so this can differ in the last bit; the ESD decisions it feeds sit far
/// from that (see CLAUDE.md).
fn mean_of(xs: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut n) = (0.0f64, 0usize);
    for x in xs {
        sum += x;
        n += 1;
    }
    sum / n as f64
}

/// Sample standard deviation (R's `sd`), two-pass around `mean`.
fn sd_of(xs: impl Iterator<Item = f64>, mean: f64) -> f64 {
    let (mut ss, mut n) = (0.0f64, 0usize);
    for x in xs {
        let dx = x - mean;
        ss += dx * dx;
        n += 1;
    }
    if n < 2 {
        return f64::NAN;
    }
    (ss / (n - 1) as f64).sqrt()
}
