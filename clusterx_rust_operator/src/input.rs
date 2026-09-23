//! Reading the crosstab and gathering it into the matrix the algorithm
//! clusters.
//!
//! The projection (R `main.R`): rows are variables (channels, markers),
//! columns are observations (cells, samples), y is the measurement. R builds
//! its data with `reshape2::acast(.ci ~ .ri, value.var = ".y", fill = NaN,
//! fun.aggregate = mean)` — one point per **column** factor value, one
//! dimension per **row** factor value, duplicates averaged, missing cells
//! `NaN` — and clusters the resulting rows. This gathers the same matrix
//! from streamed chunks, keeping only the value sums and counts until the
//! end (the matrix itself is `n_columns × n_rows` f64, booked in the memory
//! model).
use anyhow::{Result, bail};
use std::collections::HashMap;

use tercen_rs::context::ContextBase;

pub use crate::chunked::{cell_count, for_each_chunk};

/// The gathered crosstab, in the layout the R operator hands to `ClusterX`.
pub struct CrosstabMatrix {
    /// Row-major `n_points × n_vars`; entry `(i, k)` is the mean y of the
    /// cells whose column is `ci_values[i]` and whose row is `ri_values[k]`.
    pub data: Vec<f64>,
    /// Sorted distinct `.ci` values (the points), ascending.
    pub ci_values: Vec<i32>,
    /// Sorted distinct `.ri` values (the variables), ascending.
    pub ri_values: Vec<i32>,
}

impl CrosstabMatrix {
    pub fn n_points(&self) -> usize {
        self.ci_values.len()
    }
    pub fn n_vars(&self) -> usize {
        self.ri_values.len()
    }
}

/// Stream the whole cell table and aggregate it exactly like
/// `acast(.ci ~ .ri, fun.aggregate = mean, fill = NaN)`.
pub async fn gather_matrix(ctx: &ContextBase) -> Result<CrosstabMatrix> {
    let n_cells = cell_count(ctx).await?;
    let mut sums: HashMap<(i32, i32), (f64, u32)> = HashMap::with_capacity(n_cells);
    for_each_chunk(ctx, &[".ri", ".ci", ".y"], n_cells, CHUNK, |c| {
        for e in 0..c.y.len() {
            let entry = sums.entry((c.ci[e], c.ri[e])).or_insert((0.0, 0));
            entry.0 += c.y[e];
            entry.1 += 1;
        }
        Ok(())
    })
    .await?;

    let mut ci_values: Vec<i32> = sums.keys().map(|k| k.0).collect();
    ci_values.sort_unstable();
    ci_values.dedup();
    let mut ri_values: Vec<i32> = sums.keys().map(|k| k.1).collect();
    ri_values.sort_unstable();
    ri_values.dedup();
    let n_points = ci_values.len();
    let n_vars = ri_values.len();
    if n_points < 2 {
        bail!(
            "the projection has {n_points} observation column(s); the clustering needs at \
             least 2"
        );
    }
    if n_vars < 1 {
        bail!("the projection has no variable row factor values");
    }

    let mut ci_rank: HashMap<i32, usize> = HashMap::with_capacity(n_points);
    for (i, &v) in ci_values.iter().enumerate() {
        ci_rank.insert(v, i);
    }
    let mut ri_rank: HashMap<i32, usize> = HashMap::with_capacity(n_vars);
    for (k, &v) in ri_values.iter().enumerate() {
        ri_rank.insert(v, k);
    }

    let mut data = vec![f64::NAN; n_points * n_vars];
    for ((ci, ri), (sum, count)) in &sums {
        let (i, k) = (ci_rank[ci], ri_rank[ri]);
        data[i * n_vars + k] = sum / *count as f64;
    }
    Ok(CrosstabMatrix {
        data,
        ci_values,
        ri_values,
    })
}

/// Cells fetched per gRPC round trip (same measurement as asinh's).
pub const CHUNK: usize = 200_000;
