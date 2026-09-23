//! Operator properties, with the R operator's defaults
//! (`clusterx_operator/main.R`): `dimReduction` (`tsne`, `pca`, `NULL`),
//! `outDim` (2) and `seed` (-1 = random).
use anyhow::{Result, bail};
use tercen_rs::PropertyReader;
use tercen_rs::context::ContextBase;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimReduction {
    /// Cluster the projected measurements directly.
    Null,
    /// `prcomp(data, scale. = TRUE)`, then cluster the first `out_dim` axes.
    Pca,
    /// Barnes–Hut t-SNE to 2 axes, then cluster them.
    Tsne,
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub dim_reduction: DimReduction,
    pub out_dim: usize,
    /// R: `seed < 0` means "a random seed"; otherwise `set.seed(as.integer(seed))`.
    pub seed: i32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            dim_reduction: DimReduction::Null,
            out_dim: 2,
            seed: -1,
        }
    }
}

/// Read the properties off the task's `CubeQueryTask` snapshot.
pub fn settings_from_ctx(ctx: &ContextBase) -> Result<Settings> {
    let pr = PropertyReader::from_operator_settings(ctx.operator_settings());
    // Tercen serialises a numeric property as e.g. "2.0" even where the
    // operator wants an integer, so parse every number as f64 and cast
    // (create-rust-operator §2).
    let num = |name: &str, default: f64| -> Result<f64> {
        let raw = pr.get_string(name, &default.to_string());
        raw.trim()
            .parse::<f64>()
            .map_err(|_| anyhow::anyhow!("property '{name}' is not a number: '{raw}'"))
    };
    let dim_reduction = match pr.get_string("dimReduction", "NULL").trim() {
        "tsne" => DimReduction::Tsne,
        "pca" => DimReduction::Pca,
        "NULL" => DimReduction::Null,
        other => {
            bail!("property 'dimReduction' must be 'tsne', 'pca' or 'NULL', got '{other}'")
        }
    };
    let out_dim = num("outDim", 2.0)?;
    if !(out_dim.is_finite() && out_dim >= 1.0) {
        bail!("property 'outDim' must be a number >= 1, got {out_dim}");
    }
    let seed = num("seed", -1.0)?;
    if !seed.is_finite() {
        bail!("property 'seed' must be a finite number, got {seed}");
    }
    Ok(Settings {
        dim_reduction,
        out_dim: out_dim as usize,
        seed: seed.floor() as i32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_r_operator() {
        let s = Settings::default();
        assert_eq!(s.dim_reduction, DimReduction::Null);
        assert_eq!(s.out_dim, 2);
        assert_eq!(s.seed, -1);
    }
}
