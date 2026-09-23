//! clusterx_operator — Rust port of `tercen/clusterx_operator` (the ClusterX
//! density-peak clustering, Rodriguez & Laio 2014, as packaged by
//! JinmiaoChenLab/ClusterX).
//!
//! The crosstab contract: rows are variables (channels, markers), columns are
//! observations (cells, samples), y is the measurement. The operator gathers
//! the crosstab into the matrix the R operator builds with
//! `acast(.ci ~ .ri, mean)` — one point per observation — clusters it, and
//! returns one cluster label per observation column on `.ci`.
pub mod algorithm;
pub mod chunked;
pub mod context;
pub mod dimred;
pub mod input;
pub mod output;
pub mod pagecache;
pub mod progress;
pub mod props;
pub mod rng;
pub mod special;
pub mod tson;
pub mod upload;

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use tercen_rs::context::ContextBase;
use tercen_rs::{DevContext, TercenClient};

use progress::Reporter;
use props::DimReduction;
use rng::RRng;

pub fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

pub fn require_env(name: &str) -> Result<String> {
    std::env::var(name).map_err(|_| anyhow!("{name} is not set"))
}

/// Production entry point (`--taskId`).
pub async fn run(task_id: &str) -> Result<()> {
    tracing::info!("clusterx_operator starting (task_id={task_id})");
    let client = build_client().await?;
    // Deliberately not `ProductionContext::from_task_id`: see context.rs.
    let ctx = context::from_task_id(client, task_id).await?;
    execute(
        &ctx,
        Mode::Production {
            task_id: task_id.to_string(),
        },
    )
    .await
}

/// Dev entry point (`WORKFLOW_ID` / `STEP_ID`).
pub async fn run_dev(workflow_id: &str, step_id: &str) -> Result<()> {
    tracing::info!("clusterx_operator starting in dev mode ({workflow_id} / {step_id})");
    let client = build_client().await?;
    let ctx = DevContext::from_workflow_step(client, workflow_id, step_id)
        .await
        .map_err(|e| anyhow!("load workflow {workflow_id} / step {step_id}: {e}"))?;
    execute(
        &ctx,
        Mode::Dev {
            workflow_id: workflow_id.to_string(),
            step_id: step_id.to_string(),
        },
    )
    .await
}

enum Mode {
    Production {
        task_id: String,
    },
    Dev {
        workflow_id: String,
        step_id: String,
    },
}

async fn build_client() -> Result<Arc<TercenClient>> {
    let client = TercenClient::from_env()
        .await
        .map_err(|e| anyhow!("connect to Tercen: {e}"))?;
    tracing::info!("connected to Tercen");
    Ok(Arc::new(client))
}

async fn execute(ctx: &ContextBase, mode: Mode) -> Result<()> {
    let t_start = Instant::now();
    tracing::info!(
        workflow = ctx.workflow_id(),
        step = ctx.step_id(),
        namespace = ctx.namespace(),
        "context loaded"
    );
    let rep = match &mode {
        Mode::Production { task_id } => Reporter::spawn(Arc::clone(ctx.client()), task_id.clone()),
        Mode::Dev { .. } => Reporter::silent(),
    };
    let s = props::settings_from_ctx(ctx)?;
    tracing::info!(?s, "properties");

    // The whole matrix must be gathered: the density of a point depends on
    // every other point (see memory_model.json for the booking).
    rep.at(progress::READ.0, "Reading the crosstab");
    let mat = input::gather_matrix(ctx).await?;
    let n = mat.n_points();
    let d = mat.n_vars();
    tracing::info!(points = n, variables = d, "crosstab gathered");
    for (i, v) in mat.data.iter().enumerate() {
        if !v.is_finite() {
            // The R reference fills missing cells with NaN and computes with
            // them (every label comes back NA); refusing says what happened.
            anyhow::bail!(
                "the crosstab has a missing measurement (cell {i} of {n} × {d} is NaN after \
                 aggregation); every observation × variable pair must have a value"
            );
        }
    }

    let mut rng = make_rng(s.seed);
    rep.at(
        progress::WRITE.0,
        match s.dim_reduction {
            DimReduction::Null => "Clustering".to_string(),
            DimReduction::Pca => format!("PCA to {0} dimensions, then clustering", s.out_dim),
            DimReduction::Tsne => "t-SNE to 2 dimensions, then clustering".to_string(),
        },
    );

    let t = Instant::now();
    let result = match s.dim_reduction {
        DimReduction::Null => algorithm::clusterx(&mat.data, n, d, ALPHA, &mut rng)?,
        DimReduction::Pca => {
            let mapped = dimred::pca_scaled(&mat.data, n, d, s.out_dim)?;
            algorithm::clusterx(&mapped, n, mapped.len() / n, ALPHA, &mut rng)?
        }
        DimReduction::Tsne => {
            let mapped = dimred::tsne(&mat.data, n, d, &mut rng)?;
            algorithm::clusterx(&mapped, n, 2, ALPHA, &mut rng)?
        }
    };
    tracing::info!(
        dc = result.dc,
        peaks = result.peak_id.len(),
        secs = format!("{:.1}", t.elapsed().as_secs_f64()),
        "clustering done"
    );
    rep.info(format!(
        "ClusterX: dc = {:.4}, {} peak(s) over {n} observations",
        result.dc,
        result.peak_id.len()
    ));

    let work_root = std::env::temp_dir().join(format!(
        "clusterx_op_{}_{}",
        ctx.workflow_id(),
        ctx.step_id()
    ));
    std::fs::create_dir_all(&work_root)
        .with_context(|| format!("create {}", work_root.display()))?;
    let result_path = work_root.join("result.tson");
    {
        let f = std::fs::File::create(&result_path)
            .with_context(|| format!("create {}", result_path.display()))?;
        let mut w = tson::TsonWriter::new(std::io::BufWriter::with_capacity(1 << 20, f))?;
        output::write_result(&mut w, &uuid_like(ctx), &result.cluster, ctx.namespace())?;
    }

    rep.at(progress::UPLOAD.0, "Uploading the result");
    match mode {
        Mode::Production { task_id } => {
            upload::save_production(ctx, &task_id, &result_path, &rep).await?
        }
        Mode::Dev {
            workflow_id,
            step_id,
        } => {
            let saved = upload::save_dev(ctx, &workflow_id, &step_id, &result_path).await?;
            tracing::info!(
                task_id = saved.task_id,
                file_id = saved.file_id,
                "dev result saved"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&work_root);
    rep.at(100, "Done");
    tracing::info!(
        total_secs = format!("{:.1}", t_start.elapsed().as_secs_f64()),
        peak_rss_kb = peak_rss_kb().unwrap_or(0),
        "done"
    );
    Ok(())
}

/// The ESD significance level the R operator passes (`alpha = 0.001`).
const ALPHA: f64 = 0.001;

/// `seed < 0` is R's "a random seed": the reference then runs on whatever
/// state R's session RNG has, which is not reproducible; here the seed comes
/// from OS entropy instead. A seed >= 0 means `set.seed(as.integer(seed))`.
fn make_rng(seed: i32) -> RRng {
    if seed >= 0 {
        RRng::set_seed(seed as u32)
    } else {
        use std::hash::{BuildHasher, Hasher};
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
        );
        RRng::set_seed(h.finish() as u32)
    }
}

/// A stable per-run table name (`save_table` uses a uuid; the value is never read back).
fn uuid_like(ctx: &ContextBase) -> String {
    format!("{}_{}", ctx.step_id(), ctx.qt_hash())
}

fn peak_rss_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    s.lines()
        .find(|l| l.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}
