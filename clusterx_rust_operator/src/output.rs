//! The result: one table, `.ci` / `<namespace>.cluster`, one row per
//! observation column of the crosstab — a per-column result, joined by the
//! server on `.ci` (the `Observation` pair the manifest declares).
//!
//! The result is tiny (one row per observation), so it is written to a
//! buffer, not streamed to disk; the upload path is the same as asinh's.
use std::io::Write;

use anyhow::Result;

use crate::tson::TsonWriter;

pub struct ColSpec<'a> {
    pub name: &'a str,
    pub ty: &'a str, // "int32" | "string"
}

/// The two columns of the result: the index the server joins on, then the
/// namespaced label (R `ctx$addNamespace()`).
pub fn result_columns(namespace: &str) -> [ColSpec<'_>; 2] {
    [
        ColSpec {
            name: ".ci",
            ty: "int32",
        },
        ColSpec {
            name: namespace,
            ty: "string",
        },
    ]
}

/// The result column's name: `<namespace>.cluster`.
pub fn cluster_column(namespace: &str) -> String {
    format!("{namespace}.cluster")
}

/// R `paste0("cluster", cluster)`, with R's `NA` rendering for the labels
/// the reference leaves `NA`.
pub fn label(cluster: Option<usize>) -> String {
    match cluster {
        Some(k) => format!("cluster{k}"),
        None => "clusterNA".to_string(),
    }
}

/// Encode the whole `OperatorResult` (header, both columns, footer) into `w`.
pub fn write_result<W: Write>(
    w: &mut TsonWriter<W>,
    table_name: &str,
    labels: &[Option<usize>],
    namespace: &str,
) -> Result<()> {
    let column = cluster_column(namespace);
    let cols = result_columns(&column);
    let n = labels.len();
    write_header(w, table_name, n, &cols)?;
    write_column_header(w, &cols[0], n)?;
    w.i32_list(&(0..n as i32).collect::<Vec<_>>())?;
    write_column_header(w, &cols[1], n)?;
    let strings: Vec<String> = labels.iter().map(|l| label(*l)).collect();
    w.str_list(&strings.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
    w.key("joinOperators")?;
    w.list(0)?;
    w.flush()?;
    Ok(())
}

fn write_header<W: Write>(
    w: &mut TsonWriter<W>,
    table_name: &str,
    n_rows: usize,
    cols: &[ColSpec],
) -> Result<()> {
    w.map(3)?;
    w.key("kind")?;
    w.str("OperatorResult")?;
    w.key("tables")?;
    w.list(1)?;

    w.map(4)?;
    w.key("kind")?;
    w.str("Table")?;
    w.key("nRows")?;
    w.i32(i32::try_from(n_rows).map_err(|_| {
        anyhow::anyhow!(
            "the result would have {n_rows} rows, more than a Tercen table can hold (i32::MAX)"
        )
    })?)?;
    w.key("properties")?;
    w.map(4)?;
    w.key("kind")?;
    w.str("TableProperties")?;
    w.key("name")?;
    w.str(table_name)?;
    w.key("sortOrder")?;
    w.list(0)?;
    w.key("ascending")?;
    w.bool(false)?;
    w.key("columns")?;
    w.list(cols.len())?;
    Ok(())
}

fn write_column_header<W: Write>(w: &mut TsonWriter<W>, c: &ColSpec, n_rows: usize) -> Result<()> {
    w.map(6)?;
    w.key("kind")?;
    w.str("Column")?;
    w.key("name")?;
    w.str(c.name)?;
    w.key("type")?;
    w.str(c.ty)?;
    w.key("nRows")?;
    w.i32(n_rows as i32)?;
    w.key("size")?;
    w.i32(n_rows as i32)?;
    w.key("values")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cluster_column_carries_the_namespace() {
        assert_eq!(cluster_column("ds0"), "ds0.cluster");
    }

    #[test]
    fn labels_render_like_r_paste0() {
        assert_eq!(label(Some(3)), "cluster3");
        assert_eq!(label(Some(12)), "cluster12");
        assert_eq!(label(None), "clusterNA");
    }

    #[test]
    fn the_result_encodes_and_round_trips_through_the_reader() {
        let mut buf = Vec::new();
        let mut w = TsonWriter::new(&mut buf).unwrap();
        write_result(&mut w, "t", &[Some(1), None, Some(2)], "test").unwrap();
        let v = rustson::decode(std::io::Cursor::new(&buf)).unwrap();
        let rustson::Value::MAP(m) = v else {
            panic!("not a map")
        };
        let rustson::Value::LST(tables) = m.get("tables").unwrap() else {
            panic!("no tables")
        };
        assert_eq!(tables.len(), 1);
    }
}
