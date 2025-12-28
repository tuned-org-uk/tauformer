//! Minimal Parquet storage for sparse matrices (COO triplets).
//!
//! Copied/simplified from ArrowSpace storage, keeping only what's needed
//! for GraphLaplacian persistence.

use std::sync::Arc;

use arrow::array::{Float64Array, RecordBatch, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::{Tensor, backend::Backend};
use serde::Deserialize;
use sprs::{CsMat, TriMat};
use std::{
    fs::File,
    path::{Path, PathBuf},
    str::FromStr,
};

/// Mirror ArrowSpace TauMode options (the ones encoded in manifold.json). [file:27][file:28]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TauMode {
    Fixed, // not used from json in your example, but kept for completeness
    Median,
    Mean,
    Percentile, // percentile value would need an extra field if you decide to encode it
}

impl FromStr for TauMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "Median" => Ok(TauMode::Median),
            "Mean" => Ok(TauMode::Mean),
            // If you later encode Percentile(p) explicitly in JSON, extend this.
            other => anyhow::bail!("Unsupported TauMode in manifold.json: {}", other),
        }
    }
}

/// Only the JSON fields we actually need (matches your nested encoding). [file:28]
#[derive(Debug, Deserialize)]
struct ManifoldMeta {
    builder_config: BuilderConfig,
}

#[derive(Debug, Deserialize)]
struct BuilderConfig {
    synthesis: Synthesis,
    nfeatures: WrappedUsize,
}

#[derive(Debug, Deserialize)]
struct Synthesis {
    #[serde(rename = "TauMode")]
    tau_mode: String,
}

#[derive(Debug, Deserialize)]
struct WrappedUsize {
    #[serde(rename = "Usize")]
    value: usize,
}

pub struct DomainManifold {
    pub laplacian: CsMat<f64>, // CSR Laplacian [file:23]
    pub tau_mode: TauMode,     // Derived from manifold.json [file:28]
    pub nfeatures: usize,      // Should match laplacian rows [file:28]
    pub meta_path: PathBuf,
}

/// Load manifold.parquet (CSR Laplacian) + manifold.json metadata. [file:23][file:28]
pub fn load_domain_manifold(manifold_parquet_path: impl AsRef<Path>) -> Result<DomainManifold> {
    let parquet_path = manifold_parquet_path.as_ref();
    let dir = parquet_path
        .parent()
        .context("manifold parquet path has no parent directory")?;

    let meta_path = dir.join("manifold.json");

    // 1) Load metadata (tau mode + feature dimension). [file:28]
    let meta_file = File::open(&meta_path)
        .with_context(|| format!("Failed to open manifold metadata: {:?}", meta_path))?;
    let meta: ManifoldMeta = serde_json::from_reader(meta_file)
        .with_context(|| format!("Failed to parse manifold metadata JSON: {:?}", meta_path))?;

    let nfeatures = meta.builder_config.nfeatures.value;
    let tau_mode = TauMode::from_str(meta.builder_config.synthesis.tau_mode.as_str())?;

    // 2) Load CSR Laplacian (COO triplets -> CSR). [file:23]
    let laplacian = load_sparse_matrix(parquet_path).with_context(|| {
        format!(
            "Failed to load manifold laplacian parquet: {:?}",
            parquet_path
        )
    })?;

    anyhow::ensure!(
        laplacian.rows() == laplacian.cols(),
        "Manifold Laplacian must be square, got {}x{}",
        laplacian.rows(),
        laplacian.cols()
    );
    anyhow::ensure!(
        laplacian.rows() == nfeatures,
        "Metadata nfeatures={} does not match laplacian dim={} (rows)",
        nfeatures,
        laplacian.rows()
    );

    Ok(DomainManifold {
        laplacian,
        tau_mode,
        nfeatures,
        meta_path,
    })
}

/// Save a sparse matrix to Parquet (COO triplets format).
///
/// Schema:
/// - name_id: Utf8
/// - n_rows: UInt64
/// - n_cols: UInt64
/// - nnz: UInt64
/// - row: UInt64
/// - col: UInt64
/// - value: Float64
pub fn save_sparse_matrix(
    matrix: &CsMat<f64>,
    path: impl AsRef<Path>,
    name_id: &str,
) -> Result<()> {
    let (n_rows, n_cols) = matrix.shape();
    let nnz = matrix.nnz();

    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);

    for (row_idx, row_vec) in matrix.outer_iterator().enumerate() {
        for (col_idx, &value) in row_vec.iter() {
            rows.push(row_idx as u64);
            cols.push(col_idx as u64);
            vals.push(value);
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("name_id", DataType::Utf8, false),
        Field::new("n_rows", DataType::UInt64, false),
        Field::new("n_cols", DataType::UInt64, false),
        Field::new("nnz", DataType::UInt64, false),
        Field::new("row", DataType::UInt64, false),
        Field::new("col", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
    ]));

    let name_array = StringArray::from(vec![name_id; nnz]);
    let n_rows_array = UInt64Array::from(vec![n_rows as u64; nnz]);
    let n_cols_array = UInt64Array::from(vec![n_cols as u64; nnz]);
    let nnz_array = UInt64Array::from(vec![nnz as u64; nnz]);
    let row_array = UInt64Array::from(rows);
    let col_array = UInt64Array::from(cols);
    let val_array = Float64Array::from(vals);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(name_array),
            Arc::new(n_rows_array),
            Arc::new(n_cols_array),
            Arc::new(nnz_array),
            Arc::new(row_array),
            Arc::new(col_array),
            Arc::new(val_array),
        ],
    )
    .context("Failed to create RecordBatch")?;

    let file_path = path.as_ref().join(format!("{}.parquet", name_id));
    let file = File::create(&file_path)
        .with_context(|| format!("Failed to create parquet file: {:?}", file_path))?;

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    let mut writer =
        ArrowWriter::try_new(file, schema, Some(props)).context("Failed to create ArrowWriter")?;

    writer.write(&batch).context("Failed to write batch")?;
    writer.close().context("Failed to close writer")?;

    Ok(())
}

/// Load a sparse matrix from Parquet (COO triplets → CSR).
///
/// Reads schema:
/// - n_rows, n_cols, nnz
/// - row, col, value arrays
pub fn load_sparse_matrix(path: impl AsRef<Path>) -> Result<CsMat<f64>> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open parquet file: {:?}", path.as_ref()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to create parquet reader")?;

    let mut reader = builder.build().context("Failed to build parquet reader")?;

    let batch = reader
        .next()
        .ok_or_else(|| anyhow::anyhow!("No data in parquet file"))?
        .context("Failed to read parquet batch")?;

    // Extract dimensions
    let n_rows = get_u64_scalar(&batch, "n_rows")? as usize;
    let n_cols = get_u64_scalar(&batch, "n_cols")? as usize;

    // Extract triplets
    let row_col = batch
        .column_by_name("row")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .context("row column missing or wrong type")?;

    let col_col = batch
        .column_by_name("col")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .context("col column missing or wrong type")?;

    let val_col = batch
        .column_by_name("value")
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
        .context("value column missing or wrong type")?;

    // Build sparse matrix from triplets
    let mut trimat = TriMat::new((n_rows, n_cols));
    for i in 0..row_col.len() {
        trimat.add_triplet(
            row_col.value(i) as usize,
            col_col.value(i) as usize,
            val_col.value(i),
        );
    }

    Ok(trimat.to_csr())
}

fn get_u64_scalar(batch: &RecordBatch, col_name: &str) -> Result<u64> {
    let col = batch
        .column_by_name(col_name)
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .with_context(|| format!("{} column missing or wrong type", col_name))?;
    anyhow::ensure!(col.len() > 0, "{} column is empty", col_name);
    Ok(col.value(0))
}

pub fn load_manifold_laplacian_for_head_dim<B: Backend>(
    parquet_path: impl AsRef<Path>,
    head_dim: usize,
    _device: &B::Device,
) -> anyhow::Result<CsMat<f64>> {
    // Loads both manifold.parquet (CSR Laplacian) and manifold.json metadata. [file:44]
    let domain = load_domain_manifold(parquet_path.as_ref())?;

    anyhow::ensure!(head_dim > 0, "head_dim must be > 0");
    anyhow::ensure!(
        domain.nfeatures == head_dim,
        "manifold nfeatures={} must match head_dim={}",
        domain.nfeatures,
        head_dim
    );

    let lap = domain.laplacian;
    anyhow::ensure!(
        lap.rows() == head_dim && lap.cols() == head_dim,
        "laplacian shape mismatch: got {}x{}, expected {}x{}",
        lap.rows(),
        lap.cols(),
        head_dim,
        head_dim
    );

    Ok(lap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sparse_roundtrip() {
        let _ = std::fs::create_dir_all("./test_data");

        let mut tri = TriMat::new((4, 4));
        tri.add_triplet(0, 0, 2.0);
        tri.add_triplet(0, 1, -1.0);
        tri.add_triplet(1, 1, 3.0);
        tri.add_triplet(2, 2, 1.5);
        let original = tri.to_csr();

        save_sparse_matrix(&original, "./test_data", "test_sparse").unwrap();
        let loaded = load_sparse_matrix("./test_data/test_sparse.parquet").unwrap();

        assert_eq!(original.shape(), loaded.shape());
        assert_eq!(original.nnz(), loaded.nnz());

        for i in 0..4 {
            for j in 0..4 {
                let a = original.get(i, j).copied().unwrap_or(0.0);
                let b = loaded.get(i, j).copied().unwrap_or(0.0);
                assert_relative_eq!(a, b, epsilon = 1e-10);
            }
        }
    }
}
