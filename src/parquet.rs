//! Minimal Parquet storage for sparse matrices (COO triplets).
//!
//! Copied/simplified from ArrowSpace storage, keeping only what's needed
//! for GraphLaplacian persistence.

use anyhow::{Context, Result};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Array, RecordBatch, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use burn::prelude::*;
use sprs::{CsMat, TriMat};

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
    parquet_path: impl AsRef<std::path::Path>,
    head_dim: usize,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 2>> {
    // Reuse your existing parquet reading utilities here.
    // Must return [head_dim, head_dim] on `device`.
    todo!()
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
