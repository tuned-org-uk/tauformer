//! Domain manifold (GraphLaplacian) management for Tauformer.
//!
//! The manifold is "acquired knowledge": trained offline on domain embeddings,
//! persisted via parquet, and loaded at runtime for taumode(Q/K).

use anyhow::{Context, Result};
use std::path::Path;

use sprs::CsMat;

use crate::parquet;

/// Only persistent artifact needed at tauformer runtime.
#[derive(Debug, Clone)]
pub struct DomainManifold {
    pub matrix: CsMat<f64>,
}

impl DomainManifold {
    pub fn laplacian(&self) -> &CsMat<f64> {
        &self.matrix
    }

    pub fn dim(&self) -> usize {
        self.matrix.rows()
    }
}

/// Save the domain manifold Laplacian to a parquet file.
///
/// `name_id` is the base filename (e.g., "manifold" → "manifold.parquet").
pub fn save_domain_manifold(
    manifold: &DomainManifold,
    dir: impl AsRef<Path>,
    name_id: &str,
) -> Result<()> {
    std::fs::create_dir_all(dir.as_ref()).context("Failed to create manifold directory")?;

    parquet::save_sparse_matrix(&manifold.matrix, dir.as_ref(), name_id)
        .context("Failed to save GraphLaplacian to parquet")
}

/// Load the domain manifold Laplacian from a parquet file.
///
/// `path` is the full path including .parquet extension
/// (e.g., "./pretraining/manifold.parquet").
pub fn load_domain_manifold(path: impl AsRef<Path>) -> Result<DomainManifold> {
    let lap: CsMat<f64> = parquet::load_sparse_matrix(&path)
        .with_context(|| format!("Failed to load laplacian parquet at {:?}", path.as_ref()))?;

    let (r, c) = lap.shape();
    anyhow::ensure!(
        r == c,
        "Domain Laplacian must be square (feature-space F×F), got {}×{}",
        r,
        c
    );

    Ok(DomainManifold { matrix: lap })
}

/// Minimal taumode runtime config.
#[derive(Clone, Copy, Debug)]
pub struct TauModeRuntime {
    pub tau: f64,
    pub eps: f64,
}

impl Default for TauModeRuntime {
    fn default() -> Self {
        Self {
            tau: 1.0,
            eps: 1e-12,
        }
    }
}

/// Compute taumode(x, gl) for a single vector x.
///
/// Bounded Rayleigh quotient:
/// E_raw = (x^T L x) / (x^T x + eps)
/// E_bounded = E_raw / (E_raw + tau)
pub fn compute_taumode(x: &[f64], gl: &CsMat<f64>, cfg: TauModeRuntime) -> f64 {
    let l: &CsMat<f64> = &gl;

    debug_assert_eq!(l.rows(), l.cols());
    debug_assert_eq!(l.rows(), x.len());

    // numerator = x^T L x
    let mut numerator = 0.0f64;
    for (i, row) in l.outer_iterator().enumerate() {
        let xi = x[i];
        let mut row_dot = 0.0f64;
        for (j, lij) in row.iter() {
            row_dot += lij * x[j];
        }
        numerator += xi * row_dot;
    }

    // denom = x^T x + eps
    let mut denom = cfg.eps;
    for &xi in x {
        denom += xi * xi;
    }

    let e_raw = if denom > cfg.eps {
        (numerator / denom).max(0.0)
    } else {
        0.0
    };
    let tau = cfg.tau.max(cfg.eps);
    e_raw / (e_raw + tau)
}

pub fn ensure_domain_manifold_exists(path: impl AsRef<Path>) -> anyhow::Result<()> {
    // if !path.exists() { build + write parquet }
    todo!()
}

/// Convenience conversion (Burn tensors often yield f32).
pub fn vec_f32_to_f64(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| v as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;

    #[test]
    fn test_compute_taumode_identity() {
        let mut tri = TriMat::<f64>::new((3, 3));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(1, 1, 1.0);
        tri.add_triplet(2, 2, 1.0);
        let gl = tri.to_csr();

        let x = vec![1.0, 2.0, 3.0];
        let cfg = TauModeRuntime {
            tau: 1.0,
            eps: 1e-12,
        };

        let v = compute_taumode(&x, &gl, cfg);
        // e_raw=1 => e_bounded=0.5
        assert!((v - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_save_load_manifold() {
        let _ = std::fs::create_dir_all("./test_data");

        let mut tri = TriMat::<f64>::new((2, 2));
        tri.add_triplet(0, 0, 2.0);
        tri.add_triplet(0, 1, -1.0);
        tri.add_triplet(1, 0, -1.0);
        tri.add_triplet(1, 1, 2.0);

        let original = DomainManifold {
            matrix: tri.to_csr(),
        };

        save_domain_manifold(&original, "./test_data", "test_manifold").unwrap();
        let loaded = load_domain_manifold("./test_data/test_manifold.parquet").unwrap();

        assert_eq!(original.matrix.shape(), loaded.matrix.shape());
        assert_eq!(original.matrix.nnz(), loaded.matrix.nnz());
    }
}
