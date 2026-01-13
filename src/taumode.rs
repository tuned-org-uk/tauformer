//! Taumode utilities for Tauformer.
//!
//! First feasibility version:
//! - Use a simple dense feature-space Laplacian L in D=head_dim.
//! - Compute lambda per token/head from vectors using a bounded Rayleigh quotient.
//! - Convert |Δlambda| into attention logits and reuse the existing stable softmax pipeline.

use burn::module::Param;
use burn::tensor::{Bool, Tensor, activation, backend::Backend};
use rayon::prelude::*;
use sprs::CsMat;

/// Config for taumode lambda + distance attention.
#[derive(Clone, Copy, Debug)]
pub struct TauModeConfig {
    /// E_bounded = E_raw / (E_raw + tau)
    pub tau: f32,
    /// Numerical eps for denominators.
    pub eps: f32,
    /// Temperature for turning distances into logits: att = -|Δλ| / temperature
    pub temperature: f32,
}

impl Default for TauModeConfig {
    fn default() -> Self {
        Self {
            tau: 1.0,
            eps: 1.0e-6,
            temperature: 1.0,
        }
    }
}

/// Dense feature Laplacian in feature space (D x D).
#[derive(Clone, Debug)]
pub struct FeatureLaplacian<B: Backend> {
    pub matrix: Tensor<B, 2>, // [D, D]
}

impl<B: Backend> FeatureLaplacian<B> {
    pub fn dim(&self) -> usize {
        let [r, c] = self.matrix.dims();
        debug_assert_eq!(r, c, "Feature Laplacian must be square");
        r
    }
}

/// Build a simple 1D chain Laplacian (Dirichlet-ish ends) as dense [D,D].
///
/// This is a reasonable "roughness" operator for an initial test because it penalizes
/// adjacent-channel differences (like a discrete second derivative).
pub fn laplacian_chain_dense<B: Backend>(d: usize, device: &B::Device) -> FeatureLaplacian<B> {
    let mut data = vec![0.0f32; d * d];

    // L[i,i] = degree; L[i,i±1] = -1
    for i in 0..d {
        let mut deg = 0.0;
        if i > 0 {
            data[i * d + (i - 1)] = -1.0;
            deg += 1.0;
        }
        if i + 1 < d {
            data[i * d + (i + 1)] = -1.0;
            deg += 1.0;
        }
        data[i * d + i] = deg;
    }

    FeatureLaplacian {
        matrix: Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([d, d]),
    }
}

/// (Dense pathway) Compute taumode lambda per token per head from head vectors.
///
/// Input: x [B, H, T, D]
/// Output: lambda [B, H, T]
///
/// E_raw = (x^T L x) / (x^T x + eps)
/// E_bounded = E_raw / (E_raw + tau)
/// lambda = E_bounded   (first version; later you can blend in dispersion as in arrowspace)
pub fn lambdas_from_heads<B: Backend>(
    x: Tensor<B, 4>,
    lap: Param<Tensor<B, 2>>,
    cfg: &TauModeConfig,
) -> Tensor<B, 3> {
    let [b, h, t, d] = x.dims();
    let [lr, lc] = lap.val().dims();

    debug_assert_eq!(lr, d, "Laplacian rows must match head dim D");
    debug_assert_eq!(lc, d, "Laplacian cols must match head dim D");

    // Flatten [B,H,T,D] -> [N,D]
    let n = b * h * t;
    let x_nd = x.reshape([n, d]);

    // y = x L  -> [N,D]
    let y_nd = x_nd.clone().matmul(lap.val());

    // numerator = sum_i x_i * (xL)_i
    let numerator = (x_nd.clone() * y_nd).sum_dim(1); // [N]

    // denominator = sum_i x_i^2 + eps
    let denom = x_nd.powf_scalar(2.0).sum_dim(1) + cfg.eps; // [N]

    let e_raw = numerator / denom; // [N]
    let e_bounded = e_raw.clone() / (e_raw + cfg.tau); // [N]

    e_bounded.reshape([b, h, t])
}

/// Expand KV heads to match query heads (MQA), for a [B,Hkv,T,*] tensor.
///
/// This mirrors the existing attention MQA expansion logic in your gpt.rs.
pub fn mqa_expand_heads_3<B: Backend>(
    x: Tensor<B, 3>, // [B, Hkv, T]
    n_head: usize,
    n_kv_head: usize,
) -> Tensor<B, 3> {
    if n_head == n_kv_head {
        return x;
    }
    let [b, hkv, t] = x.dims();
    debug_assert_eq!(hkv, n_kv_head);

    let repeat = n_head / n_kv_head;
    let x4: Tensor<B, 4> = x.unsqueeze_dim::<4>(2); // [B,Hkv,1,T]
    x4.expand([b, hkv, repeat, t]).reshape([b, n_head, t])
}

/// Expand KV heads to match query heads (MQA), for a [B,Hkv,T,D] tensor.
///
/// This mirrors the existing attention MQA expansion logic in your gpt.rs.
pub fn mqa_expand_heads_4<B: Backend>(
    x: Tensor<B, 4>, // [B, Hkv, T, D]
    n_head: usize,
    n_kv_head: usize,
) -> Tensor<B, 4> {
    if n_head == n_kv_head {
        return x;
    }
    let [b, hkv, t, d] = x.dims();
    debug_assert_eq!(hkv, n_kv_head);

    let repeat = n_head / n_kv_head;
    let x5: Tensor<B, 5> = x.unsqueeze_dim::<5>(2); // [B,Hkv,1,T,D]
    x5.expand([b, hkv, repeat, t, d]).reshape([b, n_head, t, d])
}

/// Build attention logits from lambdas.
///
/// lambda_q: [B,H,Tq]
/// lambda_k: [B,H,Tk]
/// returns:  att [B,H,Tq,Tk] where att = -|Δλ| / temperature
pub fn taumode_distance_logits<B: Backend>(
    lambda_q: Tensor<B, 3>,
    lambda_k: Tensor<B, 3>,
    cfg: &TauModeConfig,
) -> Tensor<B, 4> {
    let [bq, hq, _tq] = lambda_q.dims();
    let [bk, hk, _tk] = lambda_k.dims();
    debug_assert_eq!(bq, bk);
    debug_assert_eq!(hq, hk);

    let lq = lambda_q.unsqueeze_dim::<4>(3); // [B,H,Tq,1]
    let lk = lambda_k.unsqueeze_dim::<4>(2); // [B,H,1,Tk]

    let temp = cfg.temperature.max(cfg.eps);
    -((lq - lk).abs() / temp)
}

/// Apply stable causal softmax over keys axis (dim=3).
///
/// This is the same stability pattern you already use:
/// mask-fill with -1e9, row-max subtraction, softmax(dim=3).
pub fn causal_softmax_over_keys<B: Backend>(
    mut att: Tensor<B, 4>, // [B,H,Tq,Tk]
    t_q: usize,
    t_k: usize,
    n_head: usize,
) -> Tensor<B, 4> {
    let [b, h, tq, tk] = att.dims();
    debug_assert_eq!(h, n_head);

    // 1) causal mask (lower-triangular)
    let diag = (tk as i64) - (tq as i64);
    let mask2: Tensor<B, 2, Bool> = Tensor::tril_mask([tq, tk], diag, &att.device());

    let mask4 = mask2
        .unsqueeze_dims::<4>(&[0, 1])
        .expand([b, n_head, t_q, t_k]);
    att = att.mask_fill(mask4, -1.0e9); // Fill upper-tri (where mask=true) with -1e9

    // 2) subtract row max (dim=3)
    let att_max = att.clone().max_dim(3).flatten::<3>(2, 3); // [B,H,Tq]
    att = att - att_max.unsqueeze_dim::<4>(3);

    // 3) softmax over keys
    activation::softmax(att, 3)
}

use crate::pretraining::parquet::TauMode as ManifoldTauMode;

const TAU_FLOOR: f32 = 1e-10;

fn select_tau_from_vector(x: &[f32], mode: ManifoldTauMode) -> f32 {
    let mut v: Vec<f32> = x
        .iter()
        .copied()
        .map(|a| a)
        .filter(|a| a.is_finite())
        .collect();
    if v.is_empty() {
        return TAU_FLOOR;
    }

    match mode {
        ManifoldTauMode::Median => {
            v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = if v.len() % 2 == 1 {
                v[v.len() / 2]
            } else {
                0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
            };
            mid.max(TAU_FLOOR)
        }
        ManifoldTauMode::Mean => {
            let sum: f32 = v.iter().sum();
            (sum / (v.len() as f32)).max(TAU_FLOOR)
        }
        // If you later support these in your JSON/enum, extend here.
        _ => TAU_FLOOR,
    }
}

// Bounded Rayleigh quotient using CSR: E_raw = (x^T L x) / (x^T x + eps), then E/(E+tau).
fn taumode_bounded_csr(x: &[f32], l: &CsMat<f32>, tau: f32, eps: f32) -> f32 {
    debug_assert_eq!(l.rows(), l.cols());
    debug_assert_eq!(l.rows(), x.len());

    // numerator = x^T L x
    let mut numerator = 0.0f32;
    for (i, row) in l.outer_iterator().enumerate() {
        let xi = x[i];
        let mut row_dot = 0.0f32;
        for (j, lij) in row.iter() {
            row_dot += lij * (x[j]);
        }
        numerator += xi * row_dot;
    }

    // denom = x^T x + eps
    let mut denom = eps;
    for &xi in x {
        let v = xi;
        denom += v * v;
    }

    let e_raw = if denom > eps {
        (numerator / denom).max(0.0)
    } else {
        0.0
    };
    let tau = tau.max(eps);
    (e_raw / (e_raw + tau)) as f32
}

/// CPU-parallel sparse lambdas from heads.
/// heads: [B,H,T,D] -> [B,H,T]
pub fn lambdas_from_heads_sparse<B: Backend>(
    heads: Tensor<B, 4>,
    laplacian: &CsMat<f32>,
    tau_mode: ManifoldTauMode,
    eps: f32,
) -> Tensor<B, 3> {
    let [b, h, t, d] = heads.dims();
    assert_eq!(laplacian.rows(), d);
    assert_eq!(laplacian.cols(), d);

    // Download once; compute lambdas on CPU in parallel; upload result back to device.
    let host: Vec<f32> = heads.to_data().to_vec().expect("heads must be f32");
    let n = b * h * t;

    let eps64 = eps.max(1e-12);

    let mut out = vec![0.0f32; n];
    out.par_iter_mut().enumerate().for_each(|(idx, dst)| {
        let start = idx * d;
        let x = &host[start..start + d];
        let tau = select_tau_from_vector(x, tau_mode);
        *dst = taumode_bounded_csr(x, laplacian, tau, eps64);
    });

    Tensor::<B, 1>::from_floats(out.as_slice(), &heads.device()).reshape([b, h, t])
}
