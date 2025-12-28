// ─────────────────────────────────────────────────────────────────────────────
// RoPE
// ─────────────────────────────────────────────────────────────────────────────

use burn::tensor::{Tensor, backend::Backend};
use log::{debug, info};

pub(crate) fn precompute_rotary_embeddings<B: Backend>(
    seq_len: usize,
    head_dim: usize,
    base: f32,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    info!(
        "Precomputing RoPE: seq_len={}, head_dim={}, base={}",
        seq_len, head_dim, base
    );

    let channel_range: Vec<f32> = (0..head_dim).step_by(2).map(|i| i as f32).collect();
    let half_dim = channel_range.len();
    debug!("RoPE half_dim={}", half_dim);

    let inv_freq: Vec<f32> = channel_range
        .iter()
        .map(|&c| 1.0 / base.powf(c / head_dim as f32))
        .collect();

    let t: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();

    let mut freqs_data = Vec::with_capacity(seq_len * half_dim);
    for &time in &t {
        for &freq in &inv_freq {
            freqs_data.push(time * freq);
        }
    }

    let freqs =
        Tensor::<B, 1>::from_floats(freqs_data.as_slice(), device).reshape([seq_len, half_dim]);

    let cos = freqs.clone().cos();
    let sin = freqs.sin();

    // [1, seq_len, 1, D/2]
    let cos: Tensor<B, 3> = cos.unsqueeze_dim::<3>(0);
    let cos: Tensor<B, 4> = cos.unsqueeze_dim::<4>(2);

    let sin: Tensor<B, 3> = sin.unsqueeze_dim::<3>(0);
    let sin: Tensor<B, 4> = sin.unsqueeze_dim::<4>(2);

    debug!("RoPE cos/sin shape: {:?}", cos.dims());
    (cos, sin)
}

pub(crate) fn apply_rotary_emb<B: Backend>(
    x: Tensor<B, 4>,    // [B, H, T, D]
    cos: &Tensor<B, 4>, // [1, MAX_SEQ, 1, D/2] - full global cache
    sin: &Tensor<B, 4>, // [1, MAX_SEQ, 1, D/2]
) -> Tensor<B, 4> {
    let [b, h, t, d] = x.dims();
    let d_half = d / 2;

    let x1 = x.clone().slice([0..b, 0..h, 0..t, 0..d_half]);
    let x2 = x.slice([0..b, 0..h, 0..t, d_half..d]);

    // Slice RoPE for positions [0..t] from global cache
    let cos = cos
        .clone()
        .slice([0..1, 0..t, 0..1, 0..d_half]) // positions 0..t (absolute)
        .swap_dims(1, 2) // [1, 1, t, d_half]
        .expand([b, h, t, d_half]);

    let sin = sin
        .clone()
        .slice([0..1, 0..t, 0..1, 0..d_half])
        .swap_dims(1, 2)
        .expand([b, h, t, d_half]);

    let y1 = x1.clone() * cos.clone() + x2.clone() * sin.clone();
    let y2 = x2 * cos - x1 * sin;

    Tensor::cat(vec![y1, y2], 3)
}

pub(crate) fn apply_rotary_emb_step<B: Backend>(
    x: Tensor<B, 4>,         // [B, H, 1, D]
    cos_step: &Tensor<B, 4>, // [1, 1, 1, D/2]
    sin_step: &Tensor<B, 4>, // [1, 1, 1, D/2]
) -> Tensor<B, 4> {
    let [b, h, t, d] = x.dims();
    debug_assert_eq!(t, 1, "apply_rotary_emb_step expects T=1");
    let d_half = d / 2;

    let x1 = x.clone().slice([0..b, 0..h, 0..1, 0..d_half]);
    let x2 = x.slice([0..b, 0..h, 0..1, d_half..d]);

    let cos = cos_step.clone().expand([b, h, 1, d_half]);
    let sin = sin_step.clone().expand([b, h, 1, d_half]);

    let y1 = x1.clone() * cos.clone() + x2.clone() * sin.clone();
    let y2 = x2 * cos - x1 * sin;

    Tensor::cat(vec![y1, y2], 3)
}
