use burn::prelude::Backend;
use burn::tensor::Tensor;

use crate::backend::AutoBackend;
use crate::config::NanoChatConfig;
use crate::gpt::CausalSelfAttention;
use crate::rope::precompute_rotary_embeddings;
use crate::tauattention::{TauCacheLayer, TauModeAttention};

type B = AutoBackend;

/// Deterministic float generator (so tests are reproducible without relying on backend RNG).
fn make_input(
    b: usize,
    t: usize,
    c: usize,
    seed: u64,
    device: &<B as Backend>::Device,
) -> Tensor<B, 3> {
    let mut s = seed;
    let mut data = Vec::<f32>::with_capacity(b * t * c);
    for _ in 0..(b * t * c) {
        // LCG
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (s >> 32) as u32;
        let x = (u as f32) / (u32::MAX as f32); // [0,1]
        data.push((x * 2.0 - 1.0) * 0.5); // roughly [-0.5, 0.5]
    }
    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([b, t, c])
}

fn max_abs_diff(a: Tensor<B, 3>, b: Tensor<B, 3>) -> f32 {
    let diff = (a - b).abs();
    let vec: Vec<f32> = diff.to_data().to_vec().unwrap();
    vec.into_iter().fold(0.0f32, |m, v| m.max(v))
}

fn base_cfg() -> NanoChatConfig {
    NanoChatConfig {
        sequence_len: 16,
        vocab_size: 64,
        n_layer: 1,
        n_head: 4,
        n_kv_head: 4,
        n_embd: 32,
        block_size: 16,
        dropout: 0.0,
    }
}

fn run_causal_case(cfg: &NanoChatConfig, seed: u64) {
    let device = Default::default();
    let bsz = 2usize;
    let t = 8usize;
    let c = cfg.n_embd;

    let head_dim = cfg.n_embd / cfg.n_head;

    // RoPE cache (forward uses full; decode uses per-step slices).
    let (cos, sin) = precompute_rotary_embeddings(cfg.sequence_len, head_dim, 10000.0, &device);

    let attn = CausalSelfAttention::new(cfg, 0, &device);

    let x = make_input(bsz, t, c, seed, &device);

    // Forward (prefill)
    let y_fwd = attn.forward(x.clone(), (&cos, &sin));

    // Decode (step-by-step), concatenate outputs along time dim=1.
    let mut cache_layer: Option<(Tensor<B, 4>, Tensor<B, 4>)> = None;
    let mut ys: Vec<Tensor<B, 3>> = Vec::with_capacity(t);

    let d2 = cos.dims()[3];
    for pos in 0..t {
        let x_step = x.clone().slice([0..bsz, pos..pos + 1, 0..c]); // [B,1,C]
        let cos_step = cos.clone().slice([0..1, pos..pos + 1, 0..1, 0..d2]); // [1,1,1,D/2]
        let sin_step = sin.clone().slice([0..1, pos..pos + 1, 0..1, 0..d2]); // [1,1,1,D/2]
        let y_step = attn.forward_decode(x_step, (&cos_step, &sin_step), &mut cache_layer); // [B,1,C]
        ys.push(y_step);
    }

    let y_dec = Tensor::cat(ys, 1); // [B,T,C]
    let mad = max_abs_diff(y_fwd, y_dec);
    assert!(
        mad < 1e-6,
        "causal forward/decode mismatch: max_abs_diff={}",
        mad
    );
}

fn run_tau_case(cfg: &NanoChatConfig, seed: u64) {
    let device = Default::default();
    let bsz = 2usize;
    let t = 8usize;
    let c = cfg.n_embd;

    let head_dim = cfg.n_embd / cfg.n_head;

    let (cos, sin) = precompute_rotary_embeddings(cfg.sequence_len, head_dim, 10000.0, &device);

    let attn = TauModeAttention::new(cfg, 0, &device);

    let x = make_input(bsz, t, c, seed, &device);

    // Forward (prefill)
    let y_fwd = attn.forward(x.clone(), (&cos, &sin));

    // Decode (step-by-step)
    let mut cache_layer: TauCacheLayer<B> = None;
    let mut ys: Vec<Tensor<B, 3>> = Vec::with_capacity(t);

    let d2 = cos.dims()[3];
    for pos in 0..t {
        let x_step = x.clone().slice([0..bsz, pos..pos + 1, 0..c]); // [B,1,C]
        let cos_step = cos.clone().slice([0..1, pos..pos + 1, 0..1, 0..d2]); // [1,1,1,D/2]
        let sin_step = sin.clone().slice([0..1, pos..pos + 1, 0..1, 0..d2]); // [1,1,1,D/2]
        let y_step = attn.forward_decode(x_step, (&cos_step, &sin_step), &mut cache_layer); // [B,1,C]
        ys.push(y_step);
    }

    let y_dec = Tensor::cat(ys, 1); // [B,T,C]
    let mad = max_abs_diff(y_fwd, y_dec);
    assert!(
        mad < 1e-6,
        "tau forward/decode mismatch: max_abs_diff={}",
        mad
    );
}

#[test]
fn test_forward_decode_matches_across_mqa_regimes_and_seeds() {
    // Two regimes:
    // 1) No MQA: n_head == n_kv_head
    // 2) MQA:    n_head != n_kv_head (and n_head % n_kv_head == 0)
    let regimes = [
        (4usize, 4usize), // no MQA expansion
        (4usize, 2usize), // MQA expansion
    ];

    // A few deterministic seeds.
    let seeds = [0u64, 1u64, 42u64, 1234u64];

    for (nh, nkh) in regimes {
        let mut cfg = base_cfg();
        cfg.n_head = nh;
        cfg.n_kv_head = nkh;
        cfg.n_embd = 32; // head_dim = 8 in both regimes above

        for &seed in &seeds {
            run_causal_case(&cfg, seed);
            run_tau_case(&cfg, seed);
        }
    }
}
