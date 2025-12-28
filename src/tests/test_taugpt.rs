// tests/test_taugpt.rs
//
// Tests for TauGptModel (taumode attention GPT) using the sparse constructor only.
//
// Covers:
// - Model construction
// - Forward output shape + finiteness
// - Greedy generation determinism
// - Decode path matches prefill forward logits (position-by-position)
// - Run in both regimes: (n_head == n_kv_head) and (n_head != n_kv_head)

use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};

use crate::backend::AutoBackend;
use crate::config::NanoChatConfig;
use crate::taugpt::{TauGptModel, TauKVCache};

use sprs::TriMat;
use std::sync::Arc;

type B = AutoBackend;

fn tiny_cfg(n_head: usize, n_kv_head: usize) -> NanoChatConfig {
    NanoChatConfig {
        sequence_len: 16,
        vocab_size: 64,
        n_layer: 2,
        n_head,
        n_kv_head,
        n_embd: 32, // head_dim = 8 when n_head=4
        block_size: 16,
        dropout: 0.0,
    }
}

fn max_abs_diff(a: Tensor<B, 3>, b: Tensor<B, 3>) -> f32 {
    let diff = (a - b).abs();
    let v: Vec<f32> = diff.to_data().to_vec().unwrap();
    v.into_iter().fold(0.0f32, |m, x| m.max(x))
}

fn build_identity_laplacian_csr(head_dim: usize) -> sprs::CsMat<f64> {
    // Minimal valid Laplacian-like matrix (PSD, square, simple).
    // This is just to satisfy the constructor and keep tests deterministic.
    let mut tri = TriMat::<f64>::new((head_dim, head_dim));
    for i in 0..head_dim {
        tri.add_triplet(i, i, 1.0);
    }
    tri.to_csr()
}

fn build_model(cfg: &NanoChatConfig, device: &<B as Backend>::Device) -> TauGptModel<B> {
    let head_dim = cfg.n_embd / cfg.n_head;
    let lap = Arc::new(build_identity_laplacian_csr(head_dim));
    let tau_mode = crate::pretraining::parquet::TauMode::Median;

    TauGptModel::<B>::new_with_sparse_laplacian(cfg, device, lap, tau_mode)
}

fn assert_logits_finite(logits: &Tensor<B, 3>) {
    let v: Vec<f32> = logits.clone().to_data().to_vec().unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "Found NaN/Inf in logits");
}

fn run_forward_decode_equivalence(cfg: NanoChatConfig) {
    let device = <B as Backend>::Device::default();
    let model = build_model(&cfg, &device);

    let bsz = 2usize;
    let t = 8usize;

    // Deterministic token pattern (no RNG dependency).
    let ids: Vec<i64> = (0..(bsz * t))
        .map(|i| (i as i64) % (cfg.vocab_size as i64))
        .collect();

    let idx = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device).reshape([bsz, t]);

    // Prefill forward logits: [B,T,V]
    let logits_fwd = model.forward(idx.clone(), false);
    let v = logits_fwd.dims()[2];
    assert_eq!(logits_fwd.dims(), [bsz, t, cfg.vocab_size]);
    assert_logits_finite(&logits_fwd);

    // Decode the same sequence token-by-token and compare logits at each position.
    let mut cache = TauKVCache::<B>::new(model.num_layers());
    cache.reset();

    for pos in 0..t {
        let last = idx.clone().slice([0..bsz, pos..pos + 1]); // [B,1]
        let logits_step = model.forward_decode(last, &mut cache, false); // [B,1,V]
        assert_eq!(logits_step.dims(), [bsz, 1, v]);
        assert_logits_finite(&logits_step);

        let logits_fwd_pos = logits_fwd.clone().slice([0..bsz, pos..pos + 1, 0..v]); // [B,1,V]

        let mad = max_abs_diff(logits_step, logits_fwd_pos);
        assert!(
            mad < 1e-6,
            "forward/decode logits mismatch at pos={}, max_abs_diff={}",
            pos,
            mad
        );
    }
}

#[test]
fn test_taugpt_construction_sparse() {
    let cfg = tiny_cfg(4, 2);
    let device = <B as Backend>::Device::default();
    let _model = build_model(&cfg, &device);
}

#[test]
fn test_taugpt_forward_shape_and_finite_sparse() {
    let cfg = tiny_cfg(4, 2);
    let device = <B as Backend>::Device::default();
    let model = build_model(&cfg, &device);

    let bsz = 3usize;
    let t = 6usize;

    let ids: Vec<i64> = (0..(bsz * t))
        .map(|i| (i as i64) % (cfg.vocab_size as i64))
        .collect();
    let idx = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device).reshape([bsz, t]);

    let logits = model.forward(idx, true);
    assert_eq!(logits.dims(), [bsz, t, cfg.vocab_size]);
    assert_logits_finite(&logits);
}

#[test]
fn test_taugpt_generation_determinism_sparse() {
    let cfg = tiny_cfg(4, 2);
    let device = <B as Backend>::Device::default();
    let model = build_model(&cfg, &device);

    let prompt: Vec<i64> = vec![1, 2, 3, 4];
    let idx = Tensor::<B, 1, Int>::from_ints(prompt.as_slice(), &device).reshape([1, prompt.len()]);

    // TauGptModel exposes `generate(...)` in your current implementation. [file:46]
    let out1 = model.generate(idx.clone(), 8);
    let out2 = model.generate(idx, 8);

    let v1: Vec<i64> = out1.to_data().to_vec().unwrap();
    let v2: Vec<i64> = out2.to_data().to_vec().unwrap();
    assert_eq!(v1, v2, "Greedy generation should be deterministic");
}

#[test]
fn test_taugpt_forward_decode_matches_no_mqa_sparse() {
    // Regime 1: n_head == n_kv_head (no MQA expansion).
    run_forward_decode_equivalence(tiny_cfg(4, 4));
}

#[test]
fn test_taugpt_forward_decode_matches_mqa_sparse() {
    // Regime 2: n_head != n_kv_head (MQA expansion).
    run_forward_decode_equivalence(tiny_cfg(4, 2));
}
