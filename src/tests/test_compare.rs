//! Tests that compare caching behavior and forward-vs-decode equivalence
//! between standard causal attention and tau attention.

use burn::tensor::Tensor;
use log::{debug, info};

use crate::gpt::CausalSelfAttention;
use crate::tauattention::{TauCacheLayer, TauModeAttention};
use crate::{backend::AutoBackend, config::NanoChatConfig};

type TestBackend = AutoBackend;

fn tiny_cfg() -> NanoChatConfig {
    NanoChatConfig {
        sequence_len: 32,
        vocab_size: 64,
        n_layer: 1,
        n_head: 4,
        n_kv_head: 2,
        n_embd: 32,
        block_size: 32,
        dropout: 0.0,
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn make_identity_rope<B: burn::prelude::Backend>(
    t: usize,
    head_dim: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    // apply_rotary_emb becomes identity when cos=1 and sin=0
    let cos = Tensor::<B, 4>::ones([1, t, 1, head_dim / 2], device);
    let sin = Tensor::<B, 4>::zeros([1, t, 1, head_dim / 2], device);
    (cos, sin)
}

fn rope_step<B: burn::prelude::Backend>(
    cos_full: &Tensor<B, 4>,
    sin_full: &Tensor<B, 4>,
    pos: usize,
    head_dim: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    // slice [1,1,1,D/2] at absolute position `pos`
    let cos_step = cos_full
        .clone()
        .slice([0..1, pos..(pos + 1), 0..1, 0..(head_dim / 2)]);
    let sin_step = sin_full
        .clone()
        .slice([0..1, pos..(pos + 1), 0..1, 0..(head_dim / 2)]);
    (cos_step, sin_step)
}

#[test]
fn test_causal_attention_decode_matches_forward_identity_rope() {
    crate::init();
    info!("══ test_causal_attention_decode_matches_forward_identity_rope ══");

    let cfg = tiny_cfg();
    let device = Default::default();

    let (b, t, c) = (2, 8, cfg.n_embd);
    let head_dim = cfg.n_embd / cfg.n_head;
    let (cos, sin) = make_identity_rope::<TestBackend>(t, head_dim, &device);

    // random input
    let x: Tensor<TestBackend, 3> = Tensor::random(
        [b, t, c],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );

    let attn = CausalSelfAttention::<TestBackend>::new(&cfg, 0, &device);

    // Forward (full sequence)
    let y_fwd = attn.forward(x.clone(), (&cos, &sin)); // [B,T,C]

    // Decode (step-by-step), then concatenate outputs over time
    let mut cache: Option<(Tensor<TestBackend, 4>, Tensor<TestBackend, 4>)> = None;
    let mut ys: Vec<Tensor<TestBackend, 3>> = Vec::with_capacity(t);

    for pos in 0..t {
        let x_step = x.clone().slice([0..b, pos..(pos + 1), 0..c]); // [B,1,C]
        let (cos_step, sin_step) = rope_step::<TestBackend>(&cos, &sin, pos, head_dim);

        let y_step = attn.forward_decode(x_step, (&cos_step, &sin_step), &mut cache); // [B,1,C]
        ys.push(y_step);

        let (k, v) = cache.as_ref().unwrap();
        debug!("pos={}: K={:?} V={:?}", pos, k.dims(), v.dims());
        assert_eq!(k.dims()[2], pos + 1);
        assert_eq!(v.dims()[2], pos + 1);
    }

    let y_dec = Tensor::cat(ys, 1); // [B,T,C]

    // Compare
    let a: Vec<f32> = y_fwd.to_data().to_vec().unwrap();
    let bvec: Vec<f32> = y_dec.to_data().to_vec().unwrap();
    let mad = max_abs_diff(&a, &bvec);

    info!("Causal attn: max_abs_diff(fwd,decode)={:.8}", mad);
    assert!(
        mad < 1e-4,
        "Forward and decode should match (identity RoPE)."
    );
}

#[test]
fn test_tau_attention_decode_matches_forward_identity_rope() {
    crate::init();
    info!("══ test_tau_attention_decode_matches_forward_identity_rope ══");

    let cfg = tiny_cfg();
    let device = Default::default();

    let (b, t, c) = (2, 8, cfg.n_embd);
    let head_dim = cfg.n_embd / cfg.n_head;
    let (cos, sin) = make_identity_rope::<TestBackend>(t, head_dim, &device);

    let x: Tensor<TestBackend, 3> = Tensor::random(
        [b, t, c],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );

    let attn = TauModeAttention::<TestBackend>::new(&cfg, 0, &device);

    let y_fwd = attn.forward(x.clone(), (&cos, &sin)); // [B,T,C]

    let mut cache: TauCacheLayer<TestBackend> = None;
    let mut ys: Vec<Tensor<TestBackend, 3>> = Vec::with_capacity(t);

    for pos in 0..t {
        let x_step = x.clone().slice([0..b, pos..(pos + 1), 0..c]);
        let (cos_step, sin_step) = rope_step::<TestBackend>(&cos, &sin, pos, head_dim);

        let y_step = attn.forward_decode(x_step, (&cos_step, &sin_step), &mut cache);
        ys.push(y_step);

        let (v, lambda_k) = cache.as_ref().unwrap();
        debug!(
            "pos={}: V={:?} lambda_k={:?}",
            pos,
            v.dims(),
            lambda_k.dims()
        );
        assert_eq!(v.dims()[2], pos + 1);
        assert_eq!(lambda_k.dims()[2], pos + 1);
    }

    let y_dec = Tensor::cat(ys, 1);

    let a: Vec<f32> = y_fwd.to_data().to_vec().unwrap();
    let bvec: Vec<f32> = y_dec.to_data().to_vec().unwrap();
    let mad = max_abs_diff(&a, &bvec);

    info!("Tau attn: max_abs_diff(fwd,decode)={:.8}", mad);
    assert!(
        mad < 1e-4,
        "Forward and decode should match (identity RoPE)."
    );
}

#[test]
fn test_tau_vs_causal_outputs_differ_expected() {
    crate::init();
    info!("══ test_tau_vs_causal_outputs_differ_expected ══");

    let cfg = tiny_cfg();
    let device = Default::default();
    let (b, t, c) = (1, 8, cfg.n_embd);

    let head_dim = cfg.n_embd / cfg.n_head;
    let (cos, sin) = make_identity_rope::<TestBackend>(t, head_dim, &device);

    let x: Tensor<TestBackend, 3> = Tensor::random(
        [b, t, c],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );

    // Same input, same RoPE, different scoring => outputs should not be identical in general.
    let causal = CausalSelfAttention::<TestBackend>::new(&cfg, 0, &device);
    let tau = TauModeAttention::<TestBackend>::new(&cfg, 0, &device);

    let y_causal = causal.forward(x.clone(), (&cos, &sin));
    let y_tau = tau.forward(x, (&cos, &sin));

    let a: Vec<f32> = y_causal.to_data().to_vec().unwrap();
    let bvec: Vec<f32> = y_tau.to_data().to_vec().unwrap();
    let mad = max_abs_diff(&a, &bvec);

    info!("max_abs_diff(causal,tau)={:.8}", mad);
    assert!(
        mad > 1e-7,
        "They should differ unless in a degenerate special case."
    );
}

#[test]
fn test_cache_memory_elements_saved() {
    crate::init();
    info!("══ test_cache_memory_elements_saved ══");

    let cfg = tiny_cfg();
    let device = Default::default();
    let (b, c) = (2, cfg.n_embd);

    let head_dim = cfg.n_embd / cfg.n_head;
    let cos: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 1, head_dim / 2], &device);
    let sin: Tensor<TestBackend, 4> = Tensor::zeros([1, 1, 1, head_dim / 2], &device);

    let steps = 8;

    // Build caches by decoding `steps` times.
    let causal = CausalSelfAttention::<TestBackend>::new(&cfg, 0, &device);
    let tau = TauModeAttention::<TestBackend>::new(&cfg, 0, &device);

    let mut causal_cache: Option<(Tensor<TestBackend, 4>, Tensor<TestBackend, 4>)> = None;
    let mut tau_cache: TauCacheLayer<TestBackend> = None;

    for _ in 0..steps {
        let x_step: Tensor<TestBackend, 3> = Tensor::random(
            [b, 1, c],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        causal.forward_decode(x_step.clone(), (&cos, &sin), &mut causal_cache);
        tau.forward_decode(x_step, (&cos, &sin), &mut tau_cache);
    }

    let (k, v) = causal_cache.unwrap();
    let (v_tau, lambda_k) = tau_cache.unwrap();

    let causal_elems = k.dims().iter().product::<usize>() + v.dims().iter().product::<usize>();
    let tau_elems =
        v_tau.dims().iter().product::<usize>() + lambda_k.dims().iter().product::<usize>();

    info!(
        "Causal cache: K={:?}, V={:?}, elems={}",
        k.dims(),
        v.dims(),
        causal_elems
    );
    info!(
        "Tau cache:    V={:?}, λk={:?}, elems={}",
        v_tau.dims(),
        lambda_k.dims(),
        tau_elems
    );

    assert!(
        tau_elems < causal_elems,
        "Tau cache should store fewer elements than causal KV."
    );
}

/// Optional: timing is noisy in unit tests; keep as #[ignore] and run with -- --nocapture.
#[test]
#[ignore]
fn bench_decode_step_latency_logged() {
    crate::init();
    info!("══ bench_decode_step_latency_logged (ignored) ══");

    use std::time::Instant;

    let cfg = tiny_cfg();
    let device = Default::default();
    let (b, c) = (2, cfg.n_embd);

    let head_dim = cfg.n_embd / cfg.n_head;
    let cos: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 1, head_dim / 2], &device);
    let sin: Tensor<TestBackend, 4> = Tensor::zeros([1, 1, 1, head_dim / 2], &device);

    let causal = CausalSelfAttention::<TestBackend>::new(&cfg, 0, &device);
    let tau = TauModeAttention::<TestBackend>::new(&cfg, 0, &device);

    let iters = 200;

    // causal
    let mut causal_cache: Option<(Tensor<TestBackend, 4>, Tensor<TestBackend, 4>)> = None;
    let start = Instant::now();
    for _ in 0..iters {
        let x_step: Tensor<TestBackend, 3> = Tensor::random(
            [b, 1, c],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        let _ = causal.forward_decode(x_step, (&cos, &sin), &mut causal_cache);
    }
    let causal_ms = start.elapsed().as_secs_f64() * 1000.0;

    // tau
    let mut tau_cache: TauCacheLayer<TestBackend> = None;
    let start = Instant::now();
    for _ in 0..iters {
        let x_step: Tensor<TestBackend, 3> = Tensor::random(
            [b, 1, c],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        let _ = tau.forward_decode(x_step, (&cos, &sin), &mut tau_cache);
    }
    let tau_ms = start.elapsed().as_secs_f64() * 1000.0;

    info!("decode iters={}", iters);
    info!(
        "causal total ms={:.3}  per-step us={:.3}",
        causal_ms,
        1000.0 * causal_ms / iters as f64
    );
    info!(
        "tau    total ms={:.3}  per-step us={:.3}",
        tau_ms,
        1000.0 * tau_ms / iters as f64
    );
}
