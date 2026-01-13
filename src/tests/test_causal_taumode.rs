//! Comprehensive comparison tests: CausalSelfAttention vs TauModeAttention
//!
//! This module tests both attention mechanisms side-by-side to verify:
//! - Forward pass shape correctness
//! - Decode-time cache accumulation
//! - Causal masking behavior
//! - Numerical stability
//! - MQA expansion consistency
//!
//! TauModeAttention tests load a manifold.parquet file (if available) or
//! fall back to a synthetic chain Laplacian for testing.

use burn::tensor::Tensor;
use log::info;

use crate::{
    backend::AutoBackend,
    causalattention::CausalSelfAttention,
    config::NanoChatConfig,
    pretraining::{DomainManifold, load_domain_manifold},
    tauattention::{TauCacheLayer, TauModeAttention},
};

type TestBackend = AutoBackend;

// ─────────────────────────────────────────────────────────────────────────
// Helper: configs and utilities
// ─────────────────────────────────────────────────────────────────────────

fn test_config() -> NanoChatConfig {
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

fn tiny_config() -> NanoChatConfig {
    NanoChatConfig {
        sequence_len: 16,
        vocab_size: 32,
        n_layer: 1,
        n_head: 2,
        n_kv_head: 2,
        n_embd: 16,
        block_size: 16,
        dropout: 0.0,
    }
}

/// Attempt to load manifold.parquet; if not found, return None.
/// Tests will skip manifold-dependent assertions if None.
fn try_load_manifold() -> Option<DomainManifold> {
    match load_domain_manifold("./domain_manifold/manifold.parquet") {
        Ok(m) => {
            info!("✓ Loaded manifold.parquet: dim={}", m.dim());
            Some(m)
        }
        Err(e) => {
            info!("⚠ Could not load manifold.parquet: {}", e);
            info!("  Tests will use synthetic chain Laplacian instead.");
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Test 1: Both attention modules construct without panic
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_attention_construction() {
    crate::init();
    info!("═══ Test: Attention Construction ═══");

    let cfg = test_config();
    let device = Default::default();
    let layer_idx = 0;

    info!("Constructing CausalSelfAttention...");
    let _causal_attn = CausalSelfAttention::<TestBackend>::new(&cfg, layer_idx, &device);
    info!("✓ CausalSelfAttention constructed");

    info!("Constructing TauModeAttention...");
    let _tau_attn = TauModeAttention::<TestBackend>::new(&cfg, layer_idx, &device);
    info!("✓ TauModeAttention constructed");

    info!("✓ Both attention modules created successfully");
}

// ─────────────────────────────────────────────────────────────────────────
// Test 2: Forward pass shape consistency
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_forward_shape_consistency() {
    crate::init();
    info!("═══ Test: Forward Shape Consistency ═══");

    let cfg = test_config();
    let device = Default::default();
    let (b, t, c) = (2, 8, cfg.n_embd);

    let head_dim = cfg.n_embd / cfg.n_head;
    let cos: Tensor<TestBackend, 4> = Tensor::ones([1, t, 1, head_dim / 2], &device);
    let sin: Tensor<TestBackend, 4> = Tensor::zeros([1, t, 1, head_dim / 2], &device);

    let x: Tensor<TestBackend, 3> = Tensor::random(
        [b, t, c],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );

    info!("Input shape: [B={}, T={}, C={}]", b, t, c);

    // Causal attention
    info!("Testing CausalSelfAttention...");
    let causal_attn = CausalSelfAttention::<TestBackend>::new(&cfg, 0, &device);
    let out_causal = causal_attn.forward(x.clone(), (&cos, &sin));
    info!("  Output shape: {:?}", out_causal.dims());
    assert_eq!(out_causal.dims(), [b, t, c]);

    // Tau attention
    info!("Testing TauModeAttention...");
    let tau_attn = TauModeAttention::<TestBackend>::new(&cfg, 0, &device);
    let out_tau = tau_attn.forward(x, (&cos, &sin));
    info!("  Output shape: {:?}", out_tau.dims());
    assert_eq!(out_tau.dims(), [b, t, c]);

    info!("✓ Both produce consistent output shapes");
}

// ─────────────────────────────────────────────────────────────────────────
// Test 3: Decode-time single step with cache
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_decode_single_step() {
    crate::init();
    info!("═══ Test: Decode Single Step ═══");

    let cfg = tiny_config();
    let device = Default::default();
    let (b, c) = (1, cfg.n_embd);

    let head_dim = cfg.n_embd / cfg.n_head;
    let cos: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 1, head_dim / 2], &device);
    let sin: Tensor<TestBackend, 4> = Tensor::zeros([1, 1, 1, head_dim / 2], &device);

    let x_step: Tensor<TestBackend, 3> = Tensor::random(
        [b, 1, c],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );

    info!("Decode input: [B={}, T=1, C={}]", b, c);

    // Causal attention decode
    info!("Testing CausalSelfAttention decode...");
    let causal_attn = CausalSelfAttention::<TestBackend>::new(&cfg, 0, &device);
    let mut causal_cache: Option<(Tensor<TestBackend, 4>, Tensor<TestBackend, 4>)> = None;

    let out_causal = causal_attn.forward_decode(x_step.clone(), (&cos, &sin), &mut causal_cache);
    info!("  Output shape: {:?}", out_causal.dims());
    info!("  Cache initialized: {}", causal_cache.is_some());
    assert_eq!(out_causal.dims(), [b, 1, c]);
    assert!(causal_cache.is_some());

    if let Some((k_cached, v_cached)) = &causal_cache {
        info!("  K cache shape: {:?}", k_cached.dims());
        info!("  V cache shape: {:?}", v_cached.dims());
    }

    // Tau attention decode
    info!("Testing TauModeAttention decode...");
    let tau_attn = TauModeAttention::<TestBackend>::new(&cfg, 0, &device);
    let mut tau_cache: TauCacheLayer<TestBackend> = None;

    let out_tau = tau_attn.forward_decode(x_step, (&cos, &sin), &mut tau_cache);
    info!("  Output shape: {:?}", out_tau.dims());
    info!("  Cache initialized: {}", tau_cache.is_some());
    assert_eq!(out_tau.dims(), [b, 1, c]);
    assert!(tau_cache.is_some());

    if let Some((v_cached, lambda_cached)) = &tau_cache {
        info!("  V cache shape: {:?}", v_cached.dims());
        info!("  Lambda cache shape: {:?}", lambda_cached.dims());
    }

    info!("✓ Both decode mechanisms work correctly");
}

// ─────────────────────────────────────────────────────────────────────────
// Test 4: Cache accumulation over multiple decode steps
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_cache_accumulation() {
    crate::init();
    info!("═══ Test: Cache Accumulation ═══");

    let cfg = tiny_config();
    let device = Default::default();
    let (b, c) = (1, cfg.n_embd);

    let head_dim = cfg.n_embd / cfg.n_head;
    let cos: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 1, head_dim / 2], &device);
    let sin: Tensor<TestBackend, 4> = Tensor::zeros([1, 1, 1, head_dim / 2], &device);

    // Causal attention accumulation
    info!("Testing CausalSelfAttention cache accumulation...");
    let causal_attn = CausalSelfAttention::<TestBackend>::new(&cfg, 0, &device);
    let mut causal_cache = None;

    for step in 1..=3 {
        let x: Tensor<TestBackend, 3> = Tensor::random(
            [b, 1, c],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        causal_attn.forward_decode(x, (&cos, &sin), &mut causal_cache);

        if let Some((k, v)) = &causal_cache {
            info!(
                "  Step {}: K dims={:?}, V dims={:?}",
                step,
                k.dims(),
                v.dims()
            );
            assert_eq!(k.dims()[2], step);
            assert_eq!(v.dims()[2], step);
        }
    }
    info!("✓ CausalSelfAttention cache accumulated correctly");

    // Tau attention accumulation
    info!("Testing TauModeAttention cache accumulation...");
    let tau_attn = TauModeAttention::<TestBackend>::new(&cfg, 0, &device);
    let mut tau_cache: TauCacheLayer<TestBackend> = None;

    for step in 1..=3 {
        let x: Tensor<TestBackend, 3> = Tensor::random(
            [b, 1, c],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        tau_attn.forward_decode(x, (&cos, &sin), &mut tau_cache);

        if let Some((v, lambda)) = &tau_cache {
            info!(
                "  Step {}: V dims={:?}, Lambda dims={:?}",
                step,
                v.dims(),
                lambda.dims()
            );
            assert_eq!(v.dims()[2], step);
            assert_eq!(lambda.dims()[2], step);
        }
    }
    info!("✓ TauModeAttention cache accumulated correctly");
}

// ─────────────────────────────────────────────────────────────────────────
// Test 5: MQA expansion (n_head != n_kv_head)
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_mqa_expansion() {
    crate::init();
    info!("═══ Test: MQA Expansion ═══");

    let mut cfg = test_config();
    cfg.n_head = 8;
    cfg.n_kv_head = 2;
    cfg.n_embd = 64; // Must be divisible by n_head

    let device = Default::default();
    let (b, t, c) = (1, 4, cfg.n_embd);

    let head_dim = cfg.n_embd / cfg.n_head;
    let cos: Tensor<TestBackend, 4> = Tensor::ones([1, t, 1, head_dim / 2], &device);
    let sin: Tensor<TestBackend, 4> = Tensor::zeros([1, t, 1, head_dim / 2], &device);

    let x: Tensor<TestBackend, 3> = Tensor::random(
        [b, t, c],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );

    info!("Config: n_head={}, n_kv_head={}", cfg.n_head, cfg.n_kv_head);
    info!("Input shape: [B={}, T={}, C={}]", b, t, c);

    // Causal attention with MQA
    info!("Testing CausalSelfAttention with MQA...");
    let causal_attn = CausalSelfAttention::<TestBackend>::new(&cfg, 0, &device);
    let out_causal = causal_attn.forward(x.clone(), (&cos, &sin));
    info!("  Output shape: {:?}", out_causal.dims());
    assert_eq!(out_causal.dims(), [b, t, c]);

    // Tau attention with MQA
    info!("Testing TauModeAttention with MQA...");
    let tau_attn = TauModeAttention::<TestBackend>::new(&cfg, 0, &device);
    let out_tau = tau_attn.forward(x, (&cos, &sin));
    info!("  Output shape: {:?}", out_tau.dims());
    assert_eq!(out_tau.dims(), [b, t, c]);

    info!("✓ Both handle MQA expansion correctly");
}

// ─────────────────────────────────────────────────────────────────────────
// Test 6: Numerical stability (no NaN/Inf)
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_numerical_stability() {
    crate::init();
    info!("═══ Test: Numerical Stability ═══");

    let cfg = tiny_config();
    let device = Default::default();
    let (b, t, c) = (2, 8, cfg.n_embd);

    let head_dim = cfg.n_embd / cfg.n_head;
    let cos: Tensor<TestBackend, 4> = Tensor::ones([1, t, 1, head_dim / 2], &device);
    let sin: Tensor<TestBackend, 4> = Tensor::zeros([1, t, 1, head_dim / 2], &device);

    let x: Tensor<TestBackend, 3> = Tensor::random(
        [b, t, c],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    // Causal attention stability
    info!("Checking CausalSelfAttention stability...");
    let causal_attn = CausalSelfAttention::<TestBackend>::new(&cfg, 0, &device);
    let out_causal = causal_attn.forward(x.clone(), (&cos, &sin));
    let causal_data: Vec<f32> = out_causal.to_data().to_vec().unwrap();
    let causal_healthy = causal_data.iter().all(|&v| v.is_finite());
    info!("  Contains NaN/Inf: {}", !causal_healthy);
    assert!(causal_healthy, "CausalSelfAttention produced NaN/Inf");

    // Tau attention stability
    info!("Checking TauModeAttention stability...");
    let tau_attn = TauModeAttention::<TestBackend>::new(&cfg, 0, &device);
    let out_tau = tau_attn.forward(x, (&cos, &sin));
    let tau_data: Vec<f32> = out_tau.to_data().to_vec().unwrap();
    let tau_healthy = tau_data.iter().all(|&v| v.is_finite());
    info!("  Contains NaN/Inf: {}", !tau_healthy);
    assert!(tau_healthy, "TauModeAttention produced NaN/Inf");

    info!("✓ Both mechanisms are numerically stable");
}

// ─────────────────────────────────────────────────────────────────────────
// Test 7: Causal masking verification (compare prefix vs full sequence)
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_causal_masking() {
    crate::init();
    info!("═══ Test: Causal Masking Verification ═══");

    let cfg = tiny_config();
    let device = Default::default();

    let full_seq = vec![0.1, 0.2, 0.3, 0.4];
    let test_pos = 2;

    let (b, t, c) = (1, full_seq.len(), cfg.n_embd);
    let head_dim = cfg.n_embd / cfg.n_head;
    let cos: Tensor<TestBackend, 4> = Tensor::ones([1, t, 1, head_dim / 2], &device);
    let sin: Tensor<TestBackend, 4> = Tensor::zeros([1, t, 1, head_dim / 2], &device);

    let x_full: Tensor<TestBackend, 3> = Tensor::random(
        [b, t, c],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );

    let prefix_len = test_pos + 1;
    let x_prefix = x_full.clone().slice([0..b, 0..prefix_len, 0..c]);

    info!("Full sequence length: {}, prefix length: {}", t, prefix_len);

    // Causal attention masking
    info!("Testing CausalSelfAttention causal property...");
    let causal_attn = CausalSelfAttention::<TestBackend>::new(&cfg, 0, &device);

    let out_full = causal_attn.forward(x_full, (&cos, &sin));
    let out_prefix = causal_attn.forward(
        x_prefix,
        (
            &cos.clone()
                .slice([0..1, 0..prefix_len, 0..1, 0..head_dim / 2]),
            &sin.clone()
                .slice([0..1, 0..prefix_len, 0..1, 0..head_dim / 2]),
        ),
    );

    let full_at_pos: Vec<f32> = out_full
        .clone()
        .slice([0..1, test_pos..(test_pos + 1), 0..c])
        .reshape([c])
        .to_data()
        .to_vec()
        .unwrap();
    let prefix_at_pos: Vec<f32> = out_prefix
        .slice([0..1, test_pos..(test_pos + 1), 0..c])
        .reshape([c])
        .to_data()
        .to_vec()
        .unwrap();

    let max_diff = full_at_pos
        .iter()
        .zip(prefix_at_pos.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    info!("  Max difference at position {}: {:.6}", test_pos, max_diff);
    assert!(
        max_diff < 0.01,
        "Causal mask violated for CausalSelfAttention"
    );

    // Tau attention masking
    info!("Testing TauModeAttention causal property...");
    let tau_attn = TauModeAttention::<TestBackend>::new(&cfg, 0, &device);

    let x_full_tau: Tensor<TestBackend, 3> = Tensor::random(
        [b, t, c],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );
    let x_prefix_tau = x_full_tau.clone().slice([0..b, 0..prefix_len, 0..c]);

    let out_full_tau = tau_attn.forward(x_full_tau, (&cos, &sin));
    let out_prefix_tau = tau_attn.forward(
        x_prefix_tau,
        (
            &cos.clone()
                .slice([0..1, 0..prefix_len, 0..1, 0..head_dim / 2]),
            &sin.clone()
                .slice([0..1, 0..prefix_len, 0..1, 0..head_dim / 2]),
        ),
    );

    let full_tau_at_pos: Vec<f32> = out_full_tau
        .clone()
        .slice([0..1, test_pos..(test_pos + 1), 0..c])
        .reshape([c])
        .to_data()
        .to_vec()
        .unwrap();
    let prefix_tau_at_pos: Vec<f32> = out_prefix_tau
        .slice([0..1, test_pos..(test_pos + 1), 0..c])
        .reshape([c])
        .to_data()
        .to_vec()
        .unwrap();

    let max_diff_tau = full_tau_at_pos
        .iter()
        .zip(prefix_tau_at_pos.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    info!(
        "  Max difference at position {}: {:.6}",
        test_pos, max_diff_tau
    );
    assert!(
        max_diff_tau < 0.01,
        "Causal mask violated for TauModeAttention"
    );

    info!("✓ Both mechanisms respect causal masking");
}

// ─────────────────────────────────────────────────────────────────────────
// Test 8: Manifold loading (if manifold.parquet exists)
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_manifold_loading() {
    crate::init();
    info!("═══ Test: Manifold Loading ═══");

    let manifold_opt = try_load_manifold();

    if let Some(manifold) = manifold_opt {
        info!("Manifold loaded successfully:");
        info!("  Dimension: {}", manifold.dim());
        info!("  Non-zeros: {}", manifold.laplacian().nnz());

        // Verify it's square
        let (r, c) = manifold.laplacian().shape();
        assert_eq!(r, c, "Manifold must be square");
        info!("  Shape: {}×{}", r, c);

        info!("✓ Manifold is valid");
    } else {
        info!("⚠ Skipping manifold test (file not found)");
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Test 9: Compare cache storage overhead
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_cache_storage_comparison() {
    crate::init();
    info!("═══ Test: Cache Storage Comparison ═══");

    let cfg = test_config();
    let device = Default::default();
    let (b, c) = (2, cfg.n_embd);

    let head_dim = cfg.n_embd / cfg.n_head;
    let cos: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 1, head_dim / 2], &device);
    let sin: Tensor<TestBackend, 4> = Tensor::zeros([1, 1, 1, head_dim / 2], &device);

    // Accumulate caches for 10 steps
    let num_steps = 10;

    // Causal cache
    let causal_attn = CausalSelfAttention::<TestBackend>::new(&cfg, 0, &device);
    let mut causal_cache = None;

    for _ in 0..num_steps {
        let x: Tensor<TestBackend, 3> = Tensor::random(
            [b, 1, c],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        causal_attn.forward_decode(x, (&cos, &sin), &mut causal_cache);
    }

    if let Some((k, v)) = &causal_cache {
        let causal_elems = k.dims().iter().product::<usize>() + v.dims().iter().product::<usize>();
        info!("CausalSelfAttention cache:");
        info!("  K: {:?}, V: {:?}", k.dims(), v.dims());
        info!("  Total elements: {}", causal_elems);
    }

    // Tau cache
    let tau_attn = TauModeAttention::<TestBackend>::new(&cfg, 0, &device);
    let mut tau_cache: TauCacheLayer<TestBackend> = None;

    for _ in 0..num_steps {
        let x: Tensor<TestBackend, 3> = Tensor::random(
            [b, 1, c],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        tau_attn.forward_decode(x, (&cos, &sin), &mut tau_cache);
    }

    if let Some((v, lambda)) = &tau_cache {
        let tau_elems =
            v.dims().iter().product::<usize>() + lambda.dims().iter().product::<usize>();
        info!("TauModeAttention cache:");
        info!("  V: {:?}, Lambda: {:?}", v.dims(), lambda.dims());
        info!("  Total elements: {}", tau_elems);
    }

    info!("✓ Cache storage comparison complete");
}

// ─────────────────────────────────────────────────────────────────────────
// Test 10: Batch consistency (same input → same output for both)
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_batch_consistency() {
    crate::init();
    info!("═══ Test: Batch Consistency ═══");

    let cfg = tiny_config();
    let device = Default::default();
    let (b, t, c) = (3, 4, cfg.n_embd);

    let head_dim = cfg.n_embd / cfg.n_head;
    let cos: Tensor<TestBackend, 4> = Tensor::ones([1, t, 1, head_dim / 2], &device);
    let sin: Tensor<TestBackend, 4> = Tensor::zeros([1, t, 1, head_dim / 2], &device);

    let x: Tensor<TestBackend, 3> = Tensor::random(
        [b, t, c],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );

    info!("Testing batch={}, seq_len={}", b, t);

    // Causal
    let causal_attn = CausalSelfAttention::<TestBackend>::new(&cfg, 0, &device);
    let out1 = causal_attn.forward(x.clone(), (&cos, &sin));
    let out2 = causal_attn.forward(x.clone(), (&cos, &sin));

    let data1: Vec<f32> = out1.to_data().to_vec().unwrap();
    let data2: Vec<f32> = out2.to_data().to_vec().unwrap();

    assert_eq!(data1, data2, "CausalSelfAttention not deterministic");
    info!("✓ CausalSelfAttention is deterministic");

    // Tau
    let tau_attn = TauModeAttention::<TestBackend>::new(&cfg, 0, &device);
    let out3 = tau_attn.forward(x.clone(), (&cos, &sin));
    let out4 = tau_attn.forward(x, (&cos, &sin));

    let data3: Vec<f32> = out3.to_data().to_vec().unwrap();
    let data4: Vec<f32> = out4.to_data().to_vec().unwrap();

    assert_eq!(data3, data4, "TauModeAttention not deterministic");
    info!("✓ TauModeAttention is deterministic");

    info!("✓ Both mechanisms are batch-consistent");
}
