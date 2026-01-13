//! Stability-focused tests (fail-fast gate):
//! - Extreme logits stress for causal softmax: no NaNs/Infs, row-sum ~ 1, masked ~ 0
//! - Extreme temperature stress for TauMode distance logits + causal softmax
//! - Long-context forward-only model run: no NaNs/Infs and logits stay bounded

#![cfg(feature = "cpu")]

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::causalattention::GptModel;
use crate::config::NanoChatConfig;
use crate::taumode::{TauModeConfig, causal_softmax_over_keys, taumode_distance_logits};

type B = burn_cpu::Cpu<f32>;
type Dev = <B as Backend>::Device;

fn assert_all_finite(name: &str, xs: &[f32]) {
    for (i, &x) in xs.iter().enumerate() {
        assert!(x.is_finite(), "{name}: non-finite at idx={i}: {x}");
    }
}

fn assert_probs_reasonable(name: &str, probs: &[f32], eps: f32) {
    for (i, &p) in probs.iter().enumerate() {
        assert!(p.is_finite(), "{name}: prob non-finite at idx={i}: {p}");
        assert!(p >= -eps, "{name}: prob < 0 at idx={i}: {p}");
        assert!(p <= 1.0 + eps, "{name}: prob > 1 at idx={i}: {p}");
    }
}

fn lcg_u32(state: &mut u32) -> u32 {
    // Simple deterministic LCG for tests (no external RNG dependency).
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    *state
}

#[test]
fn extreme_logits_causal_softmax_no_nan() {
    let device = Dev::default();

    let (b, h, tq, tk) = (1usize, 2usize, 8usize, 8usize);

    // Build extreme logits [B,H,Tq,Tk] in a way that stresses stability.
    // We'll keep the diagonal huge positive and off-diagonal huge negative.
    let mut data = vec![0.0f32; b * h * tq * tk];
    for hh in 0..h {
        for i in 0..tq {
            for j in 0..tk {
                let idx = ((hh * tq + i) * tk + j) as usize;
                let base = if j == i { 1.0e4 } else { -1.0e4 };
                // Slightly different per-head scale to avoid accidental symmetry.
                data[idx] = base + (hh as f32) * 13.0;
            }
        }
    }

    let logits = Tensor::<B, 1>::from_floats(data.as_slice(), &device).reshape([b, h, tq, tk]);
    let w = causal_softmax_over_keys(logits, tq, tk, h);

    let w_host: Vec<f32> = w.to_data().to_vec().unwrap();
    assert_all_finite("extreme_logits.softmax.weights", &w_host);

    // Check probability constraints + row sums + masking.
    // Layout: [B,H,Tq,Tk]. Here B=1.
    for hh in 0..h {
        for i in 0..tq {
            let row0 = (hh * tq + i) * tk;
            let row = &w_host[row0..row0 + tk];

            assert_probs_reasonable("extreme_logits.softmax.weights", row, 1e-6);

            // Causal mask for tq==tk blocks positions j>i.
            for j in (i + 1)..tk {
                let p = row[j];
                assert!(
                    p.abs() <= 1e-6,
                    "masked weight not ~0 at (h={hh}, tq={i}, tk={j}): {p}"
                );
            }

            // Row sum ~ 1.
            let s: f32 = row.iter().sum();
            assert!(
                (s - 1.0).abs() <= 1e-5,
                "row sum not ~1 at (h={hh}, tq={i}): {s}"
            );
        }
    }
}

#[test]
fn extreme_temperature_taumode_softmax_no_nan() {
    let device = Dev::default();

    let (b, h, tq, tk) = (1usize, 1usize, 8usize, 8usize);

    // Very small temperature => huge-magnitude logits in taumode_distance_logits.
    let cfg = TauModeConfig {
        tau: 1.0,
        eps: 1.0e-6,
        temperature: 1.0e-6,
    };

    // Lambdas with spread to generate large |Δλ|/temperature.
    let mut lq = vec![0.0f32; tq];
    let mut lk = vec![0.0f32; tk];
    for i in 0..tq {
        lq[i] = (i as f32) * 0.125;
    }
    for j in 0..tk {
        lk[j] = (j as f32) * 0.25;
    }

    let lambda_q = Tensor::<B, 1>::from_floats(lq.as_slice(), &device).reshape([b, h, tq]);
    let lambda_k = Tensor::<B, 1>::from_floats(lk.as_slice(), &device).reshape([b, h, tk]);

    let logits = taumode_distance_logits(lambda_q, lambda_k, &cfg);
    let logits_host: Vec<f32> = logits.to_data().to_vec().unwrap();
    assert_all_finite("extreme_temperature.distance_logits", &logits_host);

    let w = causal_softmax_over_keys(logits, tq, tk, h);
    let w_host: Vec<f32> = w.to_data().to_vec().unwrap();
    assert_all_finite("extreme_temperature.softmax.weights", &w_host);

    // Check row sums on unmasked prefixes.
    for i in 0..tq {
        let row0 = i * tk;
        let row = &w_host[row0..row0 + tk];

        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() <= 1e-5, "row sum not ~1 at tq={i}: {s}");

        for j in (i + 1)..tk {
            assert!(
                row[j].abs() <= 1e-6,
                "masked weight not ~0 at (tq={i}, tk={j}): {}",
                row[j]
            );
        }
    }
}

#[test]
fn long_context_gpt_forward_no_nan_and_bounded_logits() {
    let device = Dev::default();

    // Keep it small enough for CPU test speed, but long enough to hit the “long context” path.
    let t = 256usize;
    let vocab = 128usize;

    let cfg = NanoChatConfig {
        sequence_len: t,
        vocab_size: vocab,
        n_layer: 2,
        n_head: 4,
        n_kv_head: 4,
        n_embd: 64,
        block_size: t,
        dropout: 0.0,
    };

    let model = GptModel::<B>::new(&cfg, &device);

    // Deterministic pseudo-random token ids in [0, vocab).
    let mut state = 123456789u32;
    let mut ids = Vec::with_capacity(t);
    for _ in 0..t {
        let r = lcg_u32(&mut state) as usize;
        ids.push((r % vocab) as i64);
    }

    let idx = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device).reshape([1, t]);

    // Disable softcap but keep the model’s safety clamp, matching the implementation style.
    let logits = model.forward(idx, false);

    let logits_host: Vec<f32> = logits.to_data().to_vec().unwrap();
    assert_all_finite("long_context.forward.logits", &logits_host);

    // Implementation clamps logits to [-50, 50], so enforce that as a stability invariant.
    let mut max_abs = 0.0f32;
    for &x in &logits_host {
        max_abs = max_abs.max(x.abs());
    }
    assert!(
        max_abs <= 50.0 + 1e-3,
        "logits not bounded as expected; max_abs={max_abs}"
    );
}
