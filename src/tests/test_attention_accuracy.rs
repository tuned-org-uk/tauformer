//! tests/attention_accuracy.rs
//!
//! Accuracy-focused tests for TauMode (lambda-distance) attention:
//! - Reference equivalence on a tiny case (explicit loops vs Burn ops)
//! - Causal masking: masked positions get ~0 weight
//! - Probability simplex: weights row-sum ~ 1
//! - Softmax shift-invariance: softmax(L + c) == softmax(L)

#![cfg(feature = "cpu")]

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::taumode::{TauModeConfig, causal_softmax_over_keys, taumode_distance_logits};

type B = burn_cpu::Cpu<f32>;
type Dev = <B as Backend>::Device;

fn assert_allclose(name: &str, a: &[f32], b: &[f32], atol: f32, rtol: f32) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs().max(x.abs());
        assert!(
            diff <= tol || (x.is_nan() && y.is_nan()),
            "{name}: idx={i} a={x} b={y} diff={diff} tol={tol}"
        );
    }
}

/// Stable softmax over a slice, where indices >= allowed_len are treated as masked (prob=0).
fn softmax_masked_prefix(logits: &[f32], allowed_len: usize) -> Vec<f32> {
    assert!(allowed_len <= logits.len());

    if allowed_len == 0 {
        return vec![0.0; logits.len()];
    }

    // max over allowed
    let mut m = f32::NEG_INFINITY;
    for &v in &logits[..allowed_len] {
        if v > m {
            m = v;
        }
    }

    // exp and sum over allowed; masked -> 0
    let mut out = vec![0.0f32; logits.len()];
    let mut sum = 0.0f32;
    for i in 0..allowed_len {
        let e = (logits[i] - m).exp();
        out[i] = e;
        sum += e;
    }

    if !sum.is_finite() || sum <= 0.0 {
        let u = 1.0 / (allowed_len as f32);
        for i in 0..allowed_len {
            out[i] = u;
        }
        return out;
    }

    for i in 0..allowed_len {
        out[i] /= sum;
    }
    out
}

#[test]
fn taumode_distance_logits_matches_naive() {
    let device = Dev::default();
    let cfg = TauModeConfig {
        tau: 1.0,
        eps: 1.0e-6,
        temperature: 0.5,
    };

    let (b, h, tq, tk) = (1usize, 1usize, 3usize, 4usize);

    // lambda_q: [B,H,Tq]
    let lq_data = [0.0f32, 0.5, 1.0];
    let lambda_q = Tensor::<B, 1>::from_floats(lq_data, &device).reshape([b, h, tq]);

    // lambda_k: [B,H,Tk]
    let lk_data = [0.0f32, 0.25, 0.5, 1.0];
    let lambda_k = Tensor::<B, 1>::from_floats(lk_data, &device).reshape([b, h, tk]);

    let logits = taumode_distance_logits(lambda_q, lambda_k, &cfg);
    let got: Vec<f32> = logits.to_data().to_vec().expect("logits must be f32");

    // Naive broadcast: att[q,k] = -|lq-lk|/temp
    let mut exp = vec![0.0f32; b * h * tq * tk];
    let mut idx = 0;
    for _bb in 0..b {
        for _hh in 0..h {
            for i in 0..tq {
                for j in 0..tk {
                    let v = -((lq_data[i] - lk_data[j]).abs() / cfg.temperature.max(cfg.eps));
                    exp[idx] = v;
                    idx += 1;
                }
            }
        }
    }

    assert_allclose("taumode_distance_logits", &got, &exp, 1e-6, 1e-6);
}

#[test]
fn causal_softmax_shift_invariant() {
    let device = Dev::default();

    // Deterministic logits tensor with some variety.
    let (b, h, tq, tk) = (1usize, 2usize, 4usize, 4usize);
    let mut data = Vec::with_capacity(b * h * tq * tk);
    for hh in 0..h {
        for i in 0..tq {
            for j in 0..tk {
                data.push((hh as f32) * 0.3 + (i as f32) * 0.1 - (j as f32) * 0.2);
            }
        }
    }

    let logits = Tensor::<B, 1>::from_floats(data.as_slice(), &device).reshape([b, h, tq, tk]);
    let w1 = causal_softmax_over_keys(logits.clone(), tq, tk, h);

    // Add a large constant shift to logits; softmax should be unchanged.
    let logits_shifted = logits.add_scalar(123.456);
    let w2 = causal_softmax_over_keys(logits_shifted, tq, tk, h);

    let v1: Vec<f32> = w1.to_data().to_vec().unwrap();
    let v2: Vec<f32> = w2.to_data().to_vec().unwrap();
    assert_allclose("softmax_shift_invariance", &v1, &v2, 1e-6, 1e-6);
}

#[test]
fn tau_attention_pipeline_matches_naive_reference() {
    let device = Dev::default();

    let cfg = TauModeConfig {
        tau: 1.0,
        eps: 1.0e-6,
        temperature: 0.25,
    };

    let (b, h, t, d) = (1usize, 1usize, 4usize, 3usize);
    let tq = t;
    let tk = t;

    // Make lambda_q match lambda_k per position.
    let l_data = [0.0f32, 1.0, 2.0, 4.0];
    let lambda_q = Tensor::<B, 1>::from_floats(l_data, &device).reshape([b, h, tq]);
    let lambda_k = Tensor::<B, 1>::from_floats(l_data, &device).reshape([b, h, tk]);

    // V: per key a distinctive vector.
    // Shape: [B,H,Tk,D]
    let v_data: [f32; 12] = [
        1.0, 0.0, 0.0, // k0
        0.0, 1.0, 0.0, // k1
        0.0, 0.0, 1.0, // k2
        1.0, 1.0, 1.0, // k3
    ];
    let v = Tensor::<B, 1>::from_floats(v_data, &device).reshape([b, h, tk, d]);

    // Fast path: logits -> causal stable softmax -> weighted sum
    let logits = taumode_distance_logits(lambda_q.clone(), lambda_k.clone(), &cfg);
    let w = causal_softmax_over_keys(logits, tq, tk, h);
    let out = w.clone().matmul(v.clone()); // [B,H,Tq,D]
    let out_fast: Vec<f32> = out.to_data().to_vec().unwrap();

    // --- Property checks: masked weights and row-sum ---
    let w_host: Vec<f32> = w.to_data().to_vec().unwrap();
    // Layout [B,H,Tq,Tk], and here B=H=1 => contiguous rows of length Tk.
    for i in 0..tq {
        let row_start = i * tk;

        // masked positions are keys > i (standard causal when tq==tk)
        for j in (i + 1)..tk {
            let p = w_host[row_start + j];
            assert!(
                p.abs() <= 1e-6,
                "masked weight not ~0 at (tq={i}, tk={j}): {p}"
            );
        }

        // row-sum ~ 1
        let mut s = 0.0f32;
        for j in 0..tk {
            s += w_host[row_start + j];
        }
        assert!((s - 1.0).abs() <= 1e-5, "row sum not ~1 at tq={i}: {s}");
    }

    // --- Naive reference (explicit loops) ---
    let mut out_ref = vec![0.0f32; tq * d];

    for i in 0..tq {
        let mut row_logits = vec![0.0f32; tk];
        for j in 0..tk {
            row_logits[j] = -((l_data[i] - l_data[j]).abs() / cfg.temperature.max(cfg.eps));
        }
        let probs = softmax_masked_prefix(&row_logits, i + 1);

        for j in 0..tk {
            let pj = probs[j];
            for dd in 0..d {
                out_ref[i * d + dd] += pj * v_data[j * d + dd];
            }
        }
    }

    // B=H=1 => out is just [Tq,D] flattened.
    assert_eq!(out_fast.len(), out_ref.len());
    assert_allclose(
        "tau_attention_out_vs_naive",
        &out_fast,
        &out_ref,
        2e-5,
        2e-5,
    );

    // Extra weak sanity: position i should be close to V_i.
    for i in 0..tq {
        let yi = &out_fast[i * d..(i + 1) * d];
        let vi = &v_data[i * d..(i + 1) * d];

        let l1: f32 = yi.iter().zip(vi.iter()).map(|(a, b)| (a - b).abs()).sum();

        assert!(
            l1 <= 0.35,
            "position {i}: output not close enough to its matching value; L1={l1}, y={yi:?}, v={vi:?}"
        );
    }
}
