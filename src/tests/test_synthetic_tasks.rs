//! tests/synthetic_tasks.rs
//!
//! Fast synthetic “token accuracy” suites (no training, no external data).
//!
//! These are *diagnostic* tasks designed to fail if attention scoring/masking/softmax
//! are wrong, by creating contexts where the next token is deterministically implied. [file:15]
//!
//! Implementation note:
//! We build a tiny “retrieval head” using TauMode distance attention directly:
//! 1) choose lambdas so the correct key is uniquely closest to the query
//! 2) set V as one-hot token IDs (or next-token IDs for induction)
//! 3) attention output becomes a probability distribution over vocab
//! 4) argmax must match the expected token

#![cfg(feature = "cpu")]

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::taumode::{TauModeConfig, causal_softmax_over_keys, taumode_distance_logits};

type B = burn_ndarray::NdArray<f32>;
type Dev = <B as Backend>::Device;

fn lcg_u32(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    *state
}

fn argmax(xs: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn assert_all_finite(name: &str, xs: &[f32]) {
    for (i, &x) in xs.iter().enumerate() {
        assert!(x.is_finite(), "{name}: non-finite at idx={i}: {x}");
    }
}

fn one_hot_values(v_tokens: &[i64], vocab: usize) -> Vec<f32> {
    let tk = v_tokens.len();
    let mut data = vec![0.0f32; tk * vocab];
    for (pos, &tid) in v_tokens.iter().enumerate() {
        let id = tid as usize;
        assert!(id < vocab, "token id {id} out of vocab={vocab}");
        data[pos * vocab + id] = 1.0;
    }
    data
}

/// Runs TauMode distance attention for a single query (Tq=1) over a key cache.
///
/// - lambda_k: [Tk]
/// - v_tokens: [Tk] (interpreted as the “value token id” at each key position)
/// - query_lambda: scalar
///
/// Returns (predicted_id, full_distribution_over_vocab).
fn predict_one_step(
    device: &Dev,
    lambda_k: &[f32],
    v_tokens: &[i64],
    query_lambda: f32,
    vocab: usize,
    cfg: TauModeConfig,
) -> (usize, Vec<f32>) {
    assert_eq!(lambda_k.len(), v_tokens.len());
    let tk = lambda_k.len();
    let (b, h, tq) = (1usize, 1usize, 1usize);

    // lambda_q: [B,H,1]
    let lambda_q = Tensor::<B, 1>::from_floats([query_lambda], device).reshape([b, h, tq]);

    // lambda_k: [B,H,Tk]
    let lambda_k_t = Tensor::<B, 1>::from_floats(lambda_k, device).reshape([b, h, tk]);

    // V one-hot: [B,H,Tk,V]
    let v_data = one_hot_values(v_tokens, vocab);
    let v = Tensor::<B, 1>::from_floats(v_data.as_slice(), device).reshape([b, h, tk, vocab]);

    // logits -> stable causal softmax (decode-style: Tq=1, Tk=tk => diag=Tk-1 allows all keys)
    let logits = taumode_distance_logits(lambda_q, lambda_k_t, &cfg);
    let att = causal_softmax_over_keys(logits, tq, tk, h); // [B,H,1,Tk]

    // Distribution over vocab: [B,H,1,V]
    let out = att.matmul(v);
    let dist = out.reshape([vocab]).to_data().to_vec().unwrap();

    assert_all_finite("synthetic_tasks.dist", &dist);

    // Sum should be ~1 since it's a convex combination of one-hots.
    let s: f32 = dist.iter().sum();
    assert!((s - 1.0).abs() <= 1e-4, "dist sum not ~1: {s}");

    let pred = argmax(&dist);
    (pred, dist)
}

#[test]
fn task_copy_selective_copy() {
    let device = Dev::default();

    // Small vocab + many trials; still very fast on CPU.
    let vocab = 64usize;
    let t = 33usize; // include a query token at the end
    let trials = 128usize;

    // Very sharp to make the “closest lambda wins” deterministic.
    let cfg = TauModeConfig {
        tau: 1.0,
        eps: 1e-6,
        temperature: 1e-3,
    };

    let mut state = 0xC0FFEEu32;
    let mut correct = 0usize;

    for _ in 0..trials {
        // Tokens for positions 0..t-2 are “content”; position t-1 is a query marker (unused).
        let mut tokens = vec![0i64; t];
        for i in 0..(t - 1) {
            let r = lcg_u32(&mut state) as usize;
            tokens[i] = (r % vocab) as i64;
        }
        tokens[t - 1] = 0; // marker

        // Choose which earlier token must be copied.
        let copy_i = (lcg_u32(&mut state) as usize) % (t - 1);
        let expected = tokens[copy_i] as usize;

        // Keys exclude the query position: Tk = t-1
        let tk = t - 1;
        let mut lambda_k = vec![0.0f32; tk];
        for i in 0..tk {
            lambda_k[i] = i as f32; // positional lambdas
        }
        let v_tokens = tokens[..tk].to_vec();

        // Query lambda points exactly at the desired key index.
        let query_lambda = copy_i as f32;

        let (pred, _dist) =
            predict_one_step(&device, &lambda_k, &v_tokens, query_lambda, vocab, cfg);

        if pred == expected {
            correct += 1;
        } else {
            panic!("copy task failed: expected={expected}, pred={pred}, copy_i={copy_i}");
        }
    }

    let acc = (correct as f32) / (trials as f32);
    assert!(acc >= 0.999, "copy task accuracy too low: {acc}");
}

#[test]
fn task_kv_associative_recall() {
    let device = Dev::default();

    let vocab = 64usize;
    let pairs = 6usize; // k:v repeated
    let trials = 128usize;

    let cfg = TauModeConfig {
        tau: 1.0,
        eps: 1e-6,
        temperature: 1e-3,
    };

    let mut state = 0xBAD5EEDu32;
    let mut correct = 0usize;

    // helper: sample a key in [1..16)
    let sample_key = |st: &mut u32| -> i64 { (1 + ((lcg_u32(st) as usize) % 15)) as i64 };
    // helper: sample a value in [16..48)
    let sample_val = |st: &mut u32| -> i64 { (16 + ((lcg_u32(st) as usize) % 32)) as i64 };

    for _ in 0..trials {
        // Build UNIQUE keys to make the mapping deterministic.
        let mut keys: Vec<i64> = Vec::with_capacity(pairs);
        while keys.len() < pairs {
            let k = sample_key(&mut state);
            if !keys.contains(&k) {
                keys.push(k);
            }
        }

        let mut vals = vec![0i64; pairs];
        for p in 0..pairs {
            vals[p] = sample_val(&mut state);
        }

        let q_pair = (lcg_u32(&mut state) as usize) % pairs;
        let query_key = keys[q_pair];
        let expected = vals[q_pair] as usize;

        // Token stream: [k1,v1,k2,v2,...,kp,vp, query_key]
        let mut tokens: Vec<i64> = Vec::with_capacity(2 * pairs + 1);
        for p in 0..pairs {
            tokens.push(keys[p]);
            tokens.push(vals[p]);
        }
        tokens.push(query_key);

        // Keys exclude the query token.
        let tk = tokens.len() - 1;
        let key_stream = &tokens[..tk];

        // Lambda scheme:
        // - VALUE positions: lambda = key_id (so query targets the value)
        // - KEY positions:   lambda = key_id + big_offset (far away)
        let big = 1000.0f32;
        let mut lambda_k = vec![0.0f32; tk];
        for pos in 0..tk {
            if pos % 2 == 1 {
                let k = key_stream[pos - 1] as f32; // the key associated with this value
                lambda_k[pos] = k;
            } else {
                let k = key_stream[pos] as f32;
                lambda_k[pos] = k + big;
            }
        }

        let v_tokens = key_stream.to_vec();
        let query_lambda = query_key as f32;

        let (pred, _dist) =
            predict_one_step(&device, &lambda_k, &v_tokens, query_lambda, vocab, cfg);

        if pred == expected {
            correct += 1;
        } else {
            panic!("kv recall failed: expected={expected}, pred={pred}, query_key={query_key}");
        }
    }

    let acc = (correct as f32) / (trials as f32);
    assert!(acc >= 0.999, "kv task accuracy too low: {acc}");
}

#[test]
fn task_needle_in_haystack() {
    let device = Dev::default();

    let vocab = 64usize;
    let t = 129usize; // last token is query marker; keys are 0..t-2 (128 tokens)
    let trials = 64usize;

    let cfg = TauModeConfig {
        tau: 1.0,
        eps: 1e-6,
        temperature: 1e-3,
    };

    let mut state = 0x1234ABCDu32;
    let mut correct = 0usize;

    for _ in 0..trials {
        let mut tokens = vec![0i64; t];
        for i in 0..(t - 1) {
            let r = lcg_u32(&mut state) as usize;
            tokens[i] = (r % vocab) as i64;
        }

        // Place a marker + payload far back.
        let marker: i64 = 1;
        let payload: i64 = (2 + ((lcg_u32(&mut state) as usize) % (vocab - 2))) as i64;

        let needle_pos = 8 + ((lcg_u32(&mut state) as usize) % (t - 1 - 16)); // avoid edges
        tokens[needle_pos] = marker;
        tokens[needle_pos + 1] = payload;

        // Query marker at end (not used for keys)
        tokens[t - 1] = marker;

        let expected = payload as usize;

        // Keys exclude query position.
        let tk = t - 1;
        let v_tokens = tokens[..tk].to_vec();

        // Positional lambdas, query targets payload position.
        let mut lambda_k = vec![0.0f32; tk];
        for i in 0..tk {
            lambda_k[i] = i as f32;
        }
        let query_lambda = (needle_pos + 1) as f32;

        let (pred, _dist) =
            predict_one_step(&device, &lambda_k, &v_tokens, query_lambda, vocab, cfg);

        if pred == expected {
            correct += 1;
        } else {
            panic!(
                "needle failed: expected={expected}, pred={pred}, payload_pos={}",
                needle_pos + 1
            );
        }
    }

    let acc = (correct as f32) / (trials as f32);
    assert!(acc >= 0.999, "needle task accuracy too low: {acc}");
}

#[test]
fn task_induction_head_pattern() {
    let device = Dev::default();

    let vocab = 64usize;
    let t = 33usize; // last token is the query 'A'
    let trials = 128usize;

    let cfg = TauModeConfig {
        tau: 1.0,
        eps: 1e-6,
        temperature: 1e-3,
    };

    let mut state = 0xDEADBEEFu32;
    let mut correct = 0usize;

    for _ in 0..trials {
        // Construct: A, B, filler..., A  (query is final A; expected next token is B)
        let a: i64 = (2 + ((lcg_u32(&mut state) as usize) % 20)) as i64;
        let mut b_tok: i64 = (24 + ((lcg_u32(&mut state) as usize) % 20)) as i64;
        if b_tok == a {
            b_tok = (b_tok + 1) % (vocab as i64);
        }

        let mut tokens = vec![0i64; t];
        tokens[0] = a;
        tokens[1] = b_tok;

        // Fill positions 2..t-2 with tokens not equal to A to keep the induction unambiguous.
        for i in 2..(t - 1) {
            let mut v = (2 + ((lcg_u32(&mut state) as usize) % (vocab - 2))) as i64;
            if v == a {
                v = (v + 1) % (vocab as i64);
                if v < 2 {
                    v = 2;
                }
            }
            tokens[i] = v;
        }

        // Query token at end.
        tokens[t - 1] = a;

        let expected = b_tok as usize;

        // Keys are positions 0..t-2 (exclude query token).
        let tk = t - 1;
        let keys = &tokens[..tk];

        // Lambda scheme for induction:
        // - lambda_k[pos] = token_id_at_pos
        // - query_lambda = A
        // Then attention selects the *previous A* key position.
        let mut lambda_k = vec![0.0f32; tk];
        for pos in 0..tk {
            lambda_k[pos] = keys[pos] as f32;
        }
        let query_lambda = a as f32;

        // Values are "next token after key position" (classic induction head trick):
        // v_tokens[pos] = tokens[pos+1]
        // For the last key pos=tk-1 (which corresponds to tokens[t-2]), we define next as 0 (unused).
        let mut v_tokens = vec![0i64; tk];
        for pos in 0..(tk - 1) {
            v_tokens[pos] = tokens[pos + 1];
        }
        v_tokens[tk - 1] = 0;

        let (pred, _dist) =
            predict_one_step(&device, &lambda_k, &v_tokens, query_lambda, vocab, cfg);

        if pred == expected {
            correct += 1;
        } else {
            panic!("induction failed: expected={expected}, pred={pred}, A={a}, B={b_tok}");
        }
    }

    let acc = (correct as f32) / (trials as f32);
    assert!(acc >= 0.999, "induction task accuracy too low: {acc}");
}
