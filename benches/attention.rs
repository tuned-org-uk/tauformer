// benches/attention.rs

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

use burn::tensor::{Int, Tensor, backend::Backend};

use tauformer::causalattention::GptModel;
use tauformer::config::NanoChatConfig;
use tauformer::taumode::{TauModeConfig, causal_softmax_over_keys, taumode_distance_logits};

type B = burn_cpu::Cpu<f32>;
type Dev = <B as Backend>::Device;

fn lcg_u32(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    *state
}

fn make_tokens(device: &Dev, b: usize, t: usize, vocab: usize, seed: u32) -> Tensor<B, 2, Int> {
    let mut st = seed;
    let mut ids = Vec::with_capacity(b * t);
    for _ in 0..(b * t) {
        let r = lcg_u32(&mut st) as usize;
        ids.push((r % vocab) as i64);
    }
    Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([b, t])
}

/// Bench end-to-end forward of the baseline dot-product GPT model.
fn bench_gpt_forward(c: &mut Criterion) {
    let device = Dev::default();

    // Keep sizes moderate so benches run quickly on CPU, but are large enough to
    // catch accidental O(T^3) etc.
    let cases = [("t=64", 64usize), ("t=128", 128usize), ("t=256", 256usize)];

    let mut group = c.benchmark_group("gpt_forward");
    for (name, t) in cases {
        let cfg = NanoChatConfig {
            sequence_len: t,
            block_size: t,
            vocab_size: 128,
            n_layer: 4,
            n_head: 4,
            n_kv_head: 4,
            n_embd: 128,
            dropout: 0.0,
        };

        let model = GptModel::<B>::new(&cfg, &device);
        let idx = make_tokens(&device, 1, t, cfg.vocab_size, 123);

        group.throughput(Throughput::Elements((t * cfg.n_embd) as u64));
        group.bench_with_input(BenchmarkId::new("forward", name), &t, |bencher, _| {
            bencher.iter(|| {
                let logits = model.forward(black_box(idx.clone()), black_box(false));
                black_box(logits)
            })
        });
    }
    group.finish();
}

/// Bench the TauMode “kernel” (distance logits + causal softmax + weighted sum).
///
/// This is useful even before TauGPT is fully benchmarkable end-to-end and is stable across refactors. [file:15]
fn bench_taumode_kernel(c: &mut Criterion) {
    let device = Dev::default();

    let cases = [
        ("tk=64", 64usize),
        ("tk=128", 128usize),
        ("tk=256", 256usize),
    ];

    let cfg = TauModeConfig {
        tau: 1.0,
        eps: 1e-6,
        temperature: 1.0,
    };

    let mut group = c.benchmark_group("taumode_kernel");

    for (name, tk) in cases {
        let (b, h, tq, v) = (1usize, 4usize, 1usize, 128usize);

        // lambda_q: [B,H,1]
        let mut lq = vec![0.0f32; b * h * tq];
        for i in 0..lq.len() {
            lq[i] = (i as f32) * 0.01;
        }
        let lambda_q = Tensor::<B, 1>::from_floats(lq.as_slice(), &device).reshape([b, h, tq]);

        // lambda_k: [B,H,Tk]
        let mut lk = vec![0.0f32; b * h * tk];
        for i in 0..lk.len() {
            lk[i] = (i as f32) * 0.001;
        }
        let lambda_k = Tensor::<B, 1>::from_floats(lk.as_slice(), &device).reshape([b, h, tk]);

        // V: [B,H,Tk,V]
        let mut vv = vec![0.0f32; b * h * tk * v];
        // simple deterministic fill to avoid allocator effects
        for i in 0..vv.len() {
            vv[i] = ((i % 97) as f32) * 0.001;
        }
        let v_t = Tensor::<B, 1>::from_floats(vv.as_slice(), &device).reshape([b, h, tk, v]);

        group.throughput(Throughput::Elements((tk * h) as u64));
        group.bench_with_input(BenchmarkId::new("kernel", name), &tk, |bencher, _| {
            bencher.iter(|| {
                let logits = taumode_distance_logits(
                    black_box(lambda_q.clone()),
                    black_box(lambda_k.clone()),
                    black_box(&cfg),
                );
                let att = causal_softmax_over_keys(black_box(logits), tq, tk, h);
                let out = att.matmul(black_box(v_t.clone())); // [B,H,1,V]
                black_box(out)
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_gpt_forward, bench_taumode_kernel);
criterion_main!(benches);
