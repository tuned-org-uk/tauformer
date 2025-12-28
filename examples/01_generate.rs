// examples/generate.rs
use burn::tensor::{Int, Tensor};
use tauformer::{
    backend::AutoBackend as B,
    config::NanoChatConfig,
    gpt::GptModel,
    parquet,
    pretraining,
    // You add a tau-enabled GPT wrapper (see note below).
    // taugpt::TauGptModel,
};

fn main() -> anyhow::Result<()> {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "causal".to_string()); // "causal" | "tau"
    let prompt_ids: Vec<i64> = vec![1, 2, 3]; // keep it simple (replace with tokenizer later)
    let max_new: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "16".to_string())
        .parse()?;

    let device = <B as burn::prelude::Backend>::Device::default();

    let cfg = NanoChatConfig {
        sequence_len: 128,
        vocab_size: 32000,
        n_layer: 4,
        n_head: 4,
        n_kv_head: 2,
        n_embd: 32,
        block_size: 128,
        dropout: 0.0,
    };

    let idx = Tensor::<B, 1, Int>::from_ints(prompt_ids.as_slice(), &device)
        .reshape([1, prompt_ids.len()]);

    if mode == "tau" {
        let manifold_path = std::path::Path::new("./domain_manifold/manifold.parquet");
        pretraining::ensure_domain_manifold_exists(manifold_path)?; // if you already have this logic
        let head_dim = cfg.n_embd / cfg.n_head;

        let lap =
            parquet::load_manifold_laplacian_for_head_dim::<B>(manifold_path, head_dim, &device)?;

        // TODO: build a Tau-enabled GPT (see next bullet).
        // let model = TauGptModel::<B>::new_with_laplacian(&cfg, &device, lap);
        // let out = model.generate_greedy(idx, max_new);

        anyhow::bail!("Tau GPT wrapper not wired yet (needs blocks using TauModeAttention).");
    } else {
        let model = GptModel::<B>::new(&cfg, &device); // existing causal GPT
        let out = model.generate(idx, max_new); // existing generation path
        let out_ids: Vec<i64> = out.to_data().to_vec().unwrap();
        println!("{out_ids:?}");
    }

    Ok(())
}
