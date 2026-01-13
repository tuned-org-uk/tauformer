// examples/generate.rs
use burn::tensor::{Int, Tensor};
use std::path::Path;
use tauformer::{
    backend::AutoBackend as B, causalattention::GptModel, config::NanoChatConfig, pretraining,
    taugpt::TauGptModel,
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
        let manifold_parquet = std::path::Path::new("./domain_manifold/manifold.parquet");

        let domain = pretraining::parquet::load_domain_manifold(manifold_parquet)?;
        let head_dim = cfg.n_embd / cfg.n_head;

        anyhow::ensure!(
            head_dim == domain.nfeatures,
            "Tau requires head_dim == manifold nfeatures (got head_dim={}, nfeatures={})",
            head_dim,
            domain.nfeatures
        );

        // Pass CSR + tau_mode into your Tau GPT wiring (TauModeAttention keeps them as ignored fields).
        let model = TauGptModel::<B>::new_with_sparse_laplacian(
            &cfg,
            &Path::new("./domain_manifold/manifold.parquet"),
            &device,
            domain.tau_mode,
        );

        let out = model.generate(idx, max_new);
        let out_ids: Vec<i64> = out.to_data().to_vec().unwrap();
        println!("{out_ids:?}");
    } else {
        let model = GptModel::<B>::new(&cfg, &device); // existing causal GPT
        let out = model.generate(idx, max_new); // existing generation path
        let out_ids: Vec<i64> = out.to_data().to_vec().unwrap();
        println!("{out_ids:?}");
    }

    Ok(())
}
