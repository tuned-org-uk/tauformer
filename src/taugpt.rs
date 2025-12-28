//! TauGPT: GPT-like transformer that uses TauModeAttention (lambda-distance attention)
//! instead of dot-product causal attention.
//!
//! Mirrors the structure of `gpt.rs` (prenorm + RoPE cache + decode cache),
//! but the per-layer cache stores (V, lambda_k) as defined by TauModeAttention.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Int, Tensor, activation, backend::Backend},
};
use log::{debug, info};

use crate::config::NanoChatConfig;
use crate::gpt::rms_norm;
use crate::rope::precompute_rotary_embeddings;
use crate::tauattention::{TauCacheLayer, TauModeAttention};

/// Tau KV cache: per-layer cache stores (V, lambda_k), plus the current decode position.
#[derive(Debug)]
pub struct TauKVCache<B: Backend> {
    pub store: Vec<TauCacheLayer<B>>,
    pub position: usize,
}

impl<B: Backend> TauKVCache<B> {
    pub fn new(nlayer: usize) -> Self {
        Self {
            store: (0..nlayer).map(|_| None).collect(),
            position: 0,
        }
    }

    pub fn reset(&mut self) {
        for layer in self.store.iter_mut() {
            *layer = None;
        }
        self.position = 0;
    }
}

/// Simple MLP (same shape conventions as GPT MLP: C -> 4C -> C).
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    c_fc: Linear<B>,
    c_proj: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn new(cfg: &NanoChatConfig, device: &B::Device) -> Self {
        let n = cfg.n_embd;
        let init = burn::nn::Initializer::KaimingUniform {
            gain: 0.5,
            fan_out_only: false,
        };

        Self {
            c_fc: LinearConfig::new(n, 4 * n)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            c_proj: LinearConfig::new(4 * n, n)
                .with_bias(false)
                .with_initializer(init)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Keep it identical to the style in `gpt.rs` (ReLU then squared).
        let x = self.c_fc.forward(x);
        let x = activation::relu(x).powf_scalar(2.0);
        self.c_proj.forward(x)
    }
}

/// One transformer block using taumode attention.
#[derive(Module, Debug)]
pub struct TauBlock<B: Backend> {
    layer_idx: usize,
    ln1: LayerNorm<B>,
    attn: TauModeAttention<B>,
    ln2: LayerNorm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> TauBlock<B> {
    pub fn new_with_laplacian(
        cfg: &NanoChatConfig,
        layer_idx: usize,
        device: &B::Device,
        laplacian_dd: Tensor<B, 2>, // [head_dim, head_dim]
    ) -> Self {
        Self {
            layer_idx,
            ln1: LayerNormConfig::new(cfg.n_embd).init(device),
            attn: TauModeAttention::new_with_laplacian(cfg, layer_idx, device, laplacian_dd),
            ln2: LayerNormConfig::new(cfg.n_embd).init(device),
            mlp: Mlp::new(cfg, device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos_sin: (&Tensor<B, 4>, &Tensor<B, 4>),
    ) -> Tensor<B, 3> {
        // Prenorm + residual
        let x = x.clone() + self.attn.forward(rms_norm(x.clone(), 1e-6), cos_sin);
        // Prenorm + residual
        let x = x.clone() + self.mlp.forward(rms_norm(x, 1e-6));
        x
    }

    pub fn forward_decode(
        &self,
        x_step: Tensor<B, 3>,                         // [B,1,C]
        cos_sin_step: (&Tensor<B, 4>, &Tensor<B, 4>), // [1,1,1,D/2]
        cache_layer: &mut TauCacheLayer<B>,
    ) -> Tensor<B, 3> {
        debug!(
            "TauBlock {} decode: x_step {:?}",
            self.layer_idx,
            x_step.dims()
        );
        let x = x_step.clone()
            + self
                .attn
                .forward_decode(rms_norm(x_step.clone(), 1e-6), cos_sin_step, cache_layer);
        let x = x.clone() + self.mlp.forward(rms_norm(x, 1e-6));
        x
    }
}

/// TauGPT model: same top-level interface style as GPT but attention is TauModeAttention.
#[derive(Module, Debug)]
pub struct TauGptModel<B: Backend> {
    wte: Embedding<B>,
    blocks: Vec<TauBlock<B>>,
    lnf: LayerNorm<B>,
    lmhead: Linear<B>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
    nembd: usize,
    nlayer: usize,
}

impl<B: Backend> TauGptModel<B> {
    /// Build TauGPT with an externally provided Laplacian (dense [head_dim, head_dim]).
    pub fn new_with_laplacian(
        cfg: &NanoChatConfig,
        device: &B::Device,
        laplacian_dd: Tensor<B, 2>,
    ) -> Self {
        info!("Initializing TauGptModel");
        let head_dim = cfg.n_embd / cfg.n_head;
        let (cos, sin) =
            precompute_rotary_embeddings(cfg.sequence_len + 1, head_dim, 10000.0, device);

        let wte = EmbeddingConfig::new(cfg.vocab_size, cfg.n_embd).init(device);

        let blocks = (0..cfg.n_layer)
            .map(|i| TauBlock::new_with_laplacian(cfg, i, device, laplacian_dd.clone()))
            .collect::<Vec<_>>();

        let lnf = LayerNormConfig::new(cfg.n_embd).init(device);

        let init = burn::nn::Initializer::KaimingUniform {
            gain: 0.5,
            fan_out_only: false,
        };
        let lmhead = LinearConfig::new(cfg.n_embd, cfg.vocab_size)
            .with_bias(false)
            .with_initializer(init)
            .init(device);

        Self {
            wte,
            blocks,
            lnf,
            lmhead,
            cos,
            sin,
            nembd: cfg.n_embd,
            nlayer: cfg.n_layer,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.nlayer
    }

    /// Full forward (prefill): idx [B,T] -> logits [B,T,V].
    pub fn forward(&self, idx: Tensor<B, 2, Int>, use_softcap: bool) -> Tensor<B, 3> {
        let [b, t] = idx.dims();
        debug!("TauGptModel.forward: B={}, T={}", b, t);

        let mut x = self.wte.forward(idx).reshape([b, t, self.nembd]);

        for block in self.blocks.iter() {
            x = block.forward(x, (&self.cos, &self.sin));
        }

        let x = rms_norm(x, 1e-6);
        let mut logits = self.lmhead.forward(x);

        // Keep the same safety clamps/softcap pattern style as `gpt.rs`.
        logits = logits.clamp(-50.0, 50.0);
        if use_softcap {
            let softcap = 15.0;
            logits = logits
                .clone()
                .div_scalar(softcap)
                .tanh()
                .mul_scalar(softcap);
            logits = logits.clamp(-50.0, 50.0);
        }
        logits
    }

    pub fn forward_no_softcap(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward(idx, false)
    }

    /// Decode one step: last_ids [B,1] -> logits [B,1,V], updates `cache`.
    pub fn forward_decode(
        &self,
        last_ids: Tensor<B, 2, Int>, // [B,1]
        cache: &mut TauKVCache<B>,
        use_softcap: bool,
    ) -> Tensor<B, 3> {
        let [b, tq] = last_ids.dims();
        debug_assert_eq!(tq, 1);

        let tpos = cache.position;
        let d2 = self.cos.dims()[3];

        // Slice RoPE for the current absolute position: [1,1,1,D/2]
        let cos_step = self.cos.clone().slice([0..1, tpos..tpos + 1, 0..1, 0..d2]);
        let sin_step = self.sin.clone().slice([0..1, tpos..tpos + 1, 0..1, 0..d2]);

        // Embed last token -> [B,1,C]
        let mut x = self.wte.forward(last_ids).reshape([b, 1, self.nembd]);

        for (i, block) in self.blocks.iter().enumerate() {
            let layer_cache = &mut cache.store[i];
            x = block.forward_decode(x, (&cos_step, &sin_step), layer_cache);
        }

        let x = rms_norm(x, 1e-6);
        let mut logits = self.lmhead.forward(x);

        logits = logits.clamp(-50.0, 50.0);
        if use_softcap {
            let softcap = 15.0;
            logits = logits
                .clone()
                .div_scalar(softcap)
                .tanh()
                .mul_scalar(softcap);
            logits = logits.clamp(-50.0, 50.0);
        }

        cache.position += 1;
        logits
    }

    /// Greedy generation using decode cache (fast path).
    pub fn generate(
        &self,
        mut idx: Tensor<B, 2, Int>, // [B,T]
        max_new_tokens: usize,
    ) -> Tensor<B, 2, Int> {
        let [b, t0] = idx.dims();
        debug!(
            "TauGptModel.generate: B={}, T0={}, max_new={}",
            b, t0, max_new_tokens
        );

        let mut cache = TauKVCache::<B>::new(self.num_layers());
        cache.reset();

        // Prime the cache by running decode for each token in the prompt except the last,
        // then start generating from the last token.
        // (Minimal + deterministic; can be optimized later.)
        if t0 > 0 {
            for pos in 0..(t0 - 1) {
                let last = idx.clone().slice([0..b, pos..pos + 1]);
                let _ = self.forward_decode(last, &mut cache, true);
            }
        }

        for _ in 0..max_new_tokens {
            let [_, t] = idx.dims();
            let last = idx.clone().slice([0..b, (t - 1)..t]); // [B,1]
            let logits = self.forward_decode(last, &mut cache, true); // [B,1,V]
            let v = logits.dims()[2];
            let last_logits = logits.reshape([b, v]); // [B,V]
            let next = last_logits.argmax(1).reshape([b, 1]);
            idx = Tensor::cat(vec![idx, next], 1);
        }

        idx
    }
}
