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
use log::{debug, info, trace};
use std::path::Path;

use crate::config::NanoChatConfig;
use crate::rope::precompute_rotary_embeddings;
use crate::tauattention::{TauCacheLayer, TauModeAttention};
use crate::{causalattention::rms_norm, pretraining::DomainManifold};
use sprs::CsMat;

use crate::pretraining::parquet::TauMode as ManifoldTauMode;

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
    pub(crate) attn: TauModeAttention<B>,
    ln2: LayerNorm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> TauBlock<B> {
    pub fn new_with_sparse_laplacian(
        cfg: &NanoChatConfig,
        layer_idx: usize,
        device: &B::Device,
        laplacian: &CsMat<f32>,
        tau_mode: crate::pretraining::parquet::TauMode,
    ) -> Self {
        info!("Initializing TauBlock {} (sparse laplacian)", layer_idx);

        Self {
            layer_idx,
            ln1: LayerNormConfig::new(cfg.n_embd).init(device),
            attn: TauModeAttention::new_with_laplacian(cfg, layer_idx, device, laplacian, tau_mode),
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
    pub(crate) blocks: Vec<TauBlock<B>>,
    lnf: LayerNorm<B>,
    lmhead: Linear<B>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
    nembd: usize,
    nlayer: usize,
}

impl<B: Backend> TauGptModel<B> {
    /// Build TauGPT with an externally provided Laplacian (dense [head_dim, head_dim]).
    pub fn new_with_sparse_laplacian(
        cfg: &NanoChatConfig,
        laplacian_path: &Path,
        device: &B::Device,
        tau_mode: ManifoldTauMode, // from manifold.json [file:44]
    ) -> Self {
        info!("Initializing TauGptModel (sparse laplacian)");

        let head_dim = cfg.n_embd / cfg.n_head;
        let domain_manifold: DomainManifold =
            crate::pretraining::load_domain_manifold(laplacian_path).unwrap();

        assert!(domain_manifold.laplacian().rows() == head_dim);
        assert!(
            domain_manifold.laplacian().rows() == domain_manifold.laplacian().cols(),
            "Manifold Laplacian must be square, got {}x{}",
            domain_manifold.laplacian().rows(),
            domain_manifold.laplacian().cols()
        );
        assert!(
            domain_manifold.laplacian().rows() == head_dim,
            "Sparse Laplacian dim {} must match head_dim {}",
            domain_manifold.laplacian().rows(),
            head_dim
        );

        let (cos, sin) =
            precompute_rotary_embeddings(cfg.sequence_len + 1, head_dim, 10000.0, device);
        debug!("RoPE cache cos {:?}, sin {:?}", cos.dims(), sin.dims());

        let wte = EmbeddingConfig::new(cfg.vocab_size, cfg.n_embd).init(device);

        let lnf = LayerNormConfig::new(cfg.n_embd).init(device);

        let init = burn::nn::Initializer::KaimingUniform {
            gain: 0.5,
            fan_out_only: false,
        };
        let lmhead = LinearConfig::new(cfg.n_embd, cfg.vocab_size)
            .with_bias(false)
            .with_initializer(init)
            .init(device);

        info!("TauGptModel (sparse) initialization complete");

        let laplacian = domain_manifold.laplacian();
        Self {
            wte,
            // Each layer shares the same manifold Laplacian + tau selection policy.
            blocks: (0..cfg.n_layer)
                .map(|i| TauBlock::new_with_sparse_laplacian(cfg, i, device, &laplacian, tau_mode))
                .collect::<Vec<_>>(),
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

    pub fn laplacian_dims(&self) -> usize {
        self.blocks[0].clone().attn.get_laplacian_matrix().shape().0
    }
    /// Full forward (prefill): idx [B,T] -> logits [B,T,V].
    pub fn forward(&self, idx: Tensor<B, 2, Int>, use_softcap: bool) -> Tensor<B, 3> {
        let [b, t] = idx.dims();
        debug!(
            "TauGptModel.forward: enter B={}, T={}, n_embd={}, n_layer={}, use_softcap={}",
            b, t, self.nembd, self.nlayer, use_softcap
        );

        debug!("TauGptModel.forward: wte.forward(idx)...");
        let mut x = self.wte.forward(idx);
        debug!("TauGptModel.forward: wte out dims={:?}", x.dims());

        debug!(
            "TauGptModel.forward: reshape embeddings -> [B,T,C]=[{}, {}, {}]",
            b, t, self.nembd
        );
        x = x.reshape([b, t, self.nembd]);
        debug!("TauGptModel.forward: x dims after reshape={:?}", x.dims());

        // RoPE cache dims sanity (useful when mismatched seq_len shows up).
        debug!(
            "TauGptModel.forward: RoPE cache cos dims={:?}, sin dims={:?}",
            self.cos.dims(),
            self.sin.dims()
        );

        for (i, block) in self.blocks.iter().enumerate() {
            debug!("TauGptModel.forward: entering block {}", i);
            trace!(
                "TauGptModel.forward: x dims before block {}: {:?}",
                i,
                x.dims()
            );
            x = block.forward(x, (&self.cos, &self.sin));
            trace!(
                "TauGptModel.forward: x dims after block {}: {:?}",
                i,
                x.dims()
            );
        }

        debug!("TauGptModel.forward: rms_norm...");
        let x = rms_norm(x, 1e-6);
        trace!("TauGptModel.forward: x dims after rms_norm={:?}", x.dims());

        debug!("TauGptModel.forward: lmhead.forward...");
        let mut logits = self.lmhead.forward(x);
        trace!(
            "TauGptModel.forward: logits dims after lmhead={:?}",
            logits.dims()
        );

        debug!("TauGptModel.forward: clamp logits to [-50,50] (pre-softcap)");
        logits = logits.clamp(-50.0, 50.0);

        if use_softcap {
            let softcap = 15.0;
            debug!("TauGptModel.forward: applying softcap={}", softcap);
            logits = logits
                .clone()
                .div_scalar(softcap)
                .tanh()
                .mul_scalar(softcap);
            debug!("TauGptModel.forward: clamp logits to [-50,50] (post-softcap)");
            logits = logits.clamp(-50.0, 50.0);
        } else {
            debug!("TauGptModel.forward: softcap disabled");
        }

        debug!("TauGptModel.forward: exit logits dims={:?}", logits.dims());
        logits
    }

    pub fn forward_no_softcap(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        debug!("TauGptModel.forward_no_softcap: delegating to forward(use_softcap=false)");
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

        debug!(
            "TauGptModel.forward_decode: enter B={}, tq={}, cache.position={}, n_layer={}, n_embd={}, use_softcap={}",
            b, tq, cache.position, self.nlayer, self.nembd, use_softcap
        );

        // Helpful for diagnosing OOB / mismatch between seq_len and decode steps.
        debug!(
            "TauGptModel.forward_decode: RoPE cache cos dims={:?}, sin dims={:?}",
            self.cos.dims(),
            self.sin.dims()
        );

        let tpos = cache.position;
        let d2 = self.cos.dims()[3];
        debug!("TauGptModel.forward_decode: tpos={}, d2={}", tpos, d2);

        // Slice RoPE for the current absolute position: [1,1,1,D/2]
        debug!(
            "TauGptModel.forward_decode: slicing RoPE cos/sin at tpos..tpos+1 = {}..{}",
            tpos,
            tpos + 1
        );
        let cos_step = self.cos.clone().slice([0..1, tpos..tpos + 1, 0..1, 0..d2]);
        let sin_step = self.sin.clone().slice([0..1, tpos..tpos + 1, 0..1, 0..d2]);
        debug!(
            "TauGptModel.forward_decode: cos_step dims={:?}, sin_step dims={:?}",
            cos_step.dims(),
            sin_step.dims()
        );

        // Embed last token -> [B,1,C]
        debug!("TauGptModel.forward_decode: wte.forward(last_ids)...");
        let mut x = self.wte.forward(last_ids);
        debug!("TauGptModel.forward_decode: wte out dims={:?}", x.dims());

        debug!(
            "TauGptModel.forward_decode: reshape embeddings -> [B,1,C]=[{}, 1, {}]",
            b, self.nembd
        );
        x = x.reshape([b, 1, self.nembd]);
        debug!(
            "TauGptModel.forward_decode: x dims after reshape={:?}",
            x.dims()
        );

        for (i, block) in self.blocks.iter().enumerate() {
            debug!(
                "TauGptModel.forward_decode: entering block {} (cache layer present? {})",
                i,
                cache.store[i].is_some()
            );
            debug!(
                "TauGptModel.forward_decode: x dims before block {}: {:?}",
                i,
                x.dims()
            );

            let layer_cache = &mut cache.store[i];
            x = block.forward_decode(x, (&cos_step, &sin_step), layer_cache);

            debug!(
                "TauGptModel.forward_decode: x dims after block {}: {:?}",
                i,
                x.dims()
            );
            debug!(
                "TauGptModel.forward_decode: exiting block {} (cache layer present? {})",
                i,
                layer_cache.is_some()
            );

            // (Optional but very useful when chasing cache growth bugs)
            // NOTE: keep at debug or trace depending on verbosity preference.
            trace!(
                "TauGptModel.forward_decode: cache.position={} after block {} (note: position advanced outside forward_decode)",
                cache.position, i
            );
        }

        debug!("TauGptModel.forward_decode: rms_norm...");
        let x = rms_norm(x, 1e-6);
        debug!(
            "TauGptModel.forward_decode: x dims after rms_norm={:?}",
            x.dims()
        );

        debug!("TauGptModel.forward_decode: lmhead.forward...");
        let mut logits = self.lmhead.forward(x);
        debug!(
            "TauGptModel.forward_decode: logits dims after lmhead={:?}",
            logits.dims()
        );

        debug!("TauGptModel.forward_decode: clamp logits to [-50,50] (pre-softcap)");
        logits = logits.clamp(-50.0, 50.0);

        if use_softcap {
            let softcap = 15.0;
            debug!("TauGptModel.forward_decode: applying softcap={}", softcap);
            logits = logits
                .clone()
                .div_scalar(softcap)
                .tanh()
                .mul_scalar(softcap);
            debug!("TauGptModel.forward_decode: clamp logits to [-50,50] (post-softcap)");
            logits = logits.clamp(-50.0, 50.0);
        } else {
            debug!("TauGptModel.forward_decode: softcap disabled");
        }

        debug!(
            "TauGptModel.forward_decode: exit logits dims={:?}",
            logits.dims()
        );
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
