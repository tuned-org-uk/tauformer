//! Taumode attention: uses feature-space Laplacian to compress tokens into scalar lambdas,
//! then scores via lambda-distance instead of QK^T.

use burn::{
    module::{Module, Param},
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Bool, Tensor, activation},
};
use log::{debug, info, trace};

use crate::config::NanoChatConfig;
use crate::gpt::rms_norm;
use crate::rope::{apply_rotary_emb, apply_rotary_emb_step};
use crate::taumode::{
    FeatureLaplacian, TauModeConfig, lambdas_from_heads, mqa_expand_heads_4,
    taumode_distance_logits,
};

/// taumode attention layer cache: stores (V, lambda_k) only.
pub type TauCacheLayer<B> = Option<(Tensor<B, 4>, Tensor<B, 3>)>;

/// taumode attention module.
#[derive(Module, Debug)]
pub struct TauModeAttention<B: Backend> {
    pub(crate) layer_idx: usize,
    pub(crate) nhead: usize,
    pub(crate) nkv_head: usize,
    pub(crate) head_dim: usize,

    // Projections
    c_q: Linear<B>,
    c_k: Linear<B>,
    c_v: Linear<B>,
    c_proj: Linear<B>,

    // QK normalization
    qk_norm_q: LayerNorm<B>,
    qk_norm_k: LayerNorm<B>,

    // Store Laplacian matrix directly as a parameter (will be saved/loaded)
    laplacian_matrix: Param<Tensor<B, 2>>,

    // Store tau config as plain fields (not Module members)
    pub(crate) tau: f32,
    pub(crate) eps: f32,
    pub(crate) temperature: f32,
}

impl<B: Backend> TauModeAttention<B> {
    pub fn new(config: &NanoChatConfig, layer_idx: usize, device: &B::Device) -> Self {
        let nhead = config.n_head;
        let nkv_head = config.n_kv_head;
        let nembd = config.n_embd;
        let head_dim = nembd / nhead;

        assert_eq!(nembd % nhead, 0, "nembd must be divisible by nhead");
        assert!(
            nkv_head <= nhead && nhead % nkv_head == 0,
            "Invalid MQA config"
        );

        info!(
            "Layer {} TauAttn: nhead={}, nkv_head={}, head_dim={}",
            layer_idx, nhead, nkv_head, head_dim
        );

        let init = burn::nn::Initializer::KaimingUniform {
            gain: 0.5,
            fan_out_only: false,
        };

        // Build feature Laplacian
        let laplacian = crate::taumode::laplacian_chain_dense::<B>(head_dim, device);
        let tau_config = TauModeConfig::default();

        info!(
            "Layer {} TauAttn: Laplacian dim={}, tau={}, temperature={}",
            layer_idx,
            laplacian.dim(),
            tau_config.tau,
            tau_config.temperature
        );

        Self {
            layer_idx,
            nhead,
            nkv_head,
            head_dim,
            c_q: LinearConfig::new(nembd, nhead * head_dim)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            c_k: LinearConfig::new(nembd, nkv_head * head_dim)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            c_v: LinearConfig::new(nembd, nkv_head * head_dim)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            c_proj: LinearConfig::new(nembd, nembd)
                .with_bias(false)
                .with_initializer(init)
                .init(device),
            qk_norm_q: LayerNormConfig::new(head_dim).init(device),
            qk_norm_k: LayerNormConfig::new(head_dim).init(device),
            // Store as Param so it gets saved/loaded but not trained
            laplacian_matrix: Param::from_tensor(laplacian.matrix).set_require_grad(false),
            // Store config scalars directly
            tau: tau_config.tau,
            eps: tau_config.eps,
            temperature: tau_config.temperature,
        }
    }

    pub fn new_with_laplacian(
        config: &NanoChatConfig,
        layer_idx: usize,
        device: &B::Device,
        laplacian_dd: Tensor<B, 2>, // [head_dim, head_dim]
    ) -> Self {
        let mut this = Self::new(config, layer_idx, device);
        this.laplacian_matrix = Param::from_tensor(laplacian_dd).set_require_grad(false);
        this
    }

    // Helper to reconstruct FeatureLaplacian on the fly
    pub fn get_laplacian(&self) -> FeatureLaplacian<B> {
        FeatureLaplacian {
            matrix: self.laplacian_matrix.val(),
        }
    }

    // Helper to reconstruct TauModeConfig on the fly
    pub fn get_tau_config(&self) -> TauModeConfig {
        TauModeConfig {
            tau: self.tau,
            eps: self.eps,
            temperature: self.temperature,
        }
    }

    /// Forward pass (training/prefill): full sequence.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos_sin: (&Tensor<B, 4>, &Tensor<B, 4>),
    ) -> Tensor<B, 3> {
        let [b, t, c] = x.dims();
        debug!(
            "Layer {} TauAttn forward: input (B, T, C) = ({}, {}, {})",
            self.layer_idx, b, t, c
        );

        // Project Q, K, V
        let q =
            self.c_q
                .forward(x.clone())
                .clamp(-5.0, 5.0)
                .reshape([b, t, self.nhead, self.head_dim]);
        let k = self.c_k.forward(x.clone()).clamp(-5.0, 5.0).reshape([
            b,
            t,
            self.nkv_head,
            self.head_dim,
        ]);
        let v = self
            .c_v
            .forward(x)
            .clamp(-5.0, 5.0)
            .reshape([b, t, self.nkv_head, self.head_dim]);

        // (B, T, H, D) → (B, H, T, D)
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Apply RoPE
        let (cos, sin) = cos_sin;
        let q = apply_rotary_emb(q, cos, sin);
        let k = apply_rotary_emb(k, cos, sin);

        // QK normalization
        let q = rms_norm(q, 1e-6);
        let k = rms_norm(k, 1e-6);

        debug!(
            "Layer {} TauAttn: after RoPE+norm Q={:?}, K={:?}, V={:?}",
            self.layer_idx,
            q.dims(),
            k.dims(),
            v.dims()
        );

        // taumode attention
        let y = self.scaled_tau_attention(q, k, v, t, t);

        // (B, H, T, D) → (B, T, C)
        let y = y.swap_dims(1, 2).reshape([b, t, c]);
        self.c_proj.forward(y)
    }

    /// Decode-time forward
    pub fn forward_decode(
        &self,
        x_step: Tensor<B, 3>,
        cos_sin_step: (&Tensor<B, 4>, &Tensor<B, 4>), // [1,1,1,D/2]
        cache_layer: &mut TauCacheLayer<B>,
    ) -> Tensor<B, 3> {
        let [b, tq, c] = x_step.dims();
        debug_assert_eq!(tq, 1);

        // Project Q, K, V
        let q = self
            .c_q
            .forward(x_step.clone())
            .clamp(-5.0, 5.0)
            .reshape([b, 1, self.nhead, self.head_dim])
            .swap_dims(1, 2); // [B, Hq, 1, D]

        let k_new = self
            .c_k
            .forward(x_step.clone())
            .clamp(-5.0, 5.0)
            .reshape([b, 1, self.nkv_head, self.head_dim])
            .swap_dims(1, 2); // [B, Hkv, 1, D]

        let v_new = self
            .c_v
            .forward(x_step)
            .clamp(-5.0, 5.0)
            .reshape([b, 1, self.nkv_head, self.head_dim])
            .swap_dims(1, 2); // [B, Hkv, 1, D]

        // Apply RoPE for *this* absolute position (stepwise)
        let (cos_step, sin_step) = cos_sin_step;
        let q = apply_rotary_emb_step(q, cos_step, sin_step);
        let k_new = apply_rotary_emb_step(k_new, cos_step, sin_step);

        // QK norm (must match forward path)
        let q = rms_norm(q, 1e-6);
        let k_new = rms_norm(k_new, 1e-6);

        // Compute lambda_k for this step
        let laplacian = self.get_laplacian();
        let tau_config = self.get_tau_config();
        let lambda_k_new = lambdas_from_heads(k_new, &laplacian, &tau_config); // [B, Hkv, 1]

        // Cache management
        let (v_full, lambda_k_full) = match cache_layer.take() {
            Some((v_all, lk_all)) => (
                Tensor::cat(vec![v_all, v_new.clone()], 2), // time axis
                Tensor::cat(vec![lk_all, lambda_k_new.clone()], 2), // time axis
            ),
            None => (v_new.clone(), lambda_k_new.clone()),
        };

        *cache_layer = Some((v_full.clone(), lambda_k_full.clone()));

        let tk = v_full.dims()[2];
        let y = self.scaled_tau_attention_decode(q, lambda_k_full, v_full, 1, tk);

        let y = y.swap_dims(1, 2).reshape([b, 1, c]);
        self.c_proj.forward(y)
    }

    fn scaled_tau_attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        tq: usize,
        tk: usize,
    ) -> Tensor<B, 4> {
        let [b, _hq, _, _d] = q.dims();

        // MQA expansion
        let (k, v) = if self.nhead != self.nkv_head {
            (
                mqa_expand_heads_4(k, self.nhead, self.nkv_head),
                mqa_expand_heads_4(v, self.nhead, self.nkv_head),
            )
        } else {
            (k, v)
        };

        // Compute lambda scalars
        let laplacian = self.get_laplacian();
        let tau_config = self.get_tau_config();
        let lambda_q = lambdas_from_heads(q, &laplacian, &tau_config);
        let lambda_k = lambdas_from_heads(k, &laplacian, &tau_config);

        // Build attention logits
        let mut att = taumode_distance_logits(lambda_q, lambda_k, &tau_config);

        // Causal mask
        // Align queries to the end of the key cache:
        // - forward: tq == tk => diag = 0
        // - decode style: tq == 1, tk == T => diag = T - 1
        let diag = (tk as i64) - (tq as i64);
        let mask_2d: Tensor<B, 2, Bool> = Tensor::tril_mask([tq, tk], diag, &att.device());

        let mask_4d = mask_2d
            .unsqueeze_dims::<4>(&[0, 1])
            .expand([b, self.nhead, tq, tk]);
        att = att.mask_fill(mask_4d, -1.0e9);

        // Stable softmax
        let att_max = att.clone().max_dim(3).squeeze::<3>(3);
        att = att - att_max.unsqueeze_dim::<4>(3);
        let att = activation::softmax(att, 3);

        // Weighted sum
        att.matmul(v)
    }

    fn scaled_tau_attention_decode(
        &self,
        q: Tensor<B, 4>,
        lambda_k: Tensor<B, 3>,
        v: Tensor<B, 4>,
        tq: usize,
        tk: usize,
    ) -> Tensor<B, 4> {
        let [b, _hq, _, _d] = q.dims();

        // MQA expansion
        let (lambda_k, v) = if self.nhead != self.nkv_head {
            (
                crate::taumode::mqa_expand_heads_3(lambda_k, self.nhead, self.nkv_head),
                mqa_expand_heads_4(v, self.nhead, self.nkv_head),
            )
        } else {
            (lambda_k, v)
        };

        // Compute lambda_q
        let laplacian = self.get_laplacian();
        let tau_config = self.get_tau_config();
        let lambda_q = lambdas_from_heads(q, &laplacian, &tau_config);

        // Build attention logits
        let mut att = taumode_distance_logits(lambda_q, lambda_k, &tau_config);

        // Causal mask + stable softmax
        let diag = (tk as i64) - (tq as i64);
        let mask_2d: Tensor<B, 2, Bool> = Tensor::tril_mask([tq, tk], diag, &att.device());

        let mask_4d = mask_2d
            .unsqueeze_dims::<4>(&[0, 1])
            .expand([b, self.nhead, tq, tk]);
        att = att.mask_fill(mask_4d, -1.0e9);

        let att_max = att.clone().max_dim(3).squeeze::<3>(3);
        att = att - att_max.unsqueeze_dim::<4>(3);
        let att = activation::softmax(att, 3);

        // Weighted sum
        att.matmul(v)
    }
}
