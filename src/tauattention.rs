//! Taumode attention: uses feature-space Laplacian to compress tokens into scalar lambdas,
//! then scores via lambda-distance instead of softmax(QK).

use burn::{
    module::{Ignored, Module, Param},
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Bool, Tensor, activation},
};
use log::debug;

use crate::config::NanoChatConfig;
use crate::rope::{apply_rotary_emb, apply_rotary_emb_step};
use crate::taumode::{TauModeConfig, mqa_expand_heads_4, taumode_distance_logits};
use crate::{causalattention::rms_norm, pretraining::parquet::TauMode};
use sprs::CsMat;

/// taumode attention layer cache: stores (V, lambda_k) only.
pub type TauCacheLayer<B> = Option<(Tensor<B, 4>, Tensor<B, 3>)>;

/// taumode attention module.
#[derive(Module, Debug)]
pub struct TauModeAttention<B: Backend> {
    manifold_dim: usize,

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
    laplacian_tensor: Option<Param<Tensor<B, 2>>>, // using dense
    laplacian_matrix: Ignored<Option<CsMat<f32>>>, // using sparse

    pub(crate) tau_mode: Ignored<Option<TauMode>>,
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
            nkv_head <= nhead && nhead.is_multiple_of(nkv_head),
            "Invalid MQA config"
        );

        debug!(
            "Layer {} TauAttn: nhead={}, nkv_head={}, head_dim={}",
            layer_idx, nhead, nkv_head, head_dim
        );
        debug!(
            "Layer {} TauAttn: config n_embd={}, n_head={}, n_kv_head={}, computed head_dim={}",
            layer_idx, nembd, nhead, nkv_head, head_dim
        );

        let init = burn::nn::Initializer::KaimingUniform {
            gain: 0.5,
            fan_out_only: false,
        };

        // Build feature Laplacian
        let laplacian = crate::taumode::laplacian_chain_dense::<B>(head_dim, device);
        let tau_config = TauModeConfig::default();

        debug!(
            "Layer {} TauAttn: Laplacian dim={}, tau={}, temperature={}",
            layer_idx,
            laplacian.dim(),
            tau_config.tau,
            tau_config.temperature
        );
        debug!(
            "Layer {} TauAttn: dense laplacian tensor dims={:?}",
            layer_idx,
            laplacian.matrix.dims()
        );

        Self {
            manifold_dim: head_dim,
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
            laplacian_tensor: Some(Param::from_tensor(laplacian.matrix).set_require_grad(false)),
            laplacian_matrix: Ignored(None),
            tau_mode: Ignored(Some(TauMode::Median)),
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
        laplacian: &CsMat<f32>,
        tau_mode: TauMode,
    ) -> Self {
        assert_eq!(
            laplacian.rows(),
            laplacian.cols(),
            "manifold Laplacian must be square"
        );
        debug!(
            "Layer {} TauAttn: new_with_laplacian using sparse laplacian rows={} cols={} nnz={} tau_mode={:?}",
            layer_idx,
            laplacian.rows(),
            laplacian.cols(),
            laplacian.nnz(),
            tau_mode
        );

        let mut this = Self::new(config, layer_idx, device);

        debug!(
            "Layer {} TauAttn: switching from dense laplacian to sparse laplacian (dense Param will be None)",
            layer_idx
        );
        this.manifold_dim = laplacian.rows();
        this.laplacian_matrix = Ignored(Some(laplacian.clone()));
        this.laplacian_tensor = None;
        this.tau_mode = Ignored(Some(tau_mode));

        debug!(
            "Layer {} TauAttn: laplacian_matrix set? {} laplacian_tensor set? {}",
            layer_idx,
            this.laplacian_matrix.0.is_some(),
            this.laplacian_tensor.is_some()
        );

        this
    }

    // Helper to access the matrix if set
    pub fn get_laplacian_matrix(&self) -> &CsMat<f32> {
        debug!(
            "Layer {} TauAttn: get_laplacian_matrix called (is_some={})",
            self.layer_idx,
            self.laplacian_matrix.0.is_some()
        );

        if self.laplacian_matrix.0.as_ref().is_none() {
            panic!("Called for laplacian matrix sparse version but it is not set")
        }

        let lap = self.laplacian_matrix.0.as_ref().unwrap();
        debug!(
            "Layer {} TauAttn: sparse laplacian dims=({},{}) nnz={}",
            self.layer_idx,
            lap.rows(),
            lap.cols(),
            lap.nnz()
        );

        lap
    }

    // Helper to access the tensor if set
    pub fn get_laplacian_tensor(&self) -> &Param<Tensor<B, 2>> {
        debug!(
            "Layer {} TauAttn: get_laplacian_tensor called (is_some={})",
            self.layer_idx,
            self.laplacian_tensor.is_some()
        );

        if self.laplacian_tensor.as_ref().is_none() {
            panic!("Called for laplacian matrix sparse version but it is not set")
        }

        let lap = self.laplacian_tensor.as_ref().unwrap();
        debug!(
            "Layer {} TauAttn: dense laplacian dims={:?}",
            self.layer_idx,
            lap.val().dims()
        );

        lap
    }

    // Helper to reconstruct TauModeConfig on the fly
    pub fn get_tau_config(&self) -> TauModeConfig {
        debug!(
            "Layer {} TauAttn: get_tau_config tau={} eps={} temperature={}",
            self.layer_idx, self.tau, self.eps, self.temperature
        );

        TauModeConfig {
            tau: self.tau,
            eps: self.eps,
            temperature: self.temperature,
        }
    }

    fn lambdas_from_heads_any(&self, heads: Tensor<B, 4>) -> Tensor<B, 3> {
        let tau_cfg = self.get_tau_config();
        let [b, h, t, d] = heads.dims();

        debug!(
            "Layer {} TauAttn: lambdas_from_heads_any heads dims [B={},H={},T={},D={}]",
            self.layer_idx, b, h, t, d
        );

        if let Some(lap) = self.laplacian_matrix.0.as_ref() {
            let mode = self
                .tau_mode
                .0
                .unwrap_or(crate::pretraining::parquet::TauMode::Median);

            debug!(
                "Layer {} TauAttn: computing lambdas via SPARSE laplacian rows={} cols={} nnz={} mode={:?} eps={}",
                self.layer_idx,
                lap.rows(),
                lap.cols(),
                lap.nnz(),
                mode,
                tau_cfg.eps
            );

            crate::taumode::lambdas_from_heads_sparse::<B>(heads, lap, mode, tau_cfg.eps)
        } else {
            let lap = self.get_laplacian_tensor().clone();
            let [lr, lc] = lap.val().dims();

            debug!(
                "Layer {} TauAttn: computing lambdas via DENSE laplacian dims=({},{}) tau={} eps={} temperature={}",
                self.layer_idx, lr, lc, tau_cfg.tau, tau_cfg.eps, tau_cfg.temperature
            );

            crate::taumode::lambdas_from_heads::<B>(heads, lap, &tau_cfg)
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
        debug!(
            "Layer {} TauAttn forward: cache-less full attention path, tau_mode={:?}, using_sparse_lap={}",
            self.layer_idx,
            self.tau_mode.0,
            self.laplacian_matrix.0.is_some()
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

        debug!(
            "Layer {} TauAttn forward: projected Q/K/V (pre-swap) dims Q={:?}, K={:?}, V={:?}",
            self.layer_idx,
            q.dims(),
            k.dims(),
            v.dims()
        );

        // (B, T, H, D) → (B, H, T, D)
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        debug!(
            "Layer {} TauAttn forward: after swap dims Q={:?}, K={:?}, V={:?}",
            self.layer_idx,
            q.dims(),
            k.dims(),
            v.dims()
        );

        // Apply RoPE
        let (cos, sin) = cos_sin;
        debug!(
            "Layer {} TauAttn forward: RoPE cos dims={:?} sin dims={:?}",
            self.layer_idx,
            cos.dims(),
            sin.dims()
        );

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

        debug!(
            "Layer {} TauAttn forward: scaled_tau_attention output dims={:?}",
            self.layer_idx,
            y.dims()
        );

        // (B, H, T, D) → (B, T, C)
        let y = y.swap_dims(1, 2).reshape([b, t, c]);
        debug!(
            "Layer {} TauAttn forward: output reshaped to (B,T,C) dims={:?}",
            self.layer_idx,
            y.dims()
        );

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

        debug!(
            "Layer {} TauAttn decode: x_step dims [B={},Tq={},C={}] cache_present={}",
            self.layer_idx,
            b,
            tq,
            c,
            cache_layer.is_some()
        );

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

        debug!(
            "Layer {} TauAttn decode: projected step Q={:?}, K_new={:?}, V_new={:?}",
            self.layer_idx,
            q.dims(),
            k_new.dims(),
            v_new.dims()
        );

        // Apply RoPE for *this* absolute position (stepwise)
        let (cos_step, sin_step) = cos_sin_step;
        debug!(
            "Layer {} TauAttn decode: RoPE step cos dims={:?} sin dims={:?}",
            self.layer_idx,
            cos_step.dims(),
            sin_step.dims()
        );

        let q = apply_rotary_emb_step(q, cos_step, sin_step);
        let k_new = apply_rotary_emb_step(k_new, cos_step, sin_step);

        // QK norm (must match forward path)
        let q = rms_norm(q, 1e-6);
        let k_new = rms_norm(k_new, 1e-6);

        debug!(
            "Layer {} TauAttn decode: after RoPE+norm Q={:?}, K_new={:?}",
            self.layer_idx,
            q.dims(),
            k_new.dims()
        );

        // Compute lambda_k for this step
        let lambda_k_new = self.lambdas_from_heads_any(k_new); // [B, Hkv, 1]
        debug!(
            "Layer {} TauAttn decode: lambda_k_new dims={:?}",
            self.layer_idx,
            lambda_k_new.dims()
        );

        // Cache management
        let (v_full, lambda_k_full) = match cache_layer.take() {
            Some((v_all, lk_all)) => {
                let t_prev = v_all.dims()[2];
                debug!(
                    "Layer {} TauAttn decode: appending to cache (prev Tk={} -> new Tk={})",
                    self.layer_idx,
                    t_prev,
                    t_prev + 1
                );
                (
                    Tensor::cat(vec![v_all, v_new.clone()], 2), // time axis
                    Tensor::cat(vec![lk_all, lambda_k_new.clone()], 2), // time axis
                )
            }
            None => {
                debug!(
                    "Layer {} TauAttn decode: initializing cache (Tk=1)",
                    self.layer_idx
                );
                (v_new.clone(), lambda_k_new.clone())
            }
        };

        debug!(
            "Layer {} TauAttn decode: cache tensors dims V_full={:?}, lambda_k_full={:?}",
            self.layer_idx,
            v_full.dims(),
            lambda_k_full.dims()
        );

        *cache_layer = Some((v_full.clone(), lambda_k_full.clone()));

        let tk = v_full.dims()[2];
        debug!(
            "Layer {} TauAttn decode: calling scaled_tau_attention_decode tq=1 tk={}",
            self.layer_idx, tk
        );

        let y = self.scaled_tau_attention_decode(q, lambda_k_full, v_full, 1, tk);

        debug!(
            "Layer {} TauAttn decode: scaled_tau_attention_decode output dims={:?}",
            self.layer_idx,
            y.dims()
        );

        let y = y.swap_dims(1, 2).reshape([b, 1, c]);
        debug!(
            "Layer {} TauAttn decode: output reshaped to (B,1,C) dims={:?}",
            self.layer_idx,
            y.dims()
        );

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

        debug!(
            "Layer {} TauAttn: scaled_tau_attention enter tq={} tk={} Q={:?} K={:?} V={:?}",
            self.layer_idx,
            tq,
            tk,
            q.dims(),
            k.dims(),
            v.dims()
        );

        // Expand only V for the weighted sum (MQA); keep K unexpanded for lambda_k.
        let v = if self.nhead != self.nkv_head {
            debug!(
                "Layer {} TauAttn: MQA expand V (nhead={} nkv_head={})",
                self.layer_idx, self.nhead, self.nkv_head
            );
            mqa_expand_heads_4(v, self.nhead, self.nkv_head)
        } else {
            v
        };

        let tau_config = self.get_tau_config();

        // Compute lambda scalars using the unified policy (prefer sparse if present).
        let lambda_q = self.lambdas_from_heads_any(q); // [B, Hq, Tq,] -> [B, Hq, Tq]
        let lambda_k = self.lambdas_from_heads_any(k); // [B, Hkv, Tk,] -> [B, Hkv, Tk]

        debug!(
            "Layer {} TauAttn: lambdas computed lambda_q={:?}, lambda_k={:?}",
            self.layer_idx,
            lambda_q.dims(),
            lambda_k.dims()
        );

        // If MQA: expand lambda_k (not K) to match nhead.
        let lambda_k = if self.nhead != self.nkv_head {
            debug!(
                "Layer {} TauAttn: MQA expand lambda_k (nhead={} nkv_head={})",
                self.layer_idx, self.nhead, self.nkv_head
            );
            crate::taumode::mqa_expand_heads_3(lambda_k, self.nhead, self.nkv_head)
        } else {
            lambda_k
        };

        debug!(
            "Layer {} TauAttn: lambda_k after possible expand dims={:?}",
            self.layer_idx,
            lambda_k.dims()
        );

        // Build attention logits
        let mut att = taumode_distance_logits(lambda_q, lambda_k, &tau_config);
        debug!(
            "Layer {} TauAttn: att logits dims={:?} (before causal mask)",
            self.layer_idx,
            att.dims()
        );

        // Causal mask
        let diag = (tk as i64) - (tq as i64);
        let mask_2d: Tensor<B, 2, Bool> = Tensor::tril_mask([tq, tk], diag, &att.device());
        let mask_2d_dims = mask_2d.dims(); // used by loggings
        let mask_4d = mask_2d
            .unsqueeze_dims::<4>(&[0, 1])
            .expand([b, self.nhead, tq, tk]);

        debug!(
            "Layer {} TauAttn: mask dims mask_2d={:?} mask_4d={:?} diag={}",
            self.layer_idx,
            mask_2d_dims,
            mask_4d.dims(),
            diag
        );

        att = att.mask_fill(mask_4d, -1.0e9);

        // Stable softmax
        let att_max = att.clone().max_dim(3).flatten::<3>(2, 3);
        att = att - att_max.unsqueeze_dim::<4>(3);
        let att = activation::softmax(att, 3);

        debug!(
            "Layer {} TauAttn: att probs dims={:?} (after softmax)",
            self.layer_idx,
            att.dims()
        );

        // Weighted sum
        let out = att.matmul(v);
        debug!(
            "Layer {} TauAttn: weighted sum output dims={:?}",
            self.layer_idx,
            out.dims()
        );
        out
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

        debug!(
            "Layer {} TauAttn: scaled_tau_attention_decode enter tq={} tk={} Q={:?} lambda_k={:?} V={:?}",
            self.layer_idx,
            tq,
            tk,
            q.dims(),
            lambda_k.dims(),
            v.dims()
        );

        // MQA expansion (decode path receives lambda_k already computed/cached at Hkv)
        let (lambda_k, v) = if self.nhead != self.nkv_head {
            debug!(
                "Layer {} TauAttn decode: MQA expand lambda_k and V (nhead={} nkv_head={})",
                self.layer_idx, self.nhead, self.nkv_head
            );
            (
                crate::taumode::mqa_expand_heads_3(lambda_k, self.nhead, self.nkv_head),
                mqa_expand_heads_4(v, self.nhead, self.nkv_head),
            )
        } else {
            (lambda_k, v)
        };

        debug!(
            "Layer {} TauAttn decode: after MQA expand lambda_k={:?} V={:?}",
            self.layer_idx,
            lambda_k.dims(),
            v.dims()
        );

        // Compute lambda_q via the unified helper (prefer sparse if present)
        let tau_config = self.get_tau_config();
        let lambda_q = self.lambdas_from_heads_any(q);

        debug!(
            "Layer {} TauAttn decode: lambda_q dims={:?} lambda_k dims={:?}",
            self.layer_idx,
            lambda_q.dims(),
            lambda_k.dims()
        );

        // Build attention logits
        let mut att = taumode_distance_logits(lambda_q, lambda_k, &tau_config);
        debug!(
            "Layer {} TauAttn decode: att logits dims={:?} (before causal mask)",
            self.layer_idx,
            att.dims()
        );

        // Causal mask + stable softmax
        let diag = (tk as i64) - (tq as i64);
        let mask_2d: Tensor<B, 2, Bool> = Tensor::tril_mask([tq, tk], diag, &att.device());
        let mask_2d_dims = mask_2d.dims();
        let mask_4d = mask_2d
            .unsqueeze_dims::<4>(&[0, 1])
            .expand([b, self.nhead, tq, tk]);

        debug!(
            "Layer {} TauAttn decode: mask dims mask_2d={:?} mask_4d={:?} diag={}",
            self.layer_idx,
            mask_2d_dims,
            mask_4d.dims(),
            diag
        );

        att = att.mask_fill(mask_4d, -1.0e9);

        let att_max = att.clone().max_dim(3).flatten::<3>(2, 3);
        att = att - att_max.unsqueeze_dim::<4>(3);
        let att = activation::softmax(att, 3);

        debug!(
            "Layer {} TauAttn decode: att probs dims={:?} (after softmax)",
            self.layer_idx,
            att.dims()
        );

        // Weighted sum
        let out = att.matmul(v);
        debug!(
            "Layer {} TauAttn decode: weighted sum output dims={:?}",
            self.layer_idx,
            out.dims()
        );
        out
    }
}
