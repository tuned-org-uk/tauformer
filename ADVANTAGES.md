## Standard causal attention (in this codebase)

### Concept

For one layer, one head, standard causal self-attention does:

1. Project tokens to queries, keys, values:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

with $X\in\mathbb{R}^{B\times T\times C}$.
2. Compute scaled dot-product logits:

$$
A_{ij} = \frac{Q_i\cdot K_j}{\sqrt{d}},\quad i\le j;\quad A_{ij}=-\infty\ \text{if } j>i
$$

then softmax over keys.
3. Aggregate values:

$$
Y_i = \sum_j \text{softmax}(A_{i:})_j V_j
$$

### Where in the code

In `gpt.rs`:

- Projections and shapes:

```rust
// CausalSelfAttention::forward
let q = self.c_q.forward(x.clone()).clamp(-5.0, 5.0).reshape([
    b, t, self.n_head, self.head_dim,
]);
let k = self.c_k.forward(x.clone()).clamp(-5.0, 5.0).reshape([
    b, t, self.n_kv_head, self.head_dim,
]);
let v = self.c_v.forward(x).clamp(-5.0, 5.0).reshape([
    b, t, self.n_kv_head, self.head_dim,
]);
let q = q.swap_dims(1, 2); // [B,H,T,D]
let k = k.swap_dims(1, 2);
let v = v.swap_dims(1, 2);
```

- RoPE application:

```rust
let (cos, sin) = cos_sin;
let q = apply_rotary_emb(q, cos, sin);
let k = apply_rotary_emb(k, cos, sin);
```

- Core attention kernel:

```rust
// scaled_dot_product_attention
let scale = (d as f32).sqrt();
let mut att = q.matmul(k.swap_dims(2, 3)) / scale; // [B,H,Tq,Tk]

// causal mask
let mask2 = Tensor::tril_mask([t_q, t_k], 0, &att.device());
let mask4 = mask2
    .unsqueeze_dims::<4>(&[0, 1])
    .expand([b, self.n_head, t_q, t_k]);
att = att.mask_fill(mask4, -1.0e9);

// stabilize and softmax
let att_max = att.clone().max_dim(3).squeeze::<3>(3);
att = att - att_max.unsqueeze_dim::<4>(3);
let att = activation::softmax(att, 3);

// weighted sum
let out = att.matmul(v); // [B,H,Tq,D]
```

- Decode-time cache: caches **K and V** per layer in `KVCache`, and `forward_decode` appends along time.

***

## Tauformer’s taumode attention

Tauformer keeps the same outer structure (Q/K/V, RoPE, MQA, causal softmax) but **replaces the QK dot-product kernel** with a **λ-distance kernel** built from a graph Laplacian.

### 1. Feature-space Laplacian and Rayleigh energy

Tauformer uses a feature-space Laplacian $L\in\mathbb{R}^{D\times D}$ in the head dimension $D$.

- Dense version (initially a 1D chain Laplacian):

```rust
pub fn laplacian_chain_dense<B: Backend>(
    d: usize,
    device: &B::Device,
) -> FeatureLaplacian<B> {
    let mut data = vec![0.0f32; d * d];
    for i in 0..d {
        let mut deg = 0.0;
        if i > 0 {
            data[i * d + (i - 1)] = -1.0;
            deg += 1.0;
        }
        if i + 1 < d {
            data[i * d + (i + 1)] = -1.0;
            deg += 1.0;
        }
        data[i * d + i] = deg;
    }
    FeatureLaplacian {
        matrix: Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([d, d]),
    }
}
```

- taumode config (τ, ε and temperature):

```rust
#[derive(Clone, Copy, Debug)]
pub struct TauModeConfig {
    pub tau: f32,
    pub eps: f32,
    pub temperature: f32,
}

impl Default for TauModeConfig {
    fn default() -> Self {
        Self {
            tau: 1.0,
            eps: 1.0e-6,
            temperature: 1.0,
        }
    }
}
```


For a head vector $x\in\mathbb{R}^D$, taumode uses a **bounded Rayleigh quotient**:

$$
E_{\text{raw}}(x) = \frac{x^\top L x}{x^\top x + \varepsilon},\quad
E_{\text{raw}} \ge 0
$$

$$
\lambda_\tau(x) = \frac{E_{\text{raw}}}{E_{\text{raw}} + \tau}\in[0,1)
$$

In code, batched over $[B,H,T,D]$:

```rust
pub fn lambdas_from_heads<B: Backend>(
    x: Tensor<B, 4>,          // [B,H,T,D]
    lap: FeatureLaplacian<B>, // dense L [D,D]
    cfg: TauModeConfig,
) -> Tensor<B, 3> {           // [B,H,T]
    let [b, h, t, d] = x.dims();
    debug_assert_eq!(d, lap.dim());

    // Flatten B,H,T,D -> N,D
    let n = b * h * t;
    let x_nd = x.reshape([n, d]);                    // [N,D]
    let y_nd = x_nd.clone().matmul(lap.matrix);      // y = x L

    // numerator: sum_i x_i * (Lx)_i
    let numerator = x_nd.clone().mul(y_nd).sum_dim(1); // [N]

    // denom: sum_i x_i^2 + eps
    let denom = x_nd.powf_scalar(2.0).sum_dim(1) + cfg.eps;

    let e_raw = numerator / denom;                  // [N]
    let e_bounded = e_raw.clone() / (e_raw + cfg.tau);

    e_bounded.reshape([b, h, t])
}
```


### 2. From λ to attention logits

For queries $\lambda^q$ and keys $\lambda^k$ (both in $[0,1)$), tauformer uses a **distance-style kernel** instead of dot-product:

A simple first version (as in your code) is:

$$
\text{att}_{i,j} = -\frac{|\lambda^q_i - \lambda^k_j|}{\text{temperature}}
$$

Code:

```rust
pub fn taumode_distance_logits<B: Backend>(
    lambda_q: Tensor<B, 3>,  // [B,H,Tq]
    lambda_k: Tensor<B, 3>,  // [B,H,Tk]
    cfg: TauModeConfig,
) -> Tensor<B, 4> {          // [B,H,Tq,Tk]
    let [bq, hq, tq] = lambda_q.dims();
    let [bk, hk, tk] = lambda_k.dims();
    debug_assert_eq!(bq, bk);
    debug_assert_eq!(hq, hk);

    let lq = lambda_q.unsqueeze_dim::<4>(3); // [B,H,Tq,1]
    let lk = lambda_k.unsqueeze_dim::<4>(2); // [B,H,1,Tk]

    let temp = cfg.temperature.max(cfg.eps);
    -(lq - lk).abs() / temp
}
```

These logits then go through the same causal mask and numerically stable softmax as standard attention.

***

## 3. TauModeAttention structure and flow

### Module layout

In `tauattention.rs`, taumode attention is a full drop-in replacement for `CausalSelfAttention`:

- It keeps:
    - Q/K/V linear projections
    - RoPE
    - RMS norm
    - MQA expansion
    - causal mask + max-subtraction + softmax
- It changes:
    - The kernel from $QK^\top/\sqrt{d}$ to lambda-distance `taumode_distance_logits` fed from `lambdas_from_heads`.

Relevant struct:

```rust
#[derive(Module, Debug)]
pub struct TauModeAttention<B: Backend> {
    layer_idx: usize,
    nhead: usize,
    nkv_head: usize,
    head_dim: usize,

    // projections
    c_q: Linear<B>,
    c_k: Linear<B>,
    c_v: Linear<B>,
    c_proj: Linear<B>,

    // QK normalization
    qk_norm_q: LayerNorm<B>,
    qk_norm_k: LayerNorm<B>,

    // Laplacian matrix (dense, stored as parameter but not trained)
    laplacian_matrix: Param<Tensor<B, 2>>,

    // tau config scalars
    tau: f32,
    eps: f32,
    temperature: f32,

    // (after wiring manifold): sparse Laplacian fields
    use_sparse_laplacian: bool,
    sparse_laplacian: Option<SparseLaplacianCsr>,
}
```

Constructor (dense Laplacian):

```rust
pub fn new(config: &NanoChatConfig, layer_idx: usize, device: &B::Device) -> Self {
    let nhead = config.n_head;
    let nkv_head = config.n_kv_head;
    let nembd = config.n_embd;
    let head_dim = nembd / nhead;

    let laplacian = crate::taumode::laplacian_chain_dense::<B>(head_dim, device);
    let tau_config = TauModeConfig::default();

    info!(
        "Layer {} TauAttn: Laplacian dim={}, tau={}, temperature={}",
        layer_idx,
        laplacian.dim(),
        tau_config.tau,
        tau_config.temperature,
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

        laplacian_matrix: Param::from_tensor(laplacian.matrix).set_require_grad(false),

        tau: tau_config.tau,
        eps: tau_config.eps,
        temperature: tau_config.temperature,

        use_sparse_laplacian: false,
        sparse_laplacian: None,
    }
}
```

Reconstruction helpers:

```rust
fn get_laplacian(&self) -> FeatureLaplacian<B> {
    FeatureLaplacian {
        matrix: self.laplacian_matrix.val(),
    }
}

fn get_tau_config(&self) -> TauModeConfig {
    TauModeConfig {
        tau: self.tau,
        eps: self.eps,
        temperature: self.temperature,
    }
}
```


### Forward path

Core differences are in `scaled_tau_attention`, called after RoPE + RMS norm:

```rust
pub fn forward(
    &self,
    x: Tensor<B, 3>,               // [B,T,C]
    cos_sin: (&Tensor<B, 4>, &Tensor<B, 4>),
) -> Tensor<B, 3> {
    let [b, t, c] = x.dims();

    // Q,K,V projections
    let q = self.c_q.forward(x.clone()).clamp(-5.0, 5.0)
        .reshape([b, t, self.nhead, self.head_dim]);
    let k = self.c_k.forward(x.clone()).clamp(-5.0, 5.0)
        .reshape([b, t, self.nkv_head, self.head_dim]);
    let v = self.c_v.forward(x).clamp(-5.0, 5.0)
        .reshape([b, t, self.nkv_head, self.head_dim]);

    // [B,H,T,D]
    let q = q.swap_dims(1, 2);
    let k = k.swap_dims(1, 2);
    let v = v.swap_dims(1, 2);

    // RoPE
    let (cos, sin) = cos_sin;
    let q = apply_rotary_emb(q, cos, sin);
    let k = apply_rotary_emb(k, cos, sin);

    // RMS-norm
    let q = rms_norm(q, 1e-6);
    let k = rms_norm(k, 1e-6);

    // taumode attention
    let y = self.scaled_tau_attention(q, k, v, t, t);

    // back to [B,T,C]
    let y = y.swap_dims(1, 2).reshape([b, t, c]);
    self.c_proj.forward(y)
}
```

`scaled_tau_attention`:

```rust
fn scaled_tau_attention(
    &self,
    q: Tensor<B, 4>,  // [B,H,Tq,D]
    k: Tensor<B, 4>,  // [B,H,Tk,D]
    v: Tensor<B, 4>,  // [B,H,Tk,D]
    tq: usize,
    tk: usize,
) -> Tensor<B, 4> {
    let [b, hq, _, d] = q.dims();

    // MQA expansion for K,V
    let (k, v) = if self.nhead != self.nkv_head {
        (
            mqa_expand_heads_4(k, self.nhead, self.nkv_head),
            mqa_expand_heads_4(v, self.nhead, self.nkv_head),
        )
    } else {
        (k, v)
    };

    let tau_config = self.get_tau_config();

    // Compute lambda scalars
    let lambda_q = if self.use_sparse_laplacian {
        lambdas_from_heads_sparse_csr(q, self.sparse_laplacian.as_ref().unwrap(), &tau_config)
    } else {
        let laplacian = self.get_laplacian();
        lambdas_from_heads(q, &laplacian, &tau_config)
    };

    let lambda_k = if self.use_sparse_laplacian {
        lambdas_from_heads_sparse_csr(k, self.sparse_laplacian.as_ref().unwrap(), &tau_config)
    } else {
        let laplacian = self.get_laplacian();
        lambdas_from_heads(k, &laplacian, &tau_config)
    };

    // Build logits from lambda distances
    let mut att = taumode_distance_logits(lambda_q, lambda_k, &tau_config); // [B,H,Tq,Tk]

    // causal mask, max-subtraction, softmax
    let mask2d = Tensor::<B, 2, _>::tril_mask([tq, tk], 0, &att.device());
    let mask4d = mask2d.unsqueeze_dims::<4>(&[0, 1]).expand([b, self.nhead, tq, tk]);
    att = att.mask_fill(mask4d, -1.0e9);
    let att_max = att.clone().max_dim(3).squeeze::<3>(3);
    att = att - att_max.unsqueeze_dim::<4>(3);
    let att = activation::softmax(att, 3);

    // weighted sum
    att.matmul(v)
}
```


### Decode-time caching in tauformer

Key difference: tauformer’s cache stores **values and λ_k**, not full K.

Type alias:

```rust
pub type TauCacheLayer<B> = Option<(Tensor<B, 4>, Tensor<B, 3>)>; // (V, lambda_k)
```

In `forward_decode`:

```rust
pub fn forward_decode(
    &self,
    x_step: Tensor<B, 3>,           // [B,1,C]
    cos_sin_step: (&Tensor<B, 4>, &Tensor<B, 4>),
    cache_layer: &mut TauCacheLayer<B>,
) -> Tensor<B, 3> {
    let [b, tq, c] = x_step.dims();
    debug_assert_eq!(tq, 1);

    // Project Q,K,V -> [B,H,1,D]
    let q = self.c_q.forward(x_step.clone())
        .clamp(-5.0, 5.0)
        .reshape([b, 1, self.nhead, self.head_dim])
        .swap_dims(1, 2);

    let k_new = self.c_k.forward(x_step.clone())
        .clamp(-5.0, 5.0)
        .reshape([b, 1, self.nkv_head, self.head_dim])
        .swap_dims(1, 2);

    let v_new = self.c_v.forward(x_step)
        .clamp(-5.0, 5.0)
        .reshape([b, 1, self.nkv_head, self.head_dim])
        .swap_dims(1, 2);

    let (cos_step, sin_step) = cos_sin_step;
    let q = apply_rotary_emb(q, cos_step, sin_step);
    let k_new = apply_rotary_emb(k_new, cos_step, sin_step);

    let q = rms_norm(q, 1e-6);
    let k_new = rms_norm(k_new, 1e-6);

    let tau_config = self.get_tau_config();
    let lambda_k_new = if self.use_sparse_laplacian {
        lambdas_from_heads_sparse_csr(k_new, self.sparse_laplacian.as_ref().unwrap(), &tau_config)
    } else {
        let laplacian = self.get_laplacian();
        lambdas_from_heads(k_new, &laplacian, &tau_config)
    };

    // Cache: (V_full, lambda_k_full)
    let (v_full, lambda_k_full) = match cache_layer.take() {
        Some((v_all, lk_all)) => (
            Tensor::cat(vec![v_all, v_new.clone()], 2),
            Tensor::cat(vec![lk_all, lambda_k_new.clone()], 2),
        ),
        None => (v_new.clone(), lambda_k_new.clone()),
    };
    *cache_layer = Some((v_full.clone(), lambda_k_full.clone()));

    let tk = v_full.dims();

    let y = self.scaled_tau_attention_decode(q, lambda_k_full, v_full, 1, tk);
    let y = y.swap_dims(1, 2).reshape([b, 1, c]);
    self.c_proj.forward(y)
}
```

`scaled_tau_attention_decode` recomputes $\lambda_q$, uses cached $\lambda_k$ and V, then same lambda-distance logits + causal softmax.

***

## 4. Wiring manifold.parquet and sparse Laplacian

The `pretraining.rs` module defines `DomainManifold` and `load_domain_manifold` to read a sparse Laplacian from `manifold.parquet` into a CSR matrix.

```rust
// pretraining.rs
#[derive(Debug, Clone)]
pub struct DomainManifold {
    pub matrix: CsMat<f64>,
}

pub fn load_domain_manifold(path: impl AsRef<Path>) -> Result<DomainManifold> {
    let lap: CsMat<f64> = parquet::load_sparse_matrix(&path)
        .with_context(|| format!("Failed to load laplacian parquet at {:?}", path.as_ref()))?;

    let (r, c) = lap.shape();
    anyhow::ensure!(
        r == c,
        "Domain Laplacian must be square (feature-space F×F), got {}×{}",
        r, c
    );

    Ok(DomainManifold { matrix: lap })
}
```

Tauformer can use this to replace the dense `laplacian_chain_dense` with a **sparse CSR Laplacian** and a CSR-based `lambdas_from_heads_sparse_csr` that computes $x^\top L x$ in $O(\text{nnz})$ instead of $O(D^2)$.

A typical constructor would be:

```rust
impl<B: Backend> TauModeAttention<B> {
    pub fn try_new_with_manifold_path(
        config: &NanoChatConfig,
        layer_idx: usize,
        device: &B::Device,
        manifold_path: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<Self> {
        let mut attn = Self::new(config, layer_idx, device);

        let manifold = load_domain_manifold(manifold_path.as_ref())?;
        let head_dim = attn.head_dim;
        anyhow::ensure!(manifold.dim() == head_dim, "dim mismatch");

        // convert sprs::CsMat -> SparseLaplacianCsr
        // ... as sketched earlier ...

        attn.use_sparse_laplacian = true;
        attn.sparse_laplacian = Some(sparse);
        Ok(attn)
    }
}
```

This ties tauformer directly to the **offline spectral manifold** learned in ArrowSpace, making $\lambda_\tau$ reflect real graph smoothness rather than an ad hoc chain.

***

## 5. Differences and advantages vs causal attention

### Kernel semantics

- **Causal attention**:
    - Kernel depends on **dot-products** $Q_i\cdot K_j$, so it is sensitive to Euclidean angle and norm in embedding space. [^6][^5]
    - No direct notion of “fitting a manifold” or graph smoothness.
- **Tauformer**:
    - Kernel depends on **spectral smoothness energy** of each head vector under a Laplacian: $x^\top L x$.
    - $\lambda_\tau(x)\in[0,1)$ captures how “rough” or “off-manifold” a feature is; attention distances measure differences in that energy, not just cosine similarity.
    - With a domain Laplacian (from `manifold.parquet`), this encodes **global structure of the embedding manifold** (e.g., scientific citation graph, knowledge graph) into the attention scores.


### Complexity and memory

Let $B$ = batch, $H$ = heads, $T$ = time, $D$ = head_dim, and nnz(L) the number of nonzeros in the Laplacian.

- **Causal** per decode step:
    - Compute logits $Q^\topK$: $O(B H T D)$.
    - Weighted sum: $O(B H T D)$.
    - Cache: store K and V: $2 B H_{kv} T D$ floats.
- **Tauformer (dense Laplacian)**:
    - Compute lambdas $x^\top L x$ via dense [D,D]: $O(B H T D^2)$.
    - Logits: $|\lambda_q - \lambda_k|$ cheap, $O(B H T)$.
    - Weighted sum: still $O(B H T D)$.
    - Cache: store V and λ_k: $B H_{kv} T D + B H_{kv} T$ floats.
        - Relative cache saving ≈ $(D-1)/(2D)$, close to **50%** for moderate $D$.
- **Tauformer with sparse manifold (CSR)**:
    - Compute lambdas in $O(B H T \cdot \text{nnz}(L))$. If $L$ is very sparse (as in ArrowSpace graphs), this is often much cheaper than $D^2$.
    - Same lightweight λ-distance logits and V aggregation.
    - Same ~50% KV cache memory saving.

So for **long sequences** (large $T$) and **moderate head_dim**, tauformer with sparse Laplacian shifts computational burden from $O(TD)$ (QK) to $O(\text{nnz}(L))$ and reduces cache memory roughly in half, while injecting manifold structure into attention.

### Practical effect (intuitively)

- Causal attention tends to focus on **local token compatibility** (dot-products) and uses RoPE to encode position.
- Tauformer’s attention scores incorporate **“how well each token’s features fit the global domain manifold”**, biasing attention towards tokens with similar spectral energy profiles (smoothness along the learned Laplacian).

In other words, tauformer is **attention over manifold energies**, not just over raw feature dot-products.

***

If you want, the next step is to:

- finalize and integrate `lambdas_from_heads_sparse_csr` into the codebase,
- add tests that compare dense vs sparse taumode lambdas on small Laplacians,
- and run a profiling micro-benchmark to see where the crossover point is (T vs nnz vs D) on your target hardware.
<span
