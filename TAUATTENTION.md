# Notes about “same results”

## Analysis of the `test_compare.rs` module.

- The tests above **do** enforce “same results” where it’s actually correct to enforce it: `forward(...)` must match `forward_decode(...)` for the same module (cache correctness).
- They also explicitly test that **tau != causal** outputs in general, which is expected and healthy (otherwise tauattention would just be dot-product attention).


## Estimated memory savings (cache)

Standard causal attention caches **K and V** per layer: $K\in[B,H_{kv},T,D]$ and $V\in[B,H_{kv},T,D]$.
Tau attention caches **V and $\lambda_k$**: $V\in[B,H_{kv},T,D]$ and $\lambda_k\in[B,H_{kv},T]$.

In “number of floats stored” per layer:

- Causal cache: $2 \cdot B \cdot H_{kv} \cdot T \cdot D$ floats.
- Tau cache: $B \cdot H_{kv} \cdot T \cdot (D + 1)$ floats.

Relative cache saving:

$$
1 - \frac{D+1}{2D} = \frac{D-1}{2D}
$$

So for typical head dimensions:

- $D=32$ → ~48.4% fewer cached floats.
- $D=64$ → ~49.2% fewer cached floats.


## Estimated time complexity (decode step)

For a single decode step ($T_q=1$, $T_k=T$):

**Causal attention**

- Logits $QK^\top$: $O(B \cdot H \cdot T \cdot D)$.
- Weighted sum $\text{softmax}(\cdot)V$: $O(B \cdot H \cdot T \cdot D)$.

So roughly $O(2 \cdot B \cdot H \cdot T \cdot D)$ plus softmax $O(B\cdot H \cdot T)$.

**Tau attention (current implementation uses dense Laplacian in-feature space)**

- Compute $\lambda_q$ via Laplacian projection: $O(B \cdot H \cdot D^2)$ (dense $D\times D$ matmul).
- Lambda-distance logits: $O(B \cdot H \cdot T)$.
- Weighted sum with $V$: still $O(B \cdot H \cdot T \cdot D)$.

So roughly $O(B \cdot H \cdot (D^2 + T\cdot D))$.

**Break-even intuition:** tau avoids the $QK^\top$ term but adds a $D^2$ term; tau becomes computationally attractive when $T \gtrsim D$.

If you want, the next step is to wire `manifold.parquet` into `TauModeAttention` (currently it builds a chain Laplacian internally) so the Laplacian multiplication can be made **sparse** (and then the tau path can become strictly cheaper for long contexts).
