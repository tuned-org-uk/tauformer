use crate::backend::AutoBackend;
use crate::config::NanoChatConfig;
use crate::tauattention::{TauCacheLayer, TauModeAttention};
use burn::tensor::Tensor;

type TestBackend = AutoBackend;

fn test_config() -> NanoChatConfig {
    NanoChatConfig {
        vocab_size: 1000,
        n_layer: 2,
        n_head: 4,
        n_kv_head: 2,
        n_embd: 128,
        block_size: 64,
        sequence_len: 64,
        dropout: 0.0,
    }
}

#[test]
fn test_tau_mode_attention_creation() {
    let device = Default::default();
    let config = test_config();
    let layer_idx = 0;

    let attn = TauModeAttention::<TestBackend>::new(&config, layer_idx, &device);

    assert_eq!(attn.nhead, 4);
    assert_eq!(attn.nkv_head, 2);
    assert_eq!(attn.head_dim, 32);
    assert_eq!(attn.layer_idx, 0);
}

#[test]
fn test_tau_config_values() {
    let device = Default::default();
    let config = test_config();

    let attn = TauModeAttention::<TestBackend>::new(&config, 0, &device);

    assert_eq!(attn.tau, 1.0);
    assert_eq!(attn.eps, 1.0e-6);
    assert_eq!(attn.temperature, 1.0);
}

#[test]
fn test_laplacian_matrix_stored() {
    let device = Default::default();
    let config = test_config();

    let attn = TauModeAttention::<TestBackend>::new(&config, 0, &device);
    let lap = attn.get_laplacian();

    assert_eq!(lap.dim(), 32);
}

#[test]
fn test_forward_pass_shape() {
    let device = Default::default();
    let config = test_config();
    let (b, t, c) = (2, 16, 128);

    let attn = TauModeAttention::<TestBackend>::new(&config, 0, &device);

    let x: Tensor<TestBackend, 3> = Tensor::random(
        [b, t, c],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );

    let head_dim = config.n_embd / config.n_head;

    // apply_rotary_emb expects cos/sin cache shaped [1, T, 1, D/2] (then it slices + expands).
    let cos: Tensor<TestBackend, 4> = Tensor::ones([1, t, 1, head_dim / 2], &device);
    let sin: Tensor<TestBackend, 4> = Tensor::zeros([1, t, 1, head_dim / 2], &device);

    let output = attn.forward(x, (&cos, &sin));
    assert_eq!(output.dims(), [b, t, c]);
}

#[test]
fn test_forward_decode_single_step() {
    let device = Default::default();
    let config = test_config();
    let (b, c) = (2, 128);

    let attn = TauModeAttention::<TestBackend>::new(&config, 0, &device);

    let x_step: Tensor<TestBackend, 3> = Tensor::random(
        [b, 1, c],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );

    let head_dim = config.n_embd / config.n_head;

    // decode uses a single-position RoPE cache: [1, 1, 1, D/2]
    let cos: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 1, head_dim / 2], &device);
    let sin: Tensor<TestBackend, 4> = Tensor::zeros([1, 1, 1, head_dim / 2], &device);

    let mut cache: TauCacheLayer<TestBackend> = None;

    let output = attn.forward_decode(x_step, (&cos, &sin), &mut cache);

    assert_eq!(output.dims(), [b, 1, c]);
    assert!(cache.is_some());
}

#[test]
fn test_forward_decode_cache_accumulation() {
    let device = Default::default();
    let config = test_config();
    let (b, c) = (1, 128);

    let attn = TauModeAttention::<TestBackend>::new(&config, 0, &device);

    let head_dim = config.n_embd / config.n_head;
    let cos: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 1, head_dim / 2], &device);
    let sin: Tensor<TestBackend, 4> = Tensor::zeros([1, 1, 1, head_dim / 2], &device);

    let mut cache: TauCacheLayer<TestBackend> = None;

    let x1: Tensor<TestBackend, 3> = Tensor::random(
        [b, 1, c],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );
    attn.forward_decode(x1, (&cos, &sin), &mut cache);

    {
        let (v_cached, lambda_k_cached) = cache.as_ref().unwrap();
        assert_eq!(v_cached.dims()[2], 1);
        assert_eq!(lambda_k_cached.dims()[2], 1);
    }

    let x2: Tensor<TestBackend, 3> = Tensor::random(
        [b, 1, c],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );
    attn.forward_decode(x2, (&cos, &sin), &mut cache);

    {
        let (v_cached, lambda_k_cached) = cache.as_ref().unwrap();
        assert_eq!(v_cached.dims()[2], 2);
        assert_eq!(lambda_k_cached.dims()[2], 2);
    }

    let x3: Tensor<TestBackend, 3> = Tensor::random(
        [b, 1, c],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );
    attn.forward_decode(x3, (&cos, &sin), &mut cache);

    {
        let (v_cached, lambda_k_cached) = cache.as_ref().unwrap();
        assert_eq!(v_cached.dims()[2], 3);
        assert_eq!(lambda_k_cached.dims()[2], 3);
    }
}

#[test]
fn test_mqa_expansion_in_forward() {
    let device = Default::default();
    let mut config = test_config();
    config.n_head = 8;
    config.n_kv_head = 2;

    let (b, t, c) = (1, 8, config.n_embd);

    let attn = TauModeAttention::<TestBackend>::new(&config, 0, &device);

    let x: Tensor<TestBackend, 3> = Tensor::random(
        [b, t, c],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );

    let head_dim = config.n_embd / config.n_head;
    let cos: Tensor<TestBackend, 4> = Tensor::ones([1, t, 1, head_dim / 2], &device);
    let sin: Tensor<TestBackend, 4> = Tensor::zeros([1, t, 1, head_dim / 2], &device);

    let output = attn.forward(x, (&cos, &sin));
    assert_eq!(output.dims(), [b, t, c]);
}

#[test]
fn test_get_helpers_reconstruct_correctly() {
    let device = Default::default();
    let config = test_config();
    let attn = TauModeAttention::<TestBackend>::new(&config, 0, &device);

    let lap = attn.get_laplacian();
    let tau_config = attn.get_tau_config();

    assert_eq!(lap.dim(), 32);
    assert_eq!(tau_config.tau, 1.0);
    assert_eq!(tau_config.eps, 1.0e-6);
    assert_eq!(tau_config.temperature, 1.0);
}
