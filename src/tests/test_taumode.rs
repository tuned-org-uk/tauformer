use crate::backend::AutoBackend;
use crate::taumode::*;
use burn::tensor::Tensor;

type TestBackend = AutoBackend;

#[test]
fn test_tau_mode_config_default() {
    let config = TauModeConfig::default();
    assert_eq!(config.tau, 1.0);
    assert_eq!(config.eps, 1.0e-6);
    assert_eq!(config.temperature, 1.0);
}

#[test]
fn test_laplacian_chain_dense_structure() {
    let device = Default::default();
    let d = 4;
    let lap = laplacian_chain_dense::<TestBackend>(d, &device);

    assert_eq!(lap.dim(), d);

    let [rows, cols] = lap.matrix.dims();
    assert_eq!(rows, d);
    assert_eq!(cols, d);

    let data: Vec<f32> = lap.matrix.to_data().to_vec().unwrap();

    // First row: [1, -1, 0, 0] (degree 1)
    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], -1.0);
    assert_eq!(data[2], 0.0);

    // Middle row (index 1): [-1, 2, -1, 0] (degree 2)
    assert_eq!(data[4], -1.0);
    assert_eq!(data[5], 2.0);
    assert_eq!(data[6], -1.0);

    // Last row: [0, 0, -1, 1] (degree 1)
    assert_eq!(data[15], 1.0);
    assert_eq!(data[14], -1.0);
}

#[test]
fn test_lambdas_from_heads_shape() {
    let device = Default::default();
    let (b, h, t, d) = (2, 4, 8, 16);

    let x: Tensor<TestBackend, 4> = Tensor::random(
        [b, h, t, d],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    let lap = laplacian_chain_dense::<TestBackend>(d, &device);
    let cfg = TauModeConfig::default();

    let lambdas = lambdas_from_heads(x, &lap, &cfg);

    assert_eq!(lambdas.dims(), [b, h, t]);
}

#[test]
fn test_lambdas_bounded_range() {
    let device = Default::default();
    let (b, h, t, d) = (1, 1, 4, 8);

    let x: Tensor<TestBackend, 4> = Tensor::random(
        [b, h, t, d],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    let lap = laplacian_chain_dense::<TestBackend>(d, &device);
    let cfg = TauModeConfig {
        tau: 1.0,
        eps: 1e-6,
        temperature: 1.0,
    };

    let lambdas = lambdas_from_heads(x, &lap, &cfg);
    let data: Vec<f32> = lambdas.to_data().to_vec().unwrap();

    for &val in data.iter() {
        assert!(val >= 0.0, "Lambda should be non-negative, got {}", val);
        assert!(val < 1.0, "Lambda should be < 1.0, got {}", val);
    }
}

#[test]
fn test_mqa_expand_heads_3_no_expansion() {
    let device = Default::default();
    let (b, h, t) = (2, 8, 16);

    let x: Tensor<TestBackend, 3> = Tensor::random(
        [b, h, t],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    let expanded = mqa_expand_heads_3(x.clone(), h, h);
    assert_eq!(expanded.dims(), [b, h, t]);
}

#[test]
fn test_mqa_expand_heads_3_expansion() {
    let device = Default::default();
    let (b, hkv, t) = (2, 2, 16);
    let n_head = 8;

    let x: Tensor<TestBackend, 3> = Tensor::random(
        [b, hkv, t],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    let expanded = mqa_expand_heads_3(x, n_head, hkv);
    assert_eq!(expanded.dims(), [b, n_head, t]);
}

#[test]
fn test_mqa_expand_heads_4_no_expansion() {
    let device = Default::default();
    let (b, h, t, d) = (2, 8, 16, 64);

    let x: Tensor<TestBackend, 4> = Tensor::random(
        [b, h, t, d],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    let expanded = mqa_expand_heads_4(x.clone(), h, h);
    assert_eq!(expanded.dims(), [b, h, t, d]);
}

#[test]
fn test_mqa_expand_heads_4_expansion() {
    let device = Default::default();
    let (b, hkv, t, d) = (2, 2, 16, 64);
    let n_head = 8;

    let x: Tensor<TestBackend, 4> = Tensor::random(
        [b, hkv, t, d],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    let expanded = mqa_expand_heads_4(x, n_head, hkv);
    assert_eq!(expanded.dims(), [b, n_head, t, d]);
}

#[test]
fn test_taumode_distance_logits_shape() {
    let device = Default::default();
    let (b, h, tq, tk) = (2, 4, 8, 16);

    let lambda_q: Tensor<TestBackend, 3> = Tensor::random(
        [b, h, tq],
        burn::tensor::Distribution::Normal(0.0, 0.5),
        &device,
    );
    let lambda_k: Tensor<TestBackend, 3> = Tensor::random(
        [b, h, tk],
        burn::tensor::Distribution::Normal(0.0, 0.5),
        &device,
    );

    let cfg = TauModeConfig::default();
    let logits = taumode_distance_logits(lambda_q, lambda_k, &cfg);

    assert_eq!(logits.dims(), [b, h, tq, tk]);
}

#[test]
fn test_taumode_distance_logits_negative() {
    let device = Default::default();
    let (b, h, tq, tk) = (1, 1, 4, 4);

    let lambda_q: Tensor<TestBackend, 3> = Tensor::random(
        [b, h, tq],
        burn::tensor::Distribution::Normal(0.5, 0.1),
        &device,
    );
    let lambda_k: Tensor<TestBackend, 3> = Tensor::random(
        [b, h, tk],
        burn::tensor::Distribution::Normal(0.5, 0.1),
        &device,
    );

    let cfg = TauModeConfig {
        tau: 1.0,
        eps: 1e-6,
        temperature: 1.0,
    };

    let logits = taumode_distance_logits(lambda_q, lambda_k, &cfg);
    let data: Vec<f32> = logits.to_data().to_vec().unwrap();

    for &val in data.iter() {
        assert!(val <= 0.0, "Logit should be non-positive, got {}", val);
    }
}

#[test]
fn test_taumode_distance_diagonal_zero() {
    let device = Default::default();
    let (b, h, t) = (1, 1, 4);

    let lambda: Tensor<TestBackend, 3> = Tensor::from_floats([[[0.3, 0.5, 0.7, 0.9]]], &device);

    let cfg = TauModeConfig::default();
    let logits = taumode_distance_logits(lambda.clone(), lambda, &cfg);

    let data: Vec<f32> = logits.to_data().to_vec().unwrap();
    assert!((data[0] - 0.0).abs() < 1e-6);
    assert!((data[5] - 0.0).abs() < 1e-6);
    assert!((data[10] - 0.0).abs() < 1e-6);
    assert!((data[15] - 0.0).abs() < 1e-6);
}

#[test]
fn test_causal_softmax_over_keys_shape() {
    let device = Default::default();
    let (b, h, tq, tk) = (2, 4, 8, 16);

    let att: Tensor<TestBackend, 4> = Tensor::random(
        [b, h, tq, tk],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    let att_soft = causal_softmax_over_keys(att, tq, tk, h);
    assert_eq!(att_soft.dims(), [b, h, tq, tk]);
}

#[test]
fn test_causal_softmax_sums_to_one() {
    let device = Default::default();
    let (b, h, tq, tk) = (1, 1, 4, 8);

    let att: Tensor<TestBackend, 4> = Tensor::random(
        [b, h, tq, tk],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    let att_soft = causal_softmax_over_keys(att, tq, tk, h);

    let row_sums = att_soft.sum_dim(3);
    let data: Vec<f32> = row_sums.to_data().to_vec().unwrap();

    for &sum in data.iter() {
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Row sum should be 1.0, got {}",
            sum
        );
    }
}

#[test]
fn test_causal_softmax_upper_triangular_zero() {
    let device = Default::default();
    let (b, h, t) = (1, 1, 4);

    let att: Tensor<TestBackend, 4> = Tensor::ones([b, h, t, t], &device);
    let att_soft = causal_softmax_over_keys(att, t, t, h);
    let data: Vec<f32> = att_soft.to_data().to_vec().unwrap();

    assert!(
        data[1] < 1e-8,
        "Upper triangle should be ~0, got {}",
        data[1]
    );
    assert!(
        data[2] < 1e-8,
        "Upper triangle should be ~0, got {}",
        data[2]
    );
    assert!(
        data[6] < 1e-8,
        "Upper triangle should be ~0, got {}",
        data[6]
    );
}
