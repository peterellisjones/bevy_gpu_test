//! Simplest possible GPU test — pass numbers in, check results.
//!
//! Run with: `cargo test --test basic_math`

use bevy::render::render_resource::ShaderType;
use bevy_gpu_test::ComputeTest;

#[derive(Clone, Copy, Debug, ShaderType)]
struct MathInput {
    a: f32,
    b: f32,
    _pad1: f32,
    _pad2: f32,
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
struct MathOutput {
    sum: f32,
    product: f32,
    max_val: f32,
    min_val: f32,
}

#[test]
fn basic_arithmetic_on_gpu() {
    let inputs = vec![
        MathInput {
            a: 2.0,
            b: 3.0,
            _pad1: 0.0,
            _pad2: 0.0,
        },
        MathInput {
            a: -1.0,
            b: 4.0,
            _pad1: 0.0,
            _pad2: 0.0,
        },
        MathInput {
            a: 0.0,
            b: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        },
        MathInput {
            a: 100.5,
            b: -50.25,
            _pad1: 0.0,
            _pad2: 0.0,
        },
    ];

    let results: Vec<MathOutput> = ComputeTest::new("shaders/basic_math.wgsl", inputs).run();

    assert_eq!(results.len(), 4);

    // 2 + 3 = 5
    assert!((results[0].sum - 5.0).abs() < 1e-6);
    assert!((results[0].product - 6.0).abs() < 1e-6);
    assert!((results[0].max_val - 3.0).abs() < 1e-6);
    assert!((results[0].min_val - 2.0).abs() < 1e-6);

    // -1 + 4 = 3
    assert!((results[1].sum - 3.0).abs() < 1e-6);
    assert!((results[1].product - -4.0).abs() < 1e-6);
    assert!((results[1].max_val - 4.0).abs() < 1e-6);
    assert!((results[1].min_val - -1.0).abs() < 1e-6);

    // 0 + 0 = 0
    assert!((results[2].sum).abs() < 1e-6);
    assert!((results[2].product).abs() < 1e-6);

    // 100.5 + (-50.25) = 50.25
    assert!((results[3].sum - 50.25).abs() < 1e-4);
    assert!((results[3].product - -5050.125).abs() < 1e-2);
}
