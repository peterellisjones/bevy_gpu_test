//! CPU/GPU parity test — verify a WGSL function matches its Rust equivalent.
//!
//! This is the pattern for catching floating-point divergence between CPU and
//! GPU implementations of the same algorithm. Write the function in both Rust
//! and WGSL, feed the same inputs to both, and compare.
//!
//! Run with: `cargo test --test cpu_gpu_parity`

use bevy::render::render_resource::ShaderType;
use bevy_gpu_test::ComputeTest;

#[derive(Clone, Copy, Debug, ShaderType)]
struct LerpInput {
    a: f32,
    b: f32,
    t: f32,
    _pad: f32,
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
struct LerpOutput {
    result: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

/// Rust reference implementation of smoothstep interpolation.
fn smooth_lerp_cpu(a: f32, b: f32, t: f32) -> f32 {
    let clamped = t.clamp(0.0, 1.0);
    let smooth_t = clamped * clamped * (3.0 - 2.0 * clamped);
    a + (b - a) * smooth_t
}

#[test]
fn smoothstep_parity() {
    // Generate a sweep of test cases covering edge cases and typical values
    let mut inputs = Vec::new();

    // Sweep t from -0.5 to 1.5 (beyond the clamp range)
    for i in 0..64 {
        #[allow(clippy::cast_precision_loss)]
        let t = -0.5 + (i as f32 / 63.0) * 2.0;
        inputs.push(LerpInput {
            a: 0.0,
            b: 1.0,
            t,
            _pad: 0.0,
        });
    }

    // Different value ranges
    let pairs = [
        (0.0, 100.0),
        (-50.0, 50.0),
        (1000.0, 1001.0), // small delta, large values
        (-0.001, 0.001),  // small delta, small values
    ];
    for (a, b) in pairs {
        for i in 0..16 {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / 15.0;
            inputs.push(LerpInput { a, b, t, _pad: 0.0 });
        }
    }

    let gpu_results: Vec<LerpOutput> =
        ComputeTest::new("shaders/lerp_parity.wgsl", inputs.clone()).run();

    assert_eq!(gpu_results.len(), inputs.len());

    let mut max_diff: f32 = 0.0;
    let mut failures = 0;

    for (i, (input, gpu)) in inputs.iter().zip(gpu_results.iter()).enumerate() {
        let cpu = smooth_lerp_cpu(input.a, input.b, input.t);
        let diff = (gpu.result - cpu).abs();
        max_diff = max_diff.max(diff);

        // GPU and CPU floating-point arithmetic are not bit-identical.
        // A tolerance of 1e-5 is typical for single-precision parity tests.
        if diff > 1e-5 {
            failures += 1;
            if failures <= 5 {
                eprintln!(
                    "  [{i}] a={}, b={}, t={:.4}: CPU={:.8} GPU={:.8} diff={:.2e}",
                    input.a, input.b, input.t, cpu, gpu.result, diff
                );
            }
        }
    }

    assert_eq!(
        failures,
        0,
        "Parity failures: {failures}/{} (max diff: {max_diff:.2e})",
        inputs.len()
    );

    println!(
        "Parity PASSED: {} inputs, max diff: {max_diff:.2e}",
        inputs.len()
    );
}
