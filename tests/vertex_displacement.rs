//! Test a vertex displacement function with uniform config parameters.
//!
//! Demonstrates `with_uniform` for passing configuration to the shader,
//! and shows how to test logic that would normally live in a vertex shader
//! by wrapping it in a compute dispatch.
//!
//! Run with: `cargo test --test vertex_displacement`

use bevy::render::render_resource::ShaderType;
use bevy_gpu_test::ComputeTest;

#[derive(Clone, Copy, Debug, ShaderType)]
struct Config {
    amplitude: f32,
    frequency: f32,
    time: f32,
    _pad: f32,
}

#[derive(Clone, Copy, Debug, ShaderType)]
struct VertexInput {
    position_x: f32,
    position_y: f32,
    position_z: f32,
    _pad: f32,
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
struct VertexOutput {
    displaced_x: f32,
    displaced_y: f32,
    displaced_z: f32,
    _pad: f32,
}

/// Rust reference implementation of the same wave displacement function.
fn wave_displace_cpu(
    x: f32,
    y: f32,
    z: f32,
    amplitude: f32,
    frequency: f32,
    time: f32,
) -> [f32; 3] {
    let wave = (x * frequency + time).sin() * (z * frequency + time * 0.7).cos();
    [x, y + wave * amplitude, z]
}

#[test]
fn vertex_displacement_matches_cpu() {
    let config = Config {
        amplitude: 2.0,
        frequency: 0.5,
        time: 1.0,
        _pad: 0.0,
    };

    let inputs = vec![
        VertexInput {
            position_x: 0.0,
            position_y: 0.0,
            position_z: 0.0,
            _pad: 0.0,
        },
        VertexInput {
            position_x: 1.0,
            position_y: 0.0,
            position_z: 0.0,
            _pad: 0.0,
        },
        VertexInput {
            position_x: 0.0,
            position_y: 0.0,
            position_z: 1.0,
            _pad: 0.0,
        },
        VertexInput {
            position_x: 3.0,
            position_y: 5.0,
            position_z: -2.0,
            _pad: 0.0,
        },
        VertexInput {
            position_x: -10.0,
            position_y: 1.0,
            position_z: 10.0,
            _pad: 0.0,
        },
    ];

    let gpu_results: Vec<VertexOutput> =
        ComputeTest::new("shaders/vertex_displacement.wgsl", inputs.clone())
            .with_uniform(config)
            .run();

    assert_eq!(gpu_results.len(), inputs.len());

    for (i, (input, gpu)) in inputs.iter().zip(gpu_results.iter()).enumerate() {
        let cpu = wave_displace_cpu(
            input.position_x,
            input.position_y,
            input.position_z,
            config.amplitude,
            config.frequency,
            config.time,
        );

        let dx = (gpu.displaced_x - cpu[0]).abs();
        let dy = (gpu.displaced_y - cpu[1]).abs();
        let dz = (gpu.displaced_z - cpu[2]).abs();

        assert!(
            dx < 1e-5 && dy < 1e-5 && dz < 1e-5,
            "Vertex {i}: GPU=({:.6}, {:.6}, {:.6}) CPU=({:.6}, {:.6}, {:.6}) diff=({dx:.2e}, {dy:.2e}, {dz:.2e})",
            gpu.displaced_x,
            gpu.displaced_y,
            gpu.displaced_z,
            cpu[0],
            cpu[1],
            cpu[2],
        );
    }
}

#[test]
fn zero_amplitude_means_no_displacement() {
    let config = Config {
        amplitude: 0.0,
        frequency: 1.0,
        time: 42.0,
        _pad: 0.0,
    };

    let inputs = vec![
        VertexInput {
            position_x: 5.0,
            position_y: 3.0,
            position_z: -1.0,
            _pad: 0.0,
        },
        VertexInput {
            position_x: -100.0,
            position_y: 0.0,
            position_z: 50.0,
            _pad: 0.0,
        },
    ];

    let results: Vec<VertexOutput> =
        ComputeTest::new("shaders/vertex_displacement.wgsl", inputs.clone())
            .with_uniform(config)
            .run();

    for (input, output) in inputs.iter().zip(results.iter()) {
        assert!((output.displaced_x - input.position_x).abs() < 1e-6);
        assert!((output.displaced_y - input.position_y).abs() < 1e-6);
        assert!((output.displaced_z - input.position_z).abs() < 1e-6);
    }
}
