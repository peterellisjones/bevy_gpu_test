//! Verify that a broken shader path triggers a timeout panic rather than
//! hanging indefinitely.
//!
//! Run with: `cargo test --test timeout`

use std::time::Duration;

use bevy::render::render_resource::ShaderType;
use bevy_gpu_test::ComputeTest;

#[derive(Clone, Copy, Debug, ShaderType)]
struct In {
    value: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
struct Out {
    value: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

#[test]
#[should_panic(expected = "timed out waiting for results")]
fn nonexistent_shader_times_out() {
    let inputs = vec![In {
        value: 1.0,
        _pad1: 0.0,
        _pad2: 0.0,
        _pad3: 0.0,
    }];

    let _results: Vec<Out> = ComputeTest::new("shaders/does_not_exist.wgsl", inputs)
        .with_timeout(Duration::from_secs(3))
        .run();
}
