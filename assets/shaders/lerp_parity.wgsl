// CPU/GPU parity test for a smoothstep-based lerp function.
//
// Tests that the WGSL implementation matches the Rust implementation
// across a range of inputs. This pattern catches floating-point
// divergence between CPU and GPU.

struct Input {
    a: f32,
    b: f32,
    t: f32,
    _pad: f32,
}

struct Output {
    result: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<storage, read> inputs: array<Input>;
@group(0) @binding(1) var<storage, read_write> outputs: array<Output>;

// The function under test — smoothstep interpolation.
fn smooth_lerp(a: f32, b: f32, t: f32) -> f32 {
    let clamped = clamp(t, 0.0, 1.0);
    let smooth_t = clamped * clamped * (3.0 - 2.0 * clamped);
    return a + (b - a) * smooth_t;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&inputs) {
        return;
    }

    let input = inputs[i];
    outputs[i] = Output(smooth_lerp(input.a, input.b, input.t), 0.0, 0.0, 0.0);
}
