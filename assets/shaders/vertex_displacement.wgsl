// Test a vertex displacement function with configurable parameters.
//
// This demonstrates testing shader logic that would normally run in a vertex
// shader. The displacement function is called from a compute wrapper so we
// can feed it test data and read back results.

struct Config {
    amplitude: f32,
    frequency: f32,
    time: f32,
    _pad: f32,
}

struct Input {
    position_x: f32,
    position_y: f32,
    position_z: f32,
    _pad: f32,
}

struct Output {
    displaced_x: f32,
    displaced_y: f32,
    displaced_z: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> inputs: array<Input>;
@group(0) @binding(2) var<storage, read_write> outputs: array<Output>;

// The function under test — a simple wave displacement.
// In a real project this would be #imported from your vertex shader module.
fn wave_displace(pos: vec3<f32>, amplitude: f32, frequency: f32, time: f32) -> vec3<f32> {
    let wave = sin(pos.x * frequency + time) * cos(pos.z * frequency + time * 0.7);
    return vec3<f32>(pos.x, pos.y + wave * amplitude, pos.z);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&inputs) {
        return;
    }

    let input = inputs[i];
    let pos = vec3<f32>(input.position_x, input.position_y, input.position_z);
    let result = wave_displace(pos, config.amplitude, config.frequency, config.time);

    outputs[i] = Output(result.x, result.y, result.z, 0.0);
}
