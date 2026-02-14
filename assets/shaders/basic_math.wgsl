// Basic math test — reads (a, b) pairs, writes (sum, product, max, min).

struct Input {
    a: f32,
    b: f32,
    _pad1: f32,
    _pad2: f32,
}

struct Output {
    sum: f32,
    product: f32,
    max_val: f32,
    min_val: f32,
}

@group(0) @binding(0) var<storage, read> inputs: array<Input>;
@group(0) @binding(1) var<storage, read_write> outputs: array<Output>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&inputs) {
        return;
    }

    let input = inputs[i];
    outputs[i] = Output(
        input.a + input.b,
        input.a * input.b,
        max(input.a, input.b),
        min(input.a, input.b),
    );
}
