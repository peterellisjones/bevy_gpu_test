# bevy_gpu_test

Run WGSL shader code on a real GPU from a Rust test and get results back.

Testing GPU shader code in Bevy requires significant boilerplate: a headless app, storage buffers, a compute pipeline, a render graph node, workgroup dispatch, and GPU readback. This crate handles all of that so you can focus on the shader and the assertions.

The idea: wrap whatever WGSL code you want to test in a thin compute shader, pass inputs in, get outputs back, assert in Rust. This works for any shader logic -- noise functions, vertex displacement, lighting math, procedural generation, simulation. If you can call it from WGSL, you can test it.

## Quick start

Add to your `Cargo.toml`:

```toml
[dev-dependencies]
bevy_gpu_test = "*"
```

Write a compute shader (`assets/shaders/add.wgsl`):

```wgsl
struct Input {
    a: f32,
    b: f32,
    _pad1: f32,
    _pad2: f32,
}

struct Output {
    sum: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<storage, read> inputs: array<Input>;
@group(0) @binding(1) var<storage, read_write> outputs: array<Output>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&inputs) { return; }
    outputs[i] = Output(inputs[i].a + inputs[i].b, 0.0, 0.0, 0.0);
}
```

Test it (`tests/add.rs`):

```rust
use bevy::render::render_resource::ShaderType;
use bevy_gpu_test::ComputeTest;

#[derive(Clone, Copy, Debug, ShaderType)]
struct Input {
    a: f32,
    b: f32,
    _pad1: f32,
    _pad2: f32,
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
struct Output {
    sum: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

#[test]
fn addition_on_gpu() {
    let inputs = vec![
        Input { a: 1.0, b: 2.0, _pad1: 0.0, _pad2: 0.0 },
        Input { a: -5.0, b: 3.0, _pad1: 0.0, _pad2: 0.0 },
    ];

    let results: Vec<Output> = ComputeTest::new("shaders/add.wgsl", inputs).run();

    assert!((results[0].sum - 3.0).abs() < 1e-6);
    assert!((results[1].sum - -2.0).abs() < 1e-6);
}
```

Run with `cargo test`.

## Uniform buffers

Use `with_uniform` when your shader needs configuration parameters. This adds a uniform at `@binding(0)` and shifts the storage buffers to bindings 1 and 2:

```rust
let results: Vec<Output> = ComputeTest::new("shaders/my_shader.wgsl", inputs)
    .with_uniform(config)
    .run();
```

The corresponding shader layout:

```wgsl
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> inputs: array<Input>;
@group(0) @binding(2) var<storage, read_write> outputs: array<Output>;
```

## Testing non-compute shaders

You can test any WGSL code, not just compute shaders. Write your reusable logic as WGSL functions, `#import` them into a thin compute shader wrapper, and test through that:

```wgsl
#import "shaders/my_vertex_logic.wgsl" as vertex_logic

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&inputs) { return; }
    // Call the function you're actually testing
    let displaced = vertex_logic::displace(inputs[i].position, config.time);
    outputs[i] = Output(displaced.x, displaced.y, displaced.z, 0.0);
}
```

## How it works

1. A headless Bevy app starts (no window, no display)
2. Your input data is uploaded to a GPU storage buffer
3. A compute pipeline is created from your shader
4. A render graph node dispatches the compute work
5. A `Readback` copies the output buffer back to the CPU
6. The app exits and returns the typed results

Tests run on the real GPU with the real WGSL compiler. No mocking.

## Bind group layout

**Without uniform** (default):

| Binding | Type | Usage |
|---------|------|-------|
| `@binding(0)` | `storage<read>` | Input buffer |
| `@binding(1)` | `storage<read_write>` | Output buffer |

**With uniform** (`with_uniform`):

| Binding | Type | Usage |
|---------|------|-------|
| `@binding(0)` | `uniform` | Config/params |
| `@binding(1)` | `storage<read>` | Input buffer |
| `@binding(2)` | `storage<read_write>` | Output buffer |

## Timeout and diagnostics

Tests time out after 5 seconds by default. If the shader fails to compile or the pipeline never becomes ready, the panic message includes the pipeline state and common failure causes.

Override the timeout:

```rust
use std::time::Duration;

let results: Vec<Output> = ComputeTest::new("shaders/complex.wgsl", inputs)
    .with_timeout(Duration::from_secs(60))
    .run();
```

## Requirements

- Bevy 0.18
- A GPU (integrated or discrete) -- tests will fail on headless CI without one

## License

MIT OR Apache-2.0
