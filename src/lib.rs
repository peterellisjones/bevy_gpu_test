//! # `bevy_gpu_test`
//!
//! Run WGSL shader code on a real GPU from a Rust test and get results back.
//!
//! Testing GPU shader code is hard. You need a headless Bevy app, storage buffers,
//! a compute pipeline, a render graph node, workgroup dispatch, GPU readback — easily
//! 300+ lines of boilerplate before you can even check a single value. This crate
//! handles all of that.
//!
//! The idea is simple: wrap whatever WGSL code you want to test in a thin compute
//! shader, pass inputs in, get outputs back, and assert in Rust. This works for
//! testing any shader logic — noise functions, vertex displacement, lighting math,
//! procedural generation, simulation kernels — anything you can call from WGSL.
//!
//! ## Quick start
//!
//! Write a compute shader that reads inputs and writes outputs:
//!
//! ```wgsl
//! // assets/shaders/add.wgsl
//!
//! struct Input {
//!     a: f32,
//!     b: f32,
//!     _pad1: f32,
//!     _pad2: f32,
//! }
//!
//! struct Output {
//!     sum: f32,
//!     _pad1: f32,
//!     _pad2: f32,
//!     _pad3: f32,
//! }
//!
//! @group(0) @binding(0) var<storage, read> inputs: array<Input>;
//! @group(0) @binding(1) var<storage, read_write> outputs: array<Output>;
//!
//! @compute @workgroup_size(64)
//! fn main(@builtin(global_invocation_id) id: vec3<u32>) {
//!     let i = id.x;
//!     if i >= arrayLength(&inputs) { return; }
//!     outputs[i] = Output(inputs[i].a + inputs[i].b, 0.0, 0.0, 0.0);
//! }
//! ```
//!
//! Then test it from Rust:
//!
//! ```rust,no_run
//! use bevy::render::render_resource::ShaderType;
//! use bevy_gpu_test::ComputeTest;
//!
//! #[derive(Clone, Copy, Debug, ShaderType)]
//! struct Input {
//!     a: f32,
//!     b: f32,
//!     _pad1: f32,
//!     _pad2: f32,
//! }
//!
//! #[derive(Clone, Copy, Debug, Default, ShaderType)]
//! struct Output {
//!     sum: f32,
//!     _pad1: f32,
//!     _pad2: f32,
//!     _pad3: f32,
//! }
//!
//! #[test]
//! fn addition_on_gpu() {
//!     let inputs = vec![
//!         Input { a: 1.0, b: 2.0, _pad1: 0.0, _pad2: 0.0 },
//!         Input { a: -5.0, b: 3.0, _pad1: 0.0, _pad2: 0.0 },
//!     ];
//!
//!     let results: Vec<Output> = ComputeTest::new("shaders/add.wgsl", inputs).run();
//!
//!     assert!((results[0].sum - 3.0).abs() < 1e-6);
//!     assert!((results[1].sum - -2.0).abs() < 1e-6);
//! }
//! ```
//!
//! ## With a uniform buffer
//!
//! If your shader needs configuration parameters, use [`ComputeTest::with_uniform`].
//! This adds a uniform at `@binding(0)`, shifting the storage buffers to 1 and 2:
//!
//! ```rust,no_run
//! # use bevy::render::render_resource::ShaderType;
//! # use bevy_gpu_test::ComputeTest;
//! #[derive(Clone, Copy, Debug, ShaderType)]
//! struct Config {
//!     scale: f32,
//!     offset: f32,
//!     _pad1: f32,
//!     _pad2: f32,
//! }
//!
//! # #[derive(Clone, Copy, Debug, ShaderType)]
//! # struct Input { x: f32, _p1: f32, _p2: f32, _p3: f32 }
//! # #[derive(Clone, Copy, Debug, Default, ShaderType)]
//! # struct Output { v: f32, _p1: f32, _p2: f32, _p3: f32 }
//! let inputs = vec![Input { x: 5.0, _p1: 0.0, _p2: 0.0, _p3: 0.0 }];
//! let results: Vec<Output> = ComputeTest::new("shaders/scale.wgsl", inputs)
//!     .with_uniform(Config { scale: 2.0, offset: 10.0, _pad1: 0.0, _pad2: 0.0 })
//!     .run();
//! ```
//!
//! ## Testing non-compute shaders
//!
//! You can test *any* WGSL code this way, not just compute shaders. Write your
//! reusable logic as WGSL functions, `#import` them into a thin compute shader
//! wrapper, and test through that. Vertex displacement, fragment math, noise,
//! simulation — if it runs on the GPU, you can test it.
//!
//! ## How it works
//!
//! 1. Spins up a headless Bevy app (no window)
//! 2. Uploads your input data to a GPU storage buffer
//! 3. Creates a compute pipeline from your shader
//! 4. Dispatches it via a render graph node
//! 5. Reads the output buffer back to the CPU via [`Readback`](bevy::render::gpu_readback::Readback)
//! 6. Returns the typed results to your test
//!
//! Tests run on a real GPU with the real WGSL compiler — no mocking, no
//! approximations.
//!
//! ## Bind group layout
//!
//! **Without uniform** (default):
//! - `@binding(0)`: `storage<read>` — input buffer
//! - `@binding(1)`: `storage<read_write>` — output buffer
//!
//! **With uniform** ([`with_uniform`](ComputeTest::with_uniform)):
//! - `@binding(0)`: `uniform` — config/params
//! - `@binding(1)`: `storage<read>` — input buffer
//! - `@binding(2)`: `storage<read_write>` — output buffer
//!
//! ## Timeout
//!
//! By default, tests time out after 5 seconds with a diagnostic panic message
//! that includes the pipeline state and common failure causes. Override with
//! [`ComputeTest::with_timeout`].

mod run;

use bevy::render::render_resource::{encase, ShaderType};
use run::run_compute_test;
use std::sync::{Arc, Mutex};

/// Builder for a GPU compute test.
///
/// Configures a headless Bevy app that loads a compute shader, uploads input data
/// to the GPU, dispatches the shader, and reads back the output buffer.
///
/// # Type parameters
///
/// - `I`: The input element type. Must derive [`ShaderType`].
/// - `O`: The output element type. Must derive [`ShaderType`] and [`Default`].
pub struct ComputeTest<I, O> {
    pub(crate) shader_path: String,
    pub(crate) inputs: Vec<I>,
    pub(crate) workgroup_size: u32,
    pub(crate) uniform_bytes: Option<Vec<u8>>,
    pub(crate) result_channel: Arc<Mutex<Option<Vec<O>>>>,
    pub(crate) entry_point: String,
    pub(crate) timeout: std::time::Duration,
}

impl<I, O> ComputeTest<I, O>
where
    I: ShaderType + encase::ShaderSize + Clone + Send + Sync + 'static,
    O: ShaderType + encase::ShaderSize + Default + Clone + Send + Sync + 'static,
    Vec<I>: encase::internal::WriteInto,
    Vec<O>: encase::internal::WriteInto,
    O: encase::internal::ReadFrom + encase::internal::CreateFrom,
{
    /// Create a new compute test.
    ///
    /// # Arguments
    ///
    /// - `shader_path`: Asset path to the WGSL compute shader (e.g. `"shaders/my_test.wgsl"`).
    /// - `inputs`: The input data to upload. One element per invocation.
    ///
    /// The shader must read from `@binding(0)` (storage, read) and write to
    /// `@binding(1)` (storage, `read_write`). Use [`with_uniform`](Self::with_uniform)
    /// to prepend a uniform buffer, shifting storage bindings to 1 and 2.
    pub fn new(shader_path: impl Into<String>, inputs: Vec<I>) -> Self {
        Self {
            shader_path: shader_path.into(),
            inputs,
            workgroup_size: 64,
            uniform_bytes: None,
            result_channel: Arc::new(Mutex::new(None)),
            entry_point: "main".to_string(),
            timeout: std::time::Duration::from_secs(5),
        }
    }

    /// Add a uniform buffer at `@binding(0)`.
    ///
    /// This shifts the input storage buffer to `@binding(1)` and the output storage
    /// buffer to `@binding(2)`.
    ///
    /// The value is serialized with `encase::UniformBuffer` for GPU-compatible layout.
    #[must_use]
    pub fn with_uniform<U: ShaderType + encase::internal::WriteInto>(mut self, uniform: U) -> Self {
        let mut buffer = encase::UniformBuffer::new(Vec::new());
        buffer.write(&uniform).expect("Failed to serialize uniform");
        self.uniform_bytes = Some(buffer.into_inner());
        self
    }

    /// Override the compute workgroup size (default: 64).
    ///
    /// Must match the `@workgroup_size(N)` in your WGSL shader.
    #[must_use]
    pub fn with_workgroup_size(mut self, size: u32) -> Self {
        self.workgroup_size = size;
        self
    }

    /// Override the shader entry point (default: `"main"`).
    #[must_use]
    pub fn with_entry_point(mut self, entry_point: impl Into<String>) -> Self {
        self.entry_point = entry_point.into();
        self
    }

    /// Override the timeout (default: 5 seconds).
    ///
    /// If the compute shader does not produce results within this duration,
    /// the test panics with a diagnostic message including any pipeline
    /// compilation errors.
    #[must_use]
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Run the test and return the GPU output.
    ///
    /// Blocks until the headless Bevy app completes. The app loads the shader,
    /// dispatches the compute work, reads back results, and exits.
    ///
    /// # Panics
    ///
    /// Panics if the GPU readback does not complete within the timeout (default:
    /// 5 seconds). The panic message includes diagnostic information about the
    /// pipeline state. Override the timeout with [`with_timeout`](Self::with_timeout).
    ///
    /// Also panics if the Bevy app fails to start or no GPU is available.
    #[must_use]
    pub fn run(self) -> Vec<O> {
        let reader = Arc::clone(&self.result_channel);
        run_compute_test(self);
        reader.lock().unwrap().take().expect(
            "GPU test did not produce results — check shader compilation and GPU availability",
        )
    }
}
