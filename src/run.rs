//! Internal implementation — headless Bevy app, compute pipeline, readback.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use bevy::{
    app::ScheduleRunnerPlugin,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
            encase, BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            BufferInitDescriptor, BufferUsages, CachedComputePipelineId, ComputePassDescriptor,
            ComputePipelineDescriptor, PipelineCache, ShaderStages, ShaderType,
        },
        renderer::{RenderContext, RenderDevice},
        storage::{GpuShaderStorageBuffer, ShaderStorageBuffer},
        Render, RenderApp, RenderSystems,
    },
    winit::WinitPlugin,
};

use crate::ComputeTest;

// ============================================================================
// Public entry point
// ============================================================================

pub(crate) fn run_compute_test<I, O>(test: ComputeTest<I, O>)
where
    I: ShaderType + encase::ShaderSize + Clone + Send + Sync + 'static,
    O: ShaderType + encase::ShaderSize + Default + Clone + Send + Sync + 'static,
    Vec<I>: encase::internal::WriteInto,
    Vec<O>: encase::internal::WriteInto,
    O: encase::internal::ReadFrom + encase::internal::CreateFrom,
{
    let result_arc = Arc::clone(&test.result_channel);

    #[allow(clippy::cast_possible_truncation)]
    let input_count = test.inputs.len() as u32;

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: bevy::window::ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>()
                .disable::<bevy::log::LogPlugin>()
                .add(ScheduleRunnerPlugin::run_loop(
                    std::time::Duration::from_millis(16),
                )),
        )
        .add_plugins(ComputeTestPlugin {
            shader_path: test.shader_path.clone(),
            entry_point: test.entry_point.clone(),
            has_uniform: test.uniform_bytes.is_some(),
            uniform_bytes: test.uniform_bytes.clone(),
        })
        .insert_resource(ResultChannel::<O>(result_arc))
        .insert_resource(TestConfig {
            input_count,
            workgroup_size: test.workgroup_size,
        })
        .insert_resource(SetupData::<I, O> {
            inputs: test.inputs,
            uniform_bytes: test.uniform_bytes,
            _marker: std::marker::PhantomData,
        })
        .add_systems(Startup, create_buffers::<I, O>)
        .add_systems(Update, poll_results::<O>)
        .run();
}

// ============================================================================
// Resources
// ============================================================================

/// Channel for passing results from the Bevy app back to the caller.
#[derive(Resource)]
struct ResultChannel<O: Send + Sync + 'static>(Arc<Mutex<Option<Vec<O>>>>);

/// Input data and config, consumed during Startup to create GPU buffers.
#[derive(Resource)]
struct SetupData<I, O> {
    inputs: Vec<I>,
    uniform_bytes: Option<Vec<u8>>,
    _marker: std::marker::PhantomData<O>,
}

/// Holds the GPU buffer handles, extracted to the render world.
#[derive(Resource, Clone, ExtractResource)]
struct TestBuffers {
    input_handle: Handle<ShaderStorageBuffer>,
    output_handle: Handle<ShaderStorageBuffer>,
    uniform_bytes: Option<Vec<u8>>,
}

/// Dispatch configuration, extracted to the render world.
#[derive(Resource, Clone, ExtractResource)]
struct TestConfig {
    input_count: u32,
    workgroup_size: u32,
}

/// Cached compute pipeline and bind group layout.
#[derive(Resource)]
struct TestPipeline {
    bind_group_layout: BindGroupLayoutDescriptor,
    pipeline_id: CachedComputePipelineId,
    has_uniform: bool,
}

/// The prepared bind group, recreated each frame by `prepare_bind_group`.
#[derive(Resource)]
struct TestBindGroup(BindGroup);

/// Stores the typed readback results.
#[derive(Resource)]
struct ReadbackResults<O>(Vec<O>);

/// Tracks the readback entity so we can despawn it after receiving results.
#[derive(Resource)]
struct ReadbackEntity(Entity);

// ============================================================================
// Startup: create GPU buffers from input data
// ============================================================================

fn create_buffers<I, O>(
    mut commands: Commands,
    setup: Res<SetupData<I, O>>,
    mut buffer_assets: ResMut<Assets<ShaderStorageBuffer>>,
) where
    I: ShaderType + encase::ShaderSize + Clone + Send + Sync + 'static,
    O: ShaderType + encase::ShaderSize + Default + Clone + Send + Sync + 'static,
    Vec<I>: encase::internal::WriteInto,
    Vec<O>: encase::internal::WriteInto,
    O: encase::internal::ReadFrom + encase::internal::CreateFrom,
{
    // Input buffer
    let mut input_buf = ShaderStorageBuffer::from(setup.inputs.clone());
    input_buf.buffer_description.usage |= BufferUsages::COPY_SRC;
    let input_handle = buffer_assets.add(input_buf);

    // Output buffer — filled with f32::MAX sentinel so we can detect when the
    // compute shader has actually written to it.
    let sentinel_outputs: Vec<O> = vec![O::default(); setup.inputs.len()];
    let mut output_buf = ShaderStorageBuffer::from(sentinel_outputs);
    output_buf.buffer_description.usage |= BufferUsages::COPY_SRC;

    // Overwrite the raw buffer data with sentinel values.
    if let Some(ref mut data) = output_buf.data {
        for chunk in data.chunks_exact_mut(4) {
            chunk.copy_from_slice(&f32::MAX.to_ne_bytes());
        }
    }

    let output_handle = buffer_assets.add(output_buf);

    // Readback entity — triggers on_readback_complete when the GPU copy finishes.
    let readback_entity = commands
        .spawn(Readback::buffer(output_handle.clone()))
        .observe(on_readback_complete::<O>)
        .id();
    commands.insert_resource(ReadbackEntity(readback_entity));

    commands.insert_resource(TestBuffers {
        input_handle,
        output_handle,
        uniform_bytes: setup.uniform_bytes.clone(),
    });
}

fn on_readback_complete<O>(trigger: On<ReadbackComplete>, mut commands: Commands)
where
    O: ShaderType + encase::ShaderSize + Default + Clone + Send + Sync + 'static,
    O: encase::internal::ReadFrom + encase::internal::CreateFrom,
{
    let data: Vec<O> = trigger.to_shader_type();
    commands.insert_resource(ReadbackResults(data));
}

// ============================================================================
// Update: poll for results and exit
// ============================================================================

fn poll_results<O>(
    mut commands: Commands,
    results: Option<Res<ReadbackResults<O>>>,
    readback: Option<Res<ReadbackEntity>>,
    channel: Res<ResultChannel<O>>,
    mut exit: MessageWriter<AppExit>,
) where
    O: Clone + Send + Sync + 'static,
{
    let Some(results) = results else { return };
    if results.0.is_empty() {
        return;
    }

    // Despawn the readback entity to prevent a second readback attempt
    // from firing into a closed channel after we exit.
    if let Some(readback) = readback {
        commands.entity(readback.0).despawn();
        commands.remove_resource::<ReadbackEntity>();
    }

    *channel.0.lock().unwrap() = Some(results.0.clone());
    exit.write(AppExit::Success);
}

// ============================================================================
// Plugin: compute pipeline + render graph
// ============================================================================

struct ComputeTestPlugin {
    shader_path: String,
    entry_point: String,
    has_uniform: bool,
    uniform_bytes: Option<Vec<u8>>,
}

impl Plugin for ComputeTestPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<TestBuffers>::default())
            .add_plugins(ExtractResourcePlugin::<TestConfig>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.add_systems(
            Render,
            prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
        );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(ComputeTestLabel, ComputeTestNode::default());
        render_graph.add_node_edge(ComputeTestLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let shader = app
            .world()
            .resource::<AssetServer>()
            .load(&self.shader_path);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        let pipeline_cache = render_app.world().resource::<PipelineCache>();

        let bind_group_layout = if self.has_uniform {
            let uniform_size = self.uniform_bytes.as_ref().map_or(16, |b| b.len() as u64);
            BindGroupLayoutDescriptor::new(
                "bevy_gpu_test_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        uniform_buffer_sized(
                            false,
                            Some(
                                std::num::NonZero::new(uniform_size)
                                    .expect("uniform size must be > 0"),
                            ),
                        ),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_sized(false, None),
                    ),
                ),
            )
        } else {
            BindGroupLayoutDescriptor::new(
                "bevy_gpu_test_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_sized(false, None),
                    ),
                ),
            )
        };

        let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("bevy_gpu_test_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: vec![],
            shader,
            shader_defs: vec![],
            entry_point: Some(self.entry_point.clone().into()),
            zero_initialize_workgroup_memory: false,
        });

        render_app.insert_resource(TestPipeline {
            bind_group_layout,
            pipeline_id,
            has_uniform: self.has_uniform,
        });
    }
}

// ============================================================================
// Render graph node
// ============================================================================

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ComputeTestLabel;

struct ComputeTestNode {
    dispatched: AtomicBool,
}

impl Default for ComputeTestNode {
    fn default() -> Self {
        Self {
            dispatched: AtomicBool::new(false),
        }
    }
}

impl render_graph::Node for ComputeTestNode {
    fn run<'w>(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), render_graph::NodeRunError> {
        if self.dispatched.load(Ordering::Relaxed) {
            return Ok(());
        }

        let Some(pipeline) = world.get_resource::<TestPipeline>() else {
            return Ok(());
        };
        let Some(bind_group) = world.get_resource::<TestBindGroup>() else {
            return Ok(());
        };
        let Some(config) = world.get_resource::<TestConfig>() else {
            return Ok(());
        };

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline_id)
        else {
            return Ok(());
        };

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, &bind_group.0, &[]);
        pass.set_pipeline(compute_pipeline);

        let workgroups = config.input_count.div_ceil(config.workgroup_size);
        pass.dispatch_workgroups(workgroups, 1, 1);

        self.dispatched.store(true, Ordering::Relaxed);

        Ok(())
    }
}

// ============================================================================
// Bind group preparation (runs in Render schedule)
// ============================================================================

#[allow(clippy::needless_pass_by_value)]
fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Option<Res<TestPipeline>>,
    buffers: Option<Res<TestBuffers>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    buffer_assets: Res<RenderAssets<GpuShaderStorageBuffer>>,
) {
    let Some(pipeline) = pipeline else { return };
    let Some(buffers) = buffers else { return };

    let Some(input_gpu) = buffer_assets.get(&buffers.input_handle) else {
        return;
    };
    let Some(output_gpu) = buffer_assets.get(&buffers.output_handle) else {
        return;
    };

    let bind_group_layout = pipeline_cache.get_bind_group_layout(&pipeline.bind_group_layout);

    let bind_group = if pipeline.has_uniform {
        let uniform_data = buffers
            .uniform_bytes
            .as_ref()
            .expect("uniform_bytes must be set when has_uniform is true");

        let uniform_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("bevy_gpu_test_uniform_buffer"),
            contents: uniform_data,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        render_device.create_bind_group(
            "bevy_gpu_test_bind_group",
            &bind_group_layout,
            &BindGroupEntries::sequential((
                uniform_buffer.as_entire_binding(),
                input_gpu.buffer.as_entire_binding(),
                output_gpu.buffer.as_entire_binding(),
            )),
        )
    } else {
        render_device.create_bind_group(
            "bevy_gpu_test_bind_group",
            &bind_group_layout,
            &BindGroupEntries::sequential((
                input_gpu.buffer.as_entire_binding(),
                output_gpu.buffer.as_entire_binding(),
            )),
        )
    };

    commands.insert_resource(TestBindGroup(bind_group));
}
