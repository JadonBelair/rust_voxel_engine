use std::{sync::Arc, time::Instant};

use camera::{Camera, CameraController, CameraUniform, Projection};
use chunk::{Block, CHUNK_SIZE, Vertex};
use chunk_manager::ChunkManager;
use frustum::Frustum;
use glam::{IVec3, Vec3};
use texture::Texture;
use wgpu::{util::DeviceExt, PresentMode};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

mod camera;
mod chunk;
mod chunk_manager;
mod frustum;
mod texture;

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    window: Arc<Window>,
    is_cursor_visible: bool,

    chunk_manager: ChunkManager,

    camera: Camera,
    projection: Projection,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    depth_texture: Texture,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::POLYGON_MODE_LINE
                    | wgpu::Features::POLYGON_MODE_POINT
                    | wgpu::Features::PUSH_CONSTANTS,
                required_limits: wgpu::Limits {
                    max_push_constant_size: 12,
                    ..wgpu::Limits::downlevel_defaults()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let present_mode = surface_caps
            .present_modes
            .iter()
            .find(|f| **f == PresentMode::Fifo)
            .copied()
            .unwrap_or(surface_caps.present_modes[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let camera = Camera::new(Vec3::new(0.0, CHUNK_SIZE as f32, 0.0), 0.0, 0.0);
        let projection = Projection::new(size.width, size.height, 60.0, 0.1, 1000.0);
        let camera_controller = CameraController::new(10.0, 0.1);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/shader.wgsl"));
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::VERTEX,
                    range: 0..std::mem::size_of::<[f32; 3]>() as u32,
                }],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                bias: wgpu::DepthBiasState::default(),
                stencil: wgpu::StencilState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let mut chunk_manager = ChunkManager::new(10);
        chunk_manager.update_around(IVec3::ZERO);

        let depth_texture = Texture::create_depth_texture(&device, size.width, size.height, Some("Depth Texture"));

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            render_pipeline,
            window,
            is_cursor_visible: false,

            chunk_manager,

            camera,
            projection,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,

            depth_texture,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.projection.resize(width, height);
            self.is_surface_configured = true;
            self.depth_texture = Texture::create_depth_texture(&self.device, width, height, Some("Depth Texture"));
        }
    }

    pub fn handle_mouse_button(
        &mut self,
        _event_loop: &ActiveEventLoop,
        button: MouseButton,
        is_pressed: bool,
    ) {
        match (button, is_pressed) {
            (MouseButton::Left, true) => {
                if let Some((pos, _normal)) = self.chunk_manager.ray_cast(
                    self.camera.position,
                    self.camera.pitch,
                    self.camera.yaw,
                    10.0,
                ) {
                    self.chunk_manager.set_block(pos, Block::AIR);
                }
            }
            (MouseButton::Right, true) => {
                if let Some((pos, normal)) = self.chunk_manager.ray_cast(
                    self.camera.position,
                    self.camera.pitch,
                    self.camera.yaw,
                    10.0,
                ) {
                    self.chunk_manager.set_block(pos + normal, Block::DIRT);
                }
            }
            _ => (),
        }
    }

    pub fn handle_key(&mut self, _event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        if !self.camera_controller.handle_key(code, is_pressed) {
            match (code, is_pressed) {
                (KeyCode::Escape, true) => {
                    let grab_mode = if !self.is_cursor_visible {
                        CursorGrabMode::None
                    } else {
                        CursorGrabMode::Locked
                    };

                    self.window.set_cursor_grab(grab_mode).unwrap();
                    self.window.set_cursor_visible(!self.is_cursor_visible);
                    self.is_cursor_visible = !self.is_cursor_visible;
                }
                _ => (),
            }
        }
    }

    pub fn update(&mut self, dt: std::time::Duration) {
        let prev_chunk = (self.camera.position / CHUNK_SIZE as f32).floor();
        self.camera_controller.update_camera(&mut self.camera, dt);
        let new_chunk = (self.camera.position / CHUNK_SIZE as f32).floor();

        if prev_chunk != new_chunk {
            self.chunk_manager.update_around(new_chunk.as_ivec3());
        }

        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        self.chunk_manager.build_chunk_data_in_queue(15);
        self.chunk_manager
            .build_chunk_mesh_in_queue(8, &self.device);
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            let frustum = Frustum::from_camera(&self.camera, &self.projection);
            self.chunk_manager.render(&mut render_pass, &frustum);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub struct App {
    state: Option<State>,
    last_time: Instant,
}

impl App {
    pub fn new() -> Self {
        Self {
            state: None,
            last_time: Instant::now(),
        }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes();
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        window.set_cursor_grab(CursorGrabMode::Locked).unwrap();
        window.set_cursor_visible(false);
        self.state = Some(pollster::block_on(State::new(window)).unwrap());
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: State) {
        self.state = Some(event);
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let state = match &mut self.state {
            Some(state) => state,
            None => return,
        };

        match event {
            DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                state.camera_controller.handle_mouse(dx, dy);
            }
            _ => (),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(state) => state,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                let dt = self.last_time.elapsed();
                self.last_time = Instant::now();
                state.update(dt);

                match state.render() {
                    Ok(_) => (),
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size.width, size.height);
                    }
                    Err(e) => log::error!("unable to render {}", e),
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            WindowEvent::MouseWheel { delta, .. } => state.camera_controller.handle_scroll(&delta),
            WindowEvent::MouseInput {
                state: mouse_state,
                button,
                ..
            } => state.handle_mouse_button(event_loop, button, mouse_state.is_pressed()),
            _ => (),
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}

fn main() {
    run().unwrap();
}
