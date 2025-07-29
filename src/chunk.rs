use glam::Vec3;
use noise::NoiseFn;
use rand::Rng;
use wgpu::{util::DeviceExt, RenderPass};

use crate::frustum::{Aabb, Frustum};

const DO_3D: bool = true;
pub const CHUNK_SIZE: usize = 32;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub color: Vec3,
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Block {
    AIR = 0,
    DIRT = 1,
    GRASS = 2,
}

pub struct Chunk {
    pub position: Vec3,
    pub blocks: [Block; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE],
    pub mesh: Option<ChunkMesh>,
}

impl Chunk {
    const NOISE_SCALE: f64 = 20.0;

    pub fn new(position: Vec3) -> Self {
        let mut chunk = Self {
            position,
            blocks: [Block::AIR; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE],
            mesh: None,
        };

        let world_space = position * CHUNK_SIZE as f32;

        let noise = noise::Perlin::new(0);

        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    let voxel_position = Vec3::new(x as f32, y as f32, z as f32) + world_space;
                    if DO_3D {
                        let val = ((noise.get([voxel_position.x as f64 / Self::NOISE_SCALE + 0.5, voxel_position.y as f64 / Self::NOISE_SCALE + 0.5, voxel_position.z as f64 / Self::NOISE_SCALE + 0.5]) + 1.0) / 2.0 * CHUNK_SIZE as f64) as u32;
                        if val > 16 {
                            chunk.blocks[CHUNK_SIZE * CHUNK_SIZE * z + CHUNK_SIZE * y + x] = Block::GRASS;
                        }
                    } else {
                        let val = ((noise.get([voxel_position.x as f64 / Self::NOISE_SCALE + 0.5, voxel_position.z as f64 / Self::NOISE_SCALE + 0.5]) + 1.0) / 2.0 * CHUNK_SIZE as f64) as u32;
                        if val == voxel_position.y as u32 {
                            chunk.blocks[CHUNK_SIZE * CHUNK_SIZE * z + CHUNK_SIZE * y + x] = Block::GRASS;
                        } else if val > voxel_position.y as u32 {
                            chunk.blocks[CHUNK_SIZE * CHUNK_SIZE * z + CHUNK_SIZE * y + x] = Block::DIRT;
                        }
                    }
                }
            }
        }

        chunk
    }

    pub fn generate_mesh(&mut self) -> ChunkMeshData {
        let mut temp_vertices = Vec::new();
        let mut temp_indices= Vec::new();

        let mut vertex_count = 0;
        #[allow(unused)]
        let mut index_count = 0;

        const CUBE_VERTICES: [Vec3; 8] = [
            Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 1.0), Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.0, 1.0, 1.0)
        ];

        const FACE_NORMALS: [Vec3; 6] = [
            Vec3::new( 0.0,  0.0, -1.0), // Front
            Vec3::new( 0.0,  0.0,  1.0), // Back
            Vec3::new(-1.0,  0.0,  0.0), // Left
            Vec3::new( 1.0,  0.0,  0.0), // Right
            Vec3::new( 0.0, -1.0,  0.0), // Bottom
            Vec3::new( 0.0,  1.0,  0.0), // Top
        ];

        const FACE_INDICES: [[u32; 4]; 6] = [
            [0, 1, 2, 3],
            [5, 4, 7, 6],
            [4, 0, 3, 7],
            [1, 5, 6, 2],
            [4, 5, 1, 0],
            [3, 2, 6, 7],
        ];

        let mut rng = rand::rng();

        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    let block = self.blocks[CHUNK_SIZE * CHUNK_SIZE * z + CHUNK_SIZE * y + x];
                    if block == Block::AIR { continue; }

                    let shade = Vec3::splat(rng.random_range((0.8)..(1.2)));
                    // let color = Vec3::new(rng.random_range(0.0..1.0), rng.random_range(0.0..1.0), rng.random_range(0.0..1.0));

                    for face in 0..6 {
                        let mut nx = x as i32;
                        let mut ny = y as i32;
                        let mut nz = z as i32;

                        match face {
                            0 => nz -= 1,
                            1 => nz += 1,
                            2 => nx -= 1,
                            3 => nx += 1,
                            4 => ny -= 1,
                            5 => ny += 1,
                            _ => unreachable!()
                        }

                        let mut render_face = true;

                        if nx >= 0 && nx < CHUNK_SIZE as i32 &&
                            ny >= 0 && ny < CHUNK_SIZE as i32 &&
                            nz >= 0 && nz < CHUNK_SIZE as i32 {

                            if self.blocks[CHUNK_SIZE * CHUNK_SIZE * nz as usize + CHUNK_SIZE * ny as usize + nx as usize] != Block::AIR {
                                render_face = false;
                            }
                        }

                        if !render_face { continue; }

                        let base_index = vertex_count as u32;

                        for i in 0..4 {
                            let position = CUBE_VERTICES[FACE_INDICES[face][i] as usize] + Vec3::new(x as f32, y as f32, z as f32) + (self.position * CHUNK_SIZE as f32);
                            let normal = FACE_NORMALS[face];
                            let color = if block == Block::DIRT {
                                Vec3::new(0.36, 0.25, 0.125)
                            } else if block == Block::GRASS {
                                Vec3::new(0.0, 0.57, 0.0)
                            } else {
                                unreachable!();
                            } * shade;

                            let v = Vertex {
                                position,
                                normal,
                                color,
                            };

                            temp_vertices.push(v);
                            vertex_count += 1;
                        }

                        temp_indices.push(base_index);
                        temp_indices.push(base_index + 1);
                        temp_indices.push(base_index + 2);
                        temp_indices.push(base_index);
                        temp_indices.push(base_index + 2);
                        temp_indices.push(base_index + 3);
                        index_count += 6;
                    }
                }
            }
        }

        ChunkMeshData {
            vertices: temp_vertices,
            indices: temp_indices,
        }
    }

    pub fn load_mesh(&mut self, mesh_data: ChunkMeshData, device: &wgpu::Device) {
        if mesh_data.vertices.len() == 0 || mesh_data.indices.len() == 0 { return }

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(mesh_data.vertices.as_slice()),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(mesh_data.indices.as_slice()),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        self.mesh = Some(ChunkMesh {
            vertex_count: mesh_data.vertices.len() as u32,
            index_count: mesh_data.indices.len() as u32,
            vertex_buffer,
            index_buffer,
        });
    }

    pub fn render(&self, render_pass: &mut RenderPass, frustum: &Frustum) {
        if let Some(mesh) = &self.mesh {
            let aabb = Aabb {
                min: self.position * CHUNK_SIZE as f32,
                max: self.position * CHUNK_SIZE as f32 + CHUNK_SIZE as f32,
            };

            if frustum.contains_aabb(&aabb) {
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
    }
}

#[allow(unused)]
pub struct ChunkMesh {
    vertex_count: u32,
    index_count: u32,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
}

pub struct ChunkMeshData {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}
