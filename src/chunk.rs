use enum_iterator::Sequence;
use glam::{DVec3, IVec3, UVec3};
use noise::NoiseFn;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use wgpu::{RenderPass, util::DeviceExt};

use crate::frustum::{Aabb, Frustum};

pub const CHUNK_SIZE: usize = 32;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// mapped to 0b000uuuuuuuunnnxxxxxxyyyyyyzzzzzz
    pub packed_data: u32,
    pub voxel_position: IVec3,
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Uint32, 1 => Sint32x3];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, TryFromPrimitive, IntoPrimitive, Sequence)]
#[repr(usize)]
pub enum Block {
    AIR = 0,
    DIRT = 1,
    GRASS = 2,
    STONE = 3,
    LOG = 4,
    PLANK = 5,
    LEAVES = 6,
}

impl Block {
    fn get_uv(&self, side: usize) -> u8 {
        match self {
            Self::AIR => 0,
            Self::GRASS => {
                if side < 4 {
                    return 0;
                }
                if side == 5 {
                    return 1;
                } else {
                    return 2;
                }
            }
            Self::DIRT => 2,
            Self::STONE => 3,
            Self::LOG => {
                if side < 4 {
                    return 5;
                } else {
                    return 4;
                }
            }
            Self::PLANK => 6,
            Self::LEAVES => 7,
        }
    }
}

pub struct Chunk {
    pub position: IVec3,
    pub world_position: IVec3,
    pub blocks: [Block; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE],
    pub is_empty: bool,
    pub bounding_box: Aabb,
    pub mesh: Option<ChunkMesh>,
}

impl Chunk {
    const CAVE_NOISE_SCALE: f64 = 30.0;
    const HILL_NOISE_SCALE: f64 = 50.0;

    pub fn new(position: IVec3) -> Self {
        let world_position = position * CHUNK_SIZE as i32;

        let mut chunk = Self {
            position,
            world_position,
            blocks: [Block::AIR; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE],
            is_empty: true,
            bounding_box: Aabb::new(
                world_position.as_vec3(),
                world_position.as_vec3() + CHUNK_SIZE as f32,
            ),
            mesh: None,
        };

        let noise = noise::Perlin::new(0);

        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    let voxel_position = IVec3::new(x as i32, y as i32, z as i32) + world_position;
                    let mut noise_pos = DVec3::new(
                        voxel_position.x as f64,
                        voxel_position.y as f64,
                        voxel_position.z as f64,
                    );
                    noise_pos += 0.5;

                    if world_position.y < 0 {
                        noise_pos /= Self::CAVE_NOISE_SCALE;
                        let val = ((noise.get([noise_pos.x, noise_pos.y, noise_pos.z]) + 1.0) / 2.0
                            * (CHUNK_SIZE - 1) as f64) as u32;
                        if val > 16 {
                            chunk.blocks[CHUNK_SIZE * CHUNK_SIZE * z + CHUNK_SIZE * y + x] =
                                Block::STONE;
                            chunk.is_empty = false;
                        }
                    } else {
                        noise_pos /= Self::HILL_NOISE_SCALE;
                        let val = ((noise.get([noise_pos.x, noise_pos.z]) + 1.0) / 2.0
                            * CHUNK_SIZE as f64) as u32;
                        if val == voxel_position.y as u32 {
                            chunk.blocks[CHUNK_SIZE * CHUNK_SIZE * z + CHUNK_SIZE * y + x] =
                                Block::GRASS;
                            chunk.is_empty = false;
                        } else if val > voxel_position.y as u32 {
                            chunk.blocks[CHUNK_SIZE * CHUNK_SIZE * z + CHUNK_SIZE * y + x] =
                                Block::DIRT;
                            chunk.is_empty = false;
                        }
                    }
                }
            }
        }

        chunk
    }

    pub fn set_block(&mut self, position: IVec3, block: Block) -> bool {
        let index = CHUNK_SIZE * CHUNK_SIZE * position.z as usize
            + CHUNK_SIZE * position.y as usize
            + position.x as usize;

        if self.blocks[index] != block {
            self.blocks[index] = block;
            self.is_empty = self.blocks.iter().all(|b| *b == Block::AIR);
            return true;
        }

        return false;
    }

    pub fn generate_mesh(&self, neighbors: [Option<&Chunk>; 6]) -> (Option<ChunkMeshData>, bool) {
        if self.is_empty {
            return (None, false);
        }

        let mut temp_vertices = Vec::new();
        let mut temp_indices = Vec::new();

        let mut missing_neighors = false;

        let mut vertex_count = 0;
        #[allow(unused)]
        let mut index_count = 0;

        const CUBE_VERTICES: [UVec3; 8] = [
            UVec3::new(0, 0, 0),
            UVec3::new(1, 0, 0),
            UVec3::new(1, 1, 0),
            UVec3::new(0, 1, 0),
            UVec3::new(0, 0, 1),
            UVec3::new(1, 0, 1),
            UVec3::new(1, 1, 1),
            UVec3::new(0, 1, 1),
        ];

        const FACE_INDICES: [[u32; 4]; 6] = [
            [0, 1, 2, 3], // Front
            [5, 4, 7, 6], // Back
            [4, 0, 3, 7], // Left
            [1, 5, 6, 2], // Right
            [4, 5, 1, 0], // Bottom
            [3, 2, 6, 7], // Top
        ];

        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    let block = self.blocks[CHUNK_SIZE * CHUNK_SIZE * z + CHUNK_SIZE * y + x];
                    if block == Block::AIR {
                        continue;
                    }

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
                            _ => unreachable!(),
                        }

                        let mut render_face = true;

                        if nx >= 0
                            && nx < CHUNK_SIZE as i32
                            && ny >= 0
                            && ny < CHUNK_SIZE as i32
                            && nz >= 0
                            && nz < CHUNK_SIZE as i32
                        {
                            if self.blocks[CHUNK_SIZE * CHUNK_SIZE * nz as usize
                                + CHUNK_SIZE * ny as usize
                                + nx as usize]
                                != Block::AIR
                            {
                                render_face = false;
                            }
                        } else {
                            // the voxel we wanna check is in a neighboring chunk
                            if let Some(chunk) = neighbors[face] {
                                let mut pos = IVec3::new(nx, ny, nz);
                                pos %= CHUNK_SIZE as i32;
                                pos = pos.map(|v| if v < 0 { v + CHUNK_SIZE as i32 } else { v });

                                if chunk.blocks[CHUNK_SIZE * CHUNK_SIZE * pos.z as usize
                                    + CHUNK_SIZE * pos.y as usize
                                    + pos.x as usize]
                                    != Block::AIR
                                {
                                    render_face = false;
                                }
                            } else {
                                // the neighbor hasnt loaded yet so we'll need to remesh this later
                                missing_neighors = true;
                            }
                        }

                        if !render_face {
                            continue;
                        }

                        let base_index = vertex_count as u32;

                        for i in 0..4 {
                            let position = CUBE_VERTICES[FACE_INDICES[face][i] as usize]
                                + UVec3::new(x as u32, y as u32, z as u32);

                            let position =
                                (position.x << 12) | (position.y << 6) | (position.z << 0);

                            let normal_position = ((face as u32) << 18) | position;

                            let uv = block.get_uv(face);

                            let packed_data = ((uv as u32) << 21) | normal_position;

                            let voxel_position =
                                self.world_position + IVec3::new(x as i32, y as i32, z as i32);

                            let v = Vertex {
                                packed_data,
                                voxel_position,
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

        (
            Some(ChunkMeshData {
                vertices: temp_vertices,
                indices: temp_indices,
            }),
            missing_neighors,
        )
    }

    pub fn load_mesh(&mut self, mesh_data: ChunkMeshData, device: &wgpu::Device) {
        if mesh_data.vertices.len() == 0 || mesh_data.indices.len() == 0 {
            self.mesh = None;
            return;
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(mesh_data.vertices.as_slice()),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(mesh_data.indices.as_slice()),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.mesh = Some(ChunkMesh {
            vertex_count: mesh_data.vertices.len() as u32,
            index_count: mesh_data.indices.len() as u32,
            vertex_buffer,
            index_buffer,
        });
    }

    pub fn render(&self, render_pass: &mut RenderPass, frustum: &Frustum) -> bool {
        if let Some(mesh) = &self.mesh {
            if frustum.contains_aabb(&self.bounding_box) {
                render_pass.set_push_constants(
                    wgpu::ShaderStages::VERTEX,
                    0,
                    bytemuck::cast_slice(&self.world_position.to_array()),
                );
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);

                return true;
            }
        }

        return false;
    }
}

impl Chunk {
    pub fn world_to_chunk_pos(world_position: IVec3) -> IVec3 {
        (world_position.as_vec3() / CHUNK_SIZE as f32)
            .floor()
            .as_ivec3()
    }

    pub fn world_to_local_pos(world_position: IVec3) -> IVec3 {
        let mut pos = world_position % CHUNK_SIZE as i32;
        if pos.x < 0 {
            pos.x += CHUNK_SIZE as i32;
        }
        if pos.y < 0 {
            pos.y += CHUNK_SIZE as i32;
        }
        if pos.z < 0 {
            pos.z += CHUNK_SIZE as i32;
        }

        pos
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
