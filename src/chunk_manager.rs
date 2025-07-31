use std::collections::{HashMap, HashSet, VecDeque};

use glam::{IVec3, Vec3};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    chunk::{Block, Chunk, ChunkMeshData, CHUNK_SIZE},
    frustum::Frustum,
};

pub struct ChunkManager {
    pub chunk_map: HashMap<IVec3, Chunk>,
    pub chunk_data_load_queue: VecDeque<IVec3>,
    pub chunk_mesh_load_queue: VecDeque<IVec3>,
    pub chunk_mesh_reload_queue: HashSet<IVec3>,
    pub chunk_neighbor_loaded_queue: HashSet<IVec3>,
    pub chunks_with_missing_neighbors: HashSet<IVec3>,
    pub render_distance: i32,
}

impl ChunkManager {
    pub fn new(render_distance: i32) -> Self {
        Self {
            chunk_map: HashMap::new(),
            chunk_data_load_queue: VecDeque::new(),
            chunk_mesh_load_queue: VecDeque::new(),
            chunk_mesh_reload_queue: HashSet::new(),
            chunk_neighbor_loaded_queue: HashSet::new(),
            chunks_with_missing_neighbors: HashSet::new(),
            render_distance,
        }
    }

    pub fn get_block(&self, pos: IVec3) -> Option<Block> {
        let chunk_pos = Chunk::world_to_chunk_pos(pos);

        if let Some(chunk) = self.chunk_map.get(&chunk_pos) {
            let inner_pos = Chunk::world_to_local_pos(pos);

            return Some(
                chunk.blocks[CHUNK_SIZE * CHUNK_SIZE * inner_pos.z as usize
                    + CHUNK_SIZE * inner_pos.y as usize
                    + inner_pos.x as usize],
            );
        }

        None
    }

    pub fn set_block(&mut self, position: IVec3, block: Block) {
        let chunk_pos = Chunk::world_to_chunk_pos(position);
        let inner_pos = Chunk::world_to_local_pos(position);

        if let Some(chunk) = self.chunk_map.get_mut(&chunk_pos) {
            if chunk.set_block(inner_pos, block) {
                self.chunk_mesh_reload_queue.insert(chunk_pos);
                for dir in &[
                    IVec3::NEG_X,
                    IVec3::X,
                    IVec3::NEG_Y,
                    IVec3::Y,
                    IVec3::NEG_Z,
                    IVec3::Z,
                ] {
                    let neighbor_pos = chunk_pos + *dir;
                    if self.chunk_map.contains_key(&neighbor_pos) {
                        self.chunk_mesh_reload_queue.insert(neighbor_pos);
                    }
                }
            }
        }
    }

    pub fn ray_cast(
        &self,
        origin: Vec3,
        yaw: f32,
        pitch: f32,
        max_distance: f32,
    ) -> Option<(IVec3, IVec3)> {
        let (yaw_sin, yaw_cos) = yaw.sin_cos();
        let (pitch_sin, pitch_cos) = pitch.sin_cos();

        let direction = Vec3::new(yaw_cos * pitch_cos, pitch_sin, yaw_sin * pitch_cos);

        let step = direction.signum();
        let t_delta = (1.0 / direction).abs().min(Vec3::splat(f32::MAX));

        let mut voxel = origin.floor().as_ivec3();

        let mut t_max = (voxel.as_vec3() + step.max(Vec3::ZERO) - origin) / direction;
        t_max = t_max.max(Vec3::splat(0.0));

        let mut traveled = 0.0;
        let mut normal = IVec3::ZERO;

        while traveled < max_distance {
            if let Some(block) = self.get_block(voxel) {
                if !matches!(block, Block::AIR) {
                    return Some((voxel, normal));
                }
            }

            if t_max.x < t_max.y {
                if t_max.x < t_max.z {
                    voxel.x += step.x as i32;
                    traveled = t_max.x;
                    t_max.x += t_delta.x;
                    normal = IVec3::new(-step.x as i32, 0, 0);
                } else {
                    voxel.z += step.z as i32;
                    traveled = t_max.z;
                    t_max.z += t_delta.z;
                    normal = IVec3::new(0, 0, -step.z as i32);
                }
            } else if t_max.y < t_max.z {
                voxel.y += step.y as i32;
                traveled = t_max.y;
                t_max.y += t_delta.y;
                normal = IVec3::new(0, -step.y as i32, 0);
            } else {
                voxel.z += step.z as i32;
                traveled = t_max.z;
                t_max.z += t_delta.z;
                normal = IVec3::new(0, 0, -step.z as i32);
            }
        }

        None
    }

    pub fn build_chunk_data_in_queue(&mut self, amount: usize) {
        let chunks = (0..amount)
            .filter_map(|_| self.chunk_data_load_queue.pop_front())
            .collect::<Vec<IVec3>>()
            .into_par_iter()
            .map(Chunk::new)
            .collect::<Vec<Chunk>>();

        for chunk in chunks {
            for dir in [
                IVec3::NEG_X,
                IVec3::X,
                IVec3::NEG_Y,
                IVec3::Y,
                IVec3::NEG_Z,
                IVec3::Z,
            ] {
                let neighbor_pos = chunk.position + dir;
                if self.chunk_map.contains_key(&neighbor_pos) {
                    self.chunks_with_missing_neighbors.insert(neighbor_pos);
                    if !self.chunk_mesh_load_queue.contains(&neighbor_pos)
                        & !self.chunk_mesh_reload_queue.contains(&neighbor_pos) {
                        self.chunk_neighbor_loaded_queue.insert(neighbor_pos);
                    }
                }
            }

            if !chunk.is_empty
                & !self.chunk_mesh_load_queue.contains(&chunk.position)
                & !self.chunk_mesh_reload_queue.contains(&chunk.position)
                & !self.chunk_neighbor_loaded_queue.contains(&chunk.position)
            {
                self.chunk_mesh_load_queue.push_back(chunk.position);
            }

            self.chunk_map.insert(chunk.position, chunk);
        }
    }

    pub fn build_chunk_mesh_in_queue(&mut self, amount: usize, device: &wgpu::Device) {
        let reload_tasks = (0..amount)
            .filter_map(|_| {
                if let Some(&pos) = self.chunk_mesh_reload_queue.iter().next() {
                    self.chunk_mesh_reload_queue.remove(&pos);
                    Some(pos)
                } else {
                    None
                }
            })
            .collect::<Vec<IVec3>>();

        let all_tasks = (0..amount.saturating_sub(reload_tasks.len()))
            .filter_map(|_| self.chunk_mesh_load_queue.pop_front())
            .chain(reload_tasks)
            .collect::<Vec<IVec3>>();

        let neighbor_changed_tasks = (0..amount.saturating_sub(all_tasks.len()))
            .filter_map(|_| {
                if let Some(&pos) = self.chunk_neighbor_loaded_queue.iter().next() {
                    self.chunk_neighbor_loaded_queue.remove(&pos);
                    Some(pos)
                } else {
                    None
                }
            })
            .collect::<Vec<IVec3>>();

        let all_tasks = all_tasks
            .iter()
            .chain(&neighbor_changed_tasks)
            .collect::<Vec<&IVec3>>();

        let meshes = all_tasks
            .into_par_iter()
            .map(|&position| {
                let neighbors = [
                    self.chunk_map.get(&(position + IVec3::NEG_Z)), // Front
                    self.chunk_map.get(&(position + IVec3::Z)),     // Back
                    self.chunk_map.get(&(position + IVec3::NEG_X)), // Left
                    self.chunk_map.get(&(position + IVec3::X)),     // Right
                    self.chunk_map.get(&(position + IVec3::NEG_Y)), // Bottom
                    self.chunk_map.get(&(position + IVec3::Y)),     // Top
                ];

                let chunk = self.chunk_map.get(&position);
                let (mesh, missing_neighbors) = chunk
                    .map(|chunk| chunk.generate_mesh(neighbors))
                    .unwrap_or((None, false));

                (position, mesh, missing_neighbors)
            })
            .collect::<Vec<(IVec3, Option<ChunkMeshData>, bool)>>();

        for (pos, mesh, missing_neighbors) in meshes {
            if missing_neighbors {
                self.chunks_with_missing_neighbors.insert(pos);
            } else {
                self.chunks_with_missing_neighbors.remove(&pos);
            }

            if let Some(chunk) = self.chunk_map.get_mut(&pos) {
                if let Some(mesh) = mesh {
                    chunk.load_mesh(mesh, device);
                } else if chunk.is_empty {
                    chunk.mesh = None;
                }
            } else {
                self.chunk_data_load_queue.push_back(pos);
            }
        }
    }

    pub fn update_around(&mut self, position: IVec3) {
        self.chunk_data_load_queue.retain(|chunk_position| {
            chunk_position.x <= position.x + self.render_distance as i32
                && chunk_position.x >= position.x - self.render_distance as i32
                && chunk_position.y <= position.y + self.render_distance as i32
                && chunk_position.y >= position.y - self.render_distance as i32
                && chunk_position.z <= position.z + self.render_distance as i32
                && chunk_position.z >= position.z - self.render_distance as i32
        });

        self.chunk_mesh_load_queue.retain(|chunk_position| {
            chunk_position.x <= position.x + self.render_distance as i32
                && chunk_position.x >= position.x - self.render_distance as i32
                && chunk_position.y <= position.y + self.render_distance as i32
                && chunk_position.y >= position.y - self.render_distance as i32
                && chunk_position.z <= position.z + self.render_distance as i32
                && chunk_position.z >= position.z - self.render_distance as i32
        });

        self.chunk_map.retain(|_, chunk| {
            chunk.position.x <= position.x + self.render_distance as i32
                && chunk.position.x >= position.x - self.render_distance as i32
                && chunk.position.y <= position.y + self.render_distance as i32
                && chunk.position.y >= position.y - self.render_distance as i32
                && chunk.position.z <= position.z + self.render_distance as i32
                && chunk.position.z >= position.z - self.render_distance as i32
        });

        self.chunk_neighbor_loaded_queue.retain(|chunk_position| {
            chunk_position.x <= position.x + self.render_distance as i32
                && chunk_position.x >= position.x - self.render_distance as i32
                && chunk_position.y <= position.y + self.render_distance as i32
                && chunk_position.y >= position.y - self.render_distance as i32
                && chunk_position.z <= position.z + self.render_distance as i32
                && chunk_position.z >= position.z - self.render_distance as i32
        });

        self.chunks_with_missing_neighbors.retain(|chunk_position| {
            chunk_position.x <= position.x + self.render_distance as i32
                && chunk_position.x >= position.x - self.render_distance as i32
                && chunk_position.y <= position.y + self.render_distance as i32
                && chunk_position.y >= position.y - self.render_distance as i32
                && chunk_position.z <= position.z + self.render_distance as i32
                && chunk_position.z >= position.z - self.render_distance as i32
        });

        for x in -self.render_distance..=self.render_distance {
            for y in -self.render_distance..=self.render_distance {
                for z in -self.render_distance..=self.render_distance {
                    let chunk_pos = IVec3::new(x, y, z) + position;
                    if !self.chunk_map.contains_key(&chunk_pos)
                        && !self.chunk_data_load_queue.contains(&chunk_pos)
                    {
                        self.chunk_data_load_queue.push_back(chunk_pos);
                    }
                }
            }
        }

        self.chunk_data_load_queue
            .make_contiguous()
            .sort_by(|a, b| {
                (a - position)
                    .length_squared()
                    .partial_cmp(&(b - position).length_squared())
                    .unwrap()
            });
    }

    pub fn render(&self, render_pass: &mut wgpu::RenderPass, frustum: &Frustum) {
        let mut count = 0;
        for chunk in self.chunk_map.values() {
            if chunk.render(render_pass, frustum) {
                count += 1;
            }
        }
        println!(
            "{}/{}\t{}\t{}\t{}",
            count,
            self.chunk_map.len(),
            self.chunk_data_load_queue.len(),
            self.chunk_mesh_load_queue.len() + self.chunk_mesh_reload_queue.len() + self.chunk_neighbor_loaded_queue.len(),
            self.chunks_with_missing_neighbors.len(),
        );
    }
}
