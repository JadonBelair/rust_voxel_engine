use std::{collections::{HashMap, VecDeque}, sync::mpsc};

use glam::{IVec3, Vec3};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{chunk::Chunk, frustum::Frustum};

pub struct ChunkManager {
    pub chunk_map: HashMap<IVec3, Chunk>,
    pub chunk_load_queue: VecDeque<Vec3>,
    pub render_distance: i32,
}

impl ChunkManager {
    pub fn new(render_distance: i32) -> Self {
        Self {
            chunk_map: HashMap::new(),
            chunk_load_queue: VecDeque::new(),
            render_distance
        }
    }

    pub fn build_chunks_in_queue(&mut self, amount: usize, device: &wgpu::Device) {
        let (tx, rx) = mpsc::channel();
        (0..amount)
            .filter_map(|_| self.chunk_load_queue.pop_front())
            .collect::<Vec<Vec3>>()
            .into_par_iter()
            .for_each_with(tx, |s, position| {
                let mut chunk = Chunk::new(position);
                let mesh_data = chunk.generate_mesh();

                s.send((chunk, mesh_data)).unwrap();
            });

        for (mut chunk, mesh_data) in rx {
            chunk.load_mesh(mesh_data, device);
            self.chunk_map.insert(chunk.position.as_ivec3(), chunk);
        }
    }

    pub fn update_around(&mut self, position: Vec3) {
        self.chunk_load_queue.retain(|chunk_position| {
            chunk_position.x <= position.x + self.render_distance as f32 &&
            chunk_position.x >= position.x - self.render_distance as f32 &&
            chunk_position.y <= position.y + self.render_distance as f32 &&
            chunk_position.y >= position.y - self.render_distance as f32 &&
            chunk_position.z <= position.z + self.render_distance as f32 &&
            chunk_position.z >= position.z - self.render_distance as f32
        });

        self.chunk_map.retain(|_, chunk| {
            chunk.position.x <= position.x + self.render_distance as f32 &&
            chunk.position.x >= position.x - self.render_distance as f32 &&
            chunk.position.y <= position.y + self.render_distance as f32 &&
            chunk.position.y >= position.y - self.render_distance as f32 &&
            chunk.position.z <= position.z + self.render_distance as f32 &&
            chunk.position.z >= position.z - self.render_distance as f32
        });

        for x in -self.render_distance..self.render_distance {
            for y in -self.render_distance..self.render_distance {
                for z in -self.render_distance..self.render_distance {
                    let chunk_pos = Vec3::new(x as f32, y as f32, z as f32) + position;
                    if !self.chunk_map.contains_key(&chunk_pos.as_ivec3()) &&
                        !self.chunk_load_queue.contains(&chunk_pos) {
                        self.chunk_load_queue.push_back(chunk_pos);
                    }
                }
            }
        }
    }

    pub fn render(&self, render_pass: &mut wgpu::RenderPass, frustum: &Frustum) {
        for chunk in self.chunk_map.values() {
            chunk.render(render_pass, frustum);
        }
    }
}
