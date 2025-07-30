use std::{collections::{HashMap, VecDeque}, sync::mpsc};

use glam::IVec3;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{chunk::Chunk, frustum::Frustum};

pub struct ChunkManager {
    pub chunk_map: HashMap<IVec3, Chunk>,
    pub chunk_data_load_queue: VecDeque<IVec3>,
    pub chunk_mesh_load_queue: VecDeque<IVec3>,
    pub render_distance: i32,
}

impl ChunkManager {
    pub fn new(render_distance: i32) -> Self {
        Self {
            chunk_map: HashMap::new(),
            chunk_data_load_queue: VecDeque::new(),
            chunk_mesh_load_queue: VecDeque::new(),
            render_distance
        }
    }

    pub fn build_chunk_data_in_queue(&mut self, amount: usize) {
        let (tx, rx) = mpsc::channel();
        (0..amount)
            .filter_map(|_| self.chunk_data_load_queue.pop_front())
            .collect::<Vec<IVec3>>()
            .into_par_iter()
            .for_each_with(tx, |s, position| {
                let chunk = Chunk::new(position);
                s.send(chunk).unwrap();
            });

        for chunk in rx {
            if !chunk.is_empty {
                self.chunk_mesh_load_queue.push_back(chunk.position);
            }
            self.chunk_map.insert(chunk.position, chunk);
        }
    }

    pub fn build_chunk_mesh_in_queue(&mut self, amount: usize, device: &wgpu::Device) {
        let (tx, rx) = mpsc::channel();
        (0..amount)
            .filter_map(|_| self.chunk_mesh_load_queue.pop_front())
            .collect::<Vec<IVec3>>()
            .into_par_iter()
            .for_each_with(tx, |s, position| {
                let mesh = if let Some(chunk) = self.chunk_map.get(&position) {
                    let mesh = chunk.generate_mesh();
                    Some(mesh)
                } else {
                    None
                };

                s.send((position, mesh)).unwrap();
            });

        for (pos, mesh) in rx {
            match mesh {
                Some(mesh) => if let Some(chunk) = self.chunk_map.get_mut(&pos) {
                    chunk.load_mesh(mesh, device);
                },
                None => self.chunk_data_load_queue.push_back(pos),
            }
        }
    }

    pub fn update_around(&mut self, position: IVec3) {
        self.chunk_data_load_queue.retain(|chunk_position| {
            chunk_position.x <= position.x + self.render_distance as i32 &&
            chunk_position.x >= position.x - self.render_distance as i32 &&
            chunk_position.y <= position.y + self.render_distance as i32 &&
            chunk_position.y >= position.y - self.render_distance as i32 &&
            chunk_position.z <= position.z + self.render_distance as i32 &&
            chunk_position.z >= position.z - self.render_distance as i32
        });

        self.chunk_mesh_load_queue.retain(|chunk_position| {
            chunk_position.x <= position.x + self.render_distance as i32 &&
            chunk_position.x >= position.x - self.render_distance as i32 &&
            chunk_position.y <= position.y + self.render_distance as i32 &&
            chunk_position.y >= position.y - self.render_distance as i32 &&
            chunk_position.z <= position.z + self.render_distance as i32 &&
            chunk_position.z >= position.z - self.render_distance as i32
        });

        self.chunk_map.retain(|_, chunk| {
            chunk.position.x <= position.x + self.render_distance as i32 &&
            chunk.position.x >= position.x - self.render_distance as i32 &&
            chunk.position.y <= position.y + self.render_distance as i32 &&
            chunk.position.y >= position.y - self.render_distance as i32 &&
            chunk.position.z <= position.z + self.render_distance as i32 &&
            chunk.position.z >= position.z - self.render_distance as i32
        });

        for x in -self.render_distance..=self.render_distance {
            for y in -self.render_distance..=self.render_distance {
                for z in -self.render_distance..=self.render_distance {
                    let chunk_pos = IVec3::new(x, y, z) + position;
                    if !self.chunk_map.contains_key(&chunk_pos) &&
                        !self.chunk_data_load_queue.contains(&chunk_pos) {
                        self.chunk_data_load_queue.push_back(chunk_pos);

                    }
                }
            }
        }

        self.chunk_data_load_queue.make_contiguous().sort_by(|a, b| (a-position).length_squared().partial_cmp(&(b-position).length_squared()).unwrap());
    }

    pub fn render(&self, render_pass: &mut wgpu::RenderPass, frustum: &Frustum) {
        println!("{}\t{}\t{}", self.chunk_map.len(), self.chunk_data_load_queue.len(), self.chunk_mesh_load_queue.len());
        // let mut count = 0;
        for chunk in self.chunk_map.values() {
            if chunk.render(render_pass, frustum) {
                // count += 1;
            }
        }

        // println!("drew {count} chunks");
    }
}
