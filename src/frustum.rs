use glam::{Vec3, Vec4};

use crate::camera::{Camera, Projection};

pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }
}

#[derive(Default)]
pub struct Frustum {
    pub planes: [Vec4; 6],
}

impl Frustum {
    pub fn from_camera(camera: &Camera, projection: &Projection) -> Self {
        let view_proj = projection.calc_matrix() * camera.calc_matrix();
        let mut planes = [Vec4::ZERO; 6];
        let row = view_proj.to_cols_array_2d();

        // Left plane
        planes[0] = Vec4::new(
            row[0][3] + row[0][0],
            row[1][3] + row[1][0],
            row[2][3] + row[2][0],
            row[3][3] + row[3][0],
        )
        .normalize();

        // Right plane
        planes[1] = Vec4::new(
            row[0][3] - row[0][0],
            row[1][3] - row[1][0],
            row[2][3] - row[2][0],
            row[3][3] - row[3][0],
        )
        .normalize();

        // Bottom plane
        planes[2] = Vec4::new(
            row[0][3] + row[0][1],
            row[1][3] + row[1][1],
            row[2][3] + row[2][1],
            row[3][3] + row[3][1],
        )
        .normalize();

        // Top plane
        planes[3] = Vec4::new(
            row[0][3] - row[0][1],
            row[1][3] - row[1][1],
            row[2][3] - row[2][1],
            row[3][3] - row[3][1],
        )
        .normalize();

        // Near plane
        planes[4] = Vec4::new(
            row[0][3] + row[0][2],
            row[1][3] + row[1][2],
            row[2][3] + row[2][2],
            row[3][3] + row[3][2],
        )
        .normalize();

        // Far plane
        planes[5] = Vec4::new(
            row[0][3] - row[0][2],
            row[1][3] - row[1][2],
            row[2][3] - row[2][2],
            row[3][3] - row[3][2],
        )
        .normalize();

        Self { planes }
    }

    pub fn contains_aabb(&self, aabb: &Aabb) -> bool {
        let mut p;
        let mut dp;

        for i in 0..6 {
            p = self.planes[i];
            dp = p.x * (if p.x > 0.0 { aabb.max } else { aabb.min }).x
                + p.y * (if p.y > 0.0 { aabb.max } else { aabb.min }).y
                + p.z * (if p.z > 0.0 { aabb.max } else { aabb.min }).z;

            if dp < -p.w {
                return false;
            }
        }

        return true;
    }
}
