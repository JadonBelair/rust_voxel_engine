struct VertexInput {
	@location(0) packed_data: u32,
	// @location(1) uv: vec2<f32>,
	@location(1) voxel_pos: vec3<i32>,
};

struct Camera {
	view_pos: vec4<f32>,
	view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

var<push_constant> push: array<i32, 6>;

const NORMALS: array<vec3<f32>, 6> = array(
		vec3<f32>( 0.0,  0.0, -1.0), // Front
		vec3<f32>( 0.0,  0.0,  1.0), // Back
		vec3<f32>(-1.0,  0.0,  0.0), // Left
		vec3<f32>( 1.0,  0.0,  0.0), // Right
		vec3<f32>( 0.0, -1.0,  0.0), // Bottom
		vec3<f32>( 0.0,  1.0,  0.0), // Top
);

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
	@location(1) normal: vec3<f32>,
	@location(3) frag_position: vec3<f32>,
	@location(4) voxel_pos: vec3<i32>,
};

@vertex
fn vs_main(
    vertex: VertexInput,
) -> VertexOutput {
	let position = vec3<f32>(
		f32((vertex.packed_data >> 12) & 0x3F),
		f32((vertex.packed_data >>  6) & 0x3F),
		f32((vertex.packed_data >>  0) & 0x3F),
	);

	var model = mat4x4<f32>(
		1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
		0.0, 0.0, 0.0, 1.0,
	);
	let chunk_pos = vec3<f32>(f32(push[0]), f32(push[1]), f32(push[2]));
	model[3] = vec4<f32>(chunk_pos, 1.0);
	let world_position = model * vec4<f32>(position, 1.0);

	let normal_index = (vertex.packed_data >> 18) & 0x07;

	let uv_index = (vertex.packed_data >> 21) & 0xFF;
	let v_index = (vertex.packed_data >> 29) & 0x03;

	var uv = vec2<f32>(f32(uv_index % 16) / 16.0, floor(f32(uv_index) / 16.0));

	switch v_index {
		case 0: {
			uv.x += 1.0 / 16.0;
			uv.y += 1.0 / 16.0;
		}
		case 1: {
			uv.y += 1.0 / 16.0;
		}
		case 3: {
			uv.x += 1.0 / 16.0;
		}
		default: {
		}
	}


    var out: VertexOutput;
    out.uv = uv;
    out.clip_position = camera.view_proj * world_position;
    out.frag_position = world_position.xyz;
	out.normal = NORMALS[normal_index];
	out.voxel_pos = vertex.voxel_pos;
    return out;
}

const BLINN_PHONG: bool = true;

const LIGHT_RANGE: f32 = 15.0;
const LINEAR: f32 = 1.0 / LIGHT_RANGE;
const QUADRATIC: f32 = 1.0 / (LIGHT_RANGE * LIGHT_RANGE);

@group(1) @binding(0)
var t_atlas: texture_2d<f32>;
@group(1) @binding(1)
var s_atlas: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	var result = vec3<f32>(1.0, 1.0, 1.0);
	let look = vec3<i32>(push[3], push[4], push[5]);
	let color = textureSample(t_atlas, s_atlas, in.uv).xyz;
	if (BLINN_PHONG) {
		let ambient = 0.4 * color;

		var light_dir = camera.view_pos.xyz - in.frag_position;
		let distance = length(light_dir);
		light_dir = normalize(light_dir);

		let view_dir = light_dir;
		let halfway_dir = normalize(light_dir + view_dir);

		let diffuse_strength = max(dot(light_dir, in.normal), 0.0);
		let diffuse = diffuse_strength * color;

		let specular_strength = 0.3;
		let reflect_dir = reflect(-light_dir, in.normal);
		let spec = pow(max(dot(in.normal, halfway_dir), 0.0), 16);
		let specular = specular_strength * spec;

		let attenuation = 1.0 / (1.0 + LINEAR * distance + QUADRATIC * (distance * distance));
		let attenuated_specular = specular * attenuation;
		let attenuated_diffuse = diffuse * attenuation;

		result = ambient + attenuated_diffuse;
	} else {
		result = color;

		if (in.normal.x != 0.0) {
			result *= 0.6;
		} else if (in.normal.z != 0.) {
			result *= 0.8;
		}
	}

	if (in.voxel_pos.x == look.x && in.voxel_pos.y == look.y && in.voxel_pos.z == look.z) {
		result *= 1.8;
		result = clamp(result, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
	}

	return vec4<f32>(result, 1.0);
}
