struct VertexInput {
    @location(0) normal_position: u32,
    // @location(1) normal: u32,
    @location(1) color: vec3<f32>,
};

struct Camera {
	view_pos: vec4<f32>,
	view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

var<push_constant> model: mat4x4<f32>;

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
    @location(0) color: vec3<f32>,
	@location(1) normal: vec3<f32>,
	@location(2) camera_pos: vec3<f32>,
	@location(3) frag_position: vec3<f32>,
};

@vertex
fn vs_main(
    vertex: VertexInput,
) -> VertexOutput {
	let position: vec3<u32> = vec3<u32>(
		(vertex.normal_position >> 12) & 0x3F,
		(vertex.normal_position >>  6) & 0x3F,
		(vertex.normal_position >>  0) & 0x3F,
	);
	let world_position = model * vec4<f32>(vec3<f32>(position), 1.0);

	let normal_index = (vertex.normal_position >> 18) & 0x07;

    var out: VertexOutput;
    out.color = vertex.color;
    out.clip_position = camera.view_proj * world_position;
    out.frag_position = world_position.xyz;
	out.normal = NORMALS[normal_index];
	out.camera_pos = camera.view_pos.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	let ambient = 0.4 * in.color;

	var light_dir = in.camera_pos - in.frag_position;
	let distance = length(light_dir);
	light_dir = normalize(light_dir);

	let view_dir = light_dir;
	let halfway_dir = normalize(light_dir + view_dir);

	let diffuse_strength = max(dot(light_dir, in.normal), 0.0);
	let diffuse = diffuse_strength * in.color;

	let specular_strength = 0.3;
	let reflect_dir = reflect(-light_dir, in.normal);
	let spec = pow(max(dot(in.normal, halfway_dir), 0.0), 16);
	let specular = specular_strength * spec;

	let attenuation = 1.0 / (1.0 + 0.027 * distance + 0.0028 * (distance * distance));
	let attenuated_specular = specular * attenuation;


	let result = ambient + diffuse + attenuated_specular;
	return vec4<f32>(result, 1.0);
}
