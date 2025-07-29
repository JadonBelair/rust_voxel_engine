struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct Camera {
	view_pos: vec4<f32>,
	view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
	@location(1) normal: vec3<f32>,
	@location(2) camera_pos: vec3<f32>,
	@location(3) frag_position: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.color = model.color;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.frag_position = model.position;
	out.normal = model.normal;
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
	// let attenuated_diffuse = diffuse * attenuation;
	let attenuated_specular = specular * attenuation;


	let result = ambient + diffuse + attenuated_specular;
	return vec4<f32>(result, 1.0);
}
