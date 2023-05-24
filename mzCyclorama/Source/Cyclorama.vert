#version 450

layout (location = 0) in vec3 pos;

layout (location = 0) out vec4 outPos;

layout (binding = 0) uniform UBO 
{
	mat4 MVP;
	mat4 CP_MVP;
    vec3 VOffset;
    float SmoothnessCurve;
    float UVSmoothness;
} ubo;

void main() 
{
    gl_Position = ubo.MVP * vec4(pos.xyz - ubo.VOffset, 1.0);
    outPos  = ubo.CP_MVP * vec4 (pos.xyz - ubo.VOffset, 1.0);
}
