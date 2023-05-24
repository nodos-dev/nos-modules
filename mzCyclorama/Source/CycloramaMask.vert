#version 450

#include "CycloDefines.glsl"

layout (location = 0) in vec3 pos;

layout (location = 0) out vec3 outPos;

layout (binding = 0) uniform UBO 
{
	mat4 MVP;
    vec4 Smoothness;
    vec4 SmoothnessCrop;
    vec4 MaskColor;
    vec3 VOffset;
    uint Flags;
    float SmoothnessCurve;
    vec4 Scale;
    float Roundness;
    vec2 Angle;
    vec2 Diag;
} ubo;

void main()
{
    gl_Position = ubo.MVP * vec4(pos.xyz - ubo.VOffset, 1.0);
	outPos    = pos.xyz;
}
