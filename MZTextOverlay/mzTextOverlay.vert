#version 450

layout(location = 1) in vec2 Pos;
layout(location = 2) in vec2 UV;

layout(location = 0) out vec2 outUv;
layout(location = 1) out vec4 outColor;

layout(binding = 1) uniform BLOCK
{
    vec2 Position;
    vec2 Scale;
    vec3 Color;
} block;

void main() 
{
    gl_Position = vec4(block.Scale * (block.Position + Pos), 0, 1);
    outUv = UV;
    outColor = vec4(block.Color, 1);
}