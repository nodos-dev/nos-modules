#version 450

layout(binding = 0) uniform sampler2D Input;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

#include "ShaderCommon.glsl"

void main()
{
    vec4 Color = texture(Input, uv);
    rt = SRGB2Linear(Color);
}