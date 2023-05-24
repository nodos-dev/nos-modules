#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 rt;
layout(binding = 0)  uniform sampler2D lhs;
layout(binding = 1)  uniform sampler2D rhs;

void main()
{
    rt = texture(lhs, uv) + texture(rhs, uv) * 0.02;
    rt.a = 1;
}