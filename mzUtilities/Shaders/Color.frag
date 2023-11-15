#version 450

layout(binding = 0) uniform ColorParams
{
	vec4 Color;
} Params;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

void main()
{
	rt = Params.Color;
}
