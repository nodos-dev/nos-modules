// Copyright MediaZ AS. All Rights Reserved.

#version 450

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

layout(binding = 1) uniform UBO
{
    ivec2 Squares;
    vec4 OddColor;
    vec4 EvenColor;
}
ubo;

void main()
{
    const ivec2 UV = ivec2(uv * max(ubo.Squares, ivec2(2, 2)));
    rt.xyz = mix(ubo.EvenColor.xyz, ubo.OddColor.xyz, float((UV.x ^ UV.y) & 1));
    rt.w = 1;
}