// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#version 450

layout(binding = 0) uniform sampler2D Input;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

void main()
{
    rt = texture(Input, uv);
}