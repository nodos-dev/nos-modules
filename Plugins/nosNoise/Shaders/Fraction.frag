// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#version 450

layout(binding = 0) uniform sampler2D Input;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

layout(binding = 1) uniform SimplexNoiseParams {
	bool BypassAlpha;
}
Params;

void main()
{
    vec4 Color = texture(Input, uv);
    rt = vec4(fract(Color.rgb), Params.BypassAlpha ? Color.a : fract(Color.a));
}
