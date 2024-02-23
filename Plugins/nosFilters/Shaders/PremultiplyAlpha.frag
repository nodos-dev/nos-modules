// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#version 450

layout(binding = 0) uniform sampler2D Input;
layout(binding = 1) uniform PremultiplyParams
{
    bool Invert_Alpha;
} Params;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

void main()
{
    vec4 Color = texture(Input, uv);
    float Factor = abs(int(Params.Invert_Alpha) - Color.a);
    rt = vec4(Color.rgb * Factor, 1.0);
}