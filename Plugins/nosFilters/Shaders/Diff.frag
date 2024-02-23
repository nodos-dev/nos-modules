// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#version 450

layout(binding = 0) uniform sampler2D Texture_0;
layout(binding = 1) uniform sampler2D Texture_1;

layout(binding = 2) uniform DiffParams
{
    uniform vec4 Color_Gain;
    uniform bool Absolute_Diff;
    uniform bool Use_Alpha;
} Params;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

void main()
{
    vec4 Color0 = texture(Texture_0, uv);
    vec4 Color1 = texture(Texture_1, uv);

    if (!Params.Use_Alpha)
    {
        Color1.a = 0.0;
    }
    
    vec4 Color = Color0 - Color1;
    
    if (Params.Absolute_Diff)
    {
        Color = abs(Color);
    }

    rt = Params.Color_Gain * Color;
}