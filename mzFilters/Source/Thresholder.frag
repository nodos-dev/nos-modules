#version 450

layout(binding = 0) uniform sampler2D Input;
layout(binding = 1) uniform ThresholderParams
{
    uniform float Minimum_Luminance;
    uniform uint Output_Type;
} Params;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

#include "../../mzBasic/Source/Shaders/ShaderCommon.glsl"

void main()
{
    vec4 Color = texture(Input, uv);
    float Luminance = dot(Color.rgb, REC709);
    if (Luminance < Params.Minimum_Luminance)
    {
        rt = vec4(0.0, 0.0, 0.0, 1.0);
    }
    else if (Params.Output_Type == 0) // Original
    {
        rt = Color;
    }
    else // White
    {
        rt = vec4(1.0, 1.0, 1.0, 1.0);
    }
}