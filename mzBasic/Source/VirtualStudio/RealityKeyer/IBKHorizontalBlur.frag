#version 450

layout(binding = 0) uniform sampler2D Input;
layout(binding = 1) uniform HorizontalBlurParams
{
    uniform float Blur_Radius;
    uniform vec2 Input_Texture_Size;
} Params;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

#include "../ShaderCommon.glsl"
#include "IBKCommon.glsl"

void main()
{
    vec2 TexelSize = 1.0 / Params.Input_Texture_Size;
    int FlooredBlurRadius = int(Params.Blur_Radius);
    float BlurredMatte = 0.0;
    float TotalWeight = 0.0;
    for (int X = -FlooredBlurRadius; X <= FlooredBlurRadius; ++X)
    {
        float KernelX = float(X) / max(Params.Blur_Radius, 1.0);
        float Weight = KernelTriangle1D(KernelX, 0.0, 1.0);
        BlurredMatte += texture(Input, uv + TexelSize * vec2(X, 0.0)).r * Weight;
        TotalWeight += Weight;
    }
    rt = vec4(BlurredMatte / TotalWeight);
}
