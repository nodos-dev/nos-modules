// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#version 450

#define MASK_THRESHOLD 0.001

layout(binding = 0) uniform sampler2D Input;
layout(binding = 1) uniform GaussianBlurParams
{
    float Softness;
    float Kernel_Size;
    uint Pass_Type; // 0 - Horizontal, 1 - Vertical
}
Params;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

#include "../../Shaders/ShaderCommon.glsl"

float GaussCoeff1D(int i, float Sigma)
{
    return (1.0 / sqrt(2.0 * PI_FLOAT * Sigma)) * exp(-(float(i) * float(i)) / (2.0 * Sigma * Sigma));
}

void main()
{
    vec2 TextureSize = textureSize(Input, 0);
    vec2 TexelSize = 1.0 / TextureSize;

    rt = texture(Input, uv);
    if (rt.a < MASK_THRESHOLD || Params.Kernel_Size < MASK_THRESHOLD || Params.Softness < MASK_THRESHOLD)
        return;

    float KSf = Params.Kernel_Size * rt.a;
    int KS = int(trunc(KSf));
    float Frc = fract(KSf);
    float Softness = (KSf / (2.4 / Params.Softness)); // Sigma = Kernel / 2.4 as Natron does.
    float KernelSum = 0.0;
    vec4 Accum = vec4(0.0);
    for (int i = -KS; i <= KS; i++)
    {
        vec2 Coord = uv;
        if (Params.Pass_Type == 0)
            Coord.x += TexelSize.x * float(i); // * Frc;
        else if (Params.Pass_Type == 1)
            Coord.y += TexelSize.y * float(i); // * Frc;
        else {
            rt = vec4(1.0, 0.0, 0.0, 1.0);
            return;
        }
        vec4 SampleColor = texture(Input, Coord);
        float Coeff = GaussCoeff1D(i, Softness);
        KernelSum += Coeff;
        Accum += SampleColor * Coeff;
    }
    rt = Accum / KernelSum;
}