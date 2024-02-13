// Copyright MediaZ AS. All Rights Reserved.

#version 450

layout(binding = 0) uniform sampler2D Input;
layout(binding = 1) uniform GpuStress
{
    uint StressExponent; 
} Params;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

void main()
{
    vec4 Color = texture(Input, uv);
    uint LoopCount = uint(pow(2, Params.StressExponent));
    vec4 ColorSlice = Color * vec4(1.0 / float(LoopCount));
    vec4 Out = vec4(0.0);
    for (uint i = 0; i < LoopCount; ++i)
        Out += ColorSlice;
    rt = Out;
}