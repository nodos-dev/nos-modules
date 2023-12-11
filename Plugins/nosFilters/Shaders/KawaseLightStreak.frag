#version 450

layout(binding = 0) uniform sampler2D Input;
layout(binding = 1) uniform KawaseLightStreakParams
{
    uniform vec2 Streak_Direction;
    uniform uint Samples;
    uniform float Attenuation;
    uniform uint Iteration;

} Params;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

void main()
{
    vec2 TexelSize = 1.0 / vec2(textureSize(Input, 0));
    vec2 SampleCoord = vec2(0.0);
    vec4 Color = vec4(0.0);

    float B = pow(Params.Samples, Params.Iteration);
    for (uint i = 0; i < Params.Samples; i++)
    {
        float Weight = pow(Params.Attenuation, B * i);
        SampleCoord = uv + (Params.Streak_Direction * B * i * TexelSize);
        Color += clamp(Weight, 0.0, 1.0) * texture(Input, SampleCoord);
    }
    rt = clamp(Color, 0.0, 1.0);
}