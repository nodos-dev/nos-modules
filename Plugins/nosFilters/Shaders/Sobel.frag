// Copyright MediaZ AS. All Rights Reserved.

#version 450

layout(binding = 0) uniform sampler2D Input;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

#include "../../Shaders/ShaderCommon.glsl"

const float SobelHorz[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1,
};

const float SobelVert[9] = {
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1,
};

void main()
{
    vec2 TexelSize = 1.0 / vec2(textureSize(Input, 0));

    float Horz = 0.0;
    float Vert = 0.0;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            vec4 Color = texture(Input, uv + vec2(i, j) * TexelSize);
            float Lum = dot(Color.rgb, REC709);
            Horz += Lum * SobelHorz[i + j * 3];
            Vert += Lum * SobelVert[i + j * 3];
        }
    }

    float G = sqrt(Horz * Horz + Vert * Vert);
    rt = vec4(G, G, G, 1.0);
}