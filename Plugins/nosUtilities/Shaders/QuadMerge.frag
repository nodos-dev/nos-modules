// Copyright MediaZ AS. All Rights Reserved.

#version 450

layout(binding = 0) uniform sampler2D Input0;
layout(binding = 1) uniform sampler2D Input1;
layout(binding = 2) uniform sampler2D Input2;
layout(binding = 3) uniform sampler2D Input3;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

void main()
{
    const vec2 UV = uv * 2;
    
    if(uv.x < .5 && uv.y < 0.5) // 0
        rt = texture(Input0, UV - vec2(0, 0));
    if(uv.x > .5 && uv.y < 0.5) // 1
        rt = texture(Input1, UV - vec2(1, 0));
    if(uv.x < .5 && uv.y > 0.5) // 2
        rt = texture(Input2, UV - vec2(0, 1));
    if(uv.x > .5 && uv.y > 0.5) // 3
        rt = texture(Input3, UV - vec2(1, 1));
}