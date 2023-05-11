#version 450

layout(binding = 0) uniform sampler2D Input;
layout(binding = 1) uniform Block
{
    vec2 Offset;
} Params;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;


void main()
{
    vec2 UV = uv + Params.Offset;
    if(UV.x < 0 || UV.y < 0 ||  UV.x > 1 || UV.y > 1)
    {
        rt = vec4(0, 0, 0, 0);
    }
    else
    {
        rt = texture(Input, UV);
    }
}