#version 450

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

layout(binding = 1) uniform UBO
{
    vec4 OddColor;
    vec4 EvenColor;
    uint Kind;
}
ubo;

float GetCoeff()
{
    switch(ubo.Kind)
    {
        case 0: return length((2 * uv - 1) * vec2(16.0/9.0,1)); // Radial
        case 1: return uv.x; // Horizontal
        case 2: return uv.y; // Vertical
        case 3: return dot(vec2(uv.x, 1-uv.y) + .5, vec2(.70710678118)) - .5; // Diagonal
        case 4: return dot(uv + .5, vec2(.70710678118)) - .5; // Diagonal
        default: return 0.5;   
    }
}

void main()
{
    rt = mix(ubo.OddColor, ubo.EvenColor, GetCoeff());
    rt.w = 1;
}
