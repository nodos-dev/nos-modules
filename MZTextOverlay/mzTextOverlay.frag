
#version 450

layout(location = 0) in vec2 uv;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 fragColor;
layout(binding  = 0) uniform sampler2D Atlas;

void main() 
{
    fragColor = color * texture(Atlas, uv);
}