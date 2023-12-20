#version 450

layout (location = 0) in vec3 pos;

layout (location = 0) out vec4 outPos;
layout (location = 1) out vec3 modelPos;

layout (binding = 0) uniform UBO 
{
	mat4 MV, P;
    float Smoothness;
};

void main() 
{
    modelPos = pos;
    outPos = MV * vec4(pos.xyz, 1.0);
    gl_Position = P * outPos;
}