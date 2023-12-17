#version 450

layout (location = 0) in vec3 pos;

layout (location = 0) out vec4 outPos;

layout (binding = 0) uniform UBO 
{
	mat4 MVP;
    float Coeff;
};

void main() 
{
    outPos = gl_Position = MVP * vec4(pos.xyz, 1.0);
}