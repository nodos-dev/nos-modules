#version 450

layout(location = 0) in vec4 pos;
layout(location = 1) in vec3 modelPos;
layout(location = 0) out vec4 rt;
layout (binding = 0) uniform UBO 
{
	mat4 MV, P;
    float Smoothness;
};


void main()
{   
    rt = vec4(pos.z) / Smoothness;
}