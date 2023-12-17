#version 450

layout(location = 0) in vec4 pos;
layout(location = 0) out vec4 rt;
layout (binding = 0) uniform UBO 
{
	mat4 MVP;
    float Coeff;
};

void main()
{   
    rt = vec4(1,1,1,1);
}