// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#version 450

layout(binding = 0) uniform sampler2D Source;
layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

layout(binding=1) uniform CorrectParams 
{
    vec4 Saturation;
    vec4 Contrast;
    vec4 Gamma;
    vec4 Gain;
    vec4 Offset;
    vec3 ContrastCenter;
} CC;

const vec3 RGB2Y =
    vec3(
        0.2722287168, //AP1_2_XYZ_MAT[0][1],
        0.6740817658, //AP1_2_XYZ_MAT[1][1],
        0.0536895174  //AP1_2_XYZ_MAT[2][1]
    );

vec3 Correct(vec3 WorkingColor, vec4 Saturation, vec4 Contrast, vec4 Gamma, vec4 Gain, vec4 Offset, vec3 ContrastCenter)
{
	float Luma = dot(WorkingColor, RGB2Y);
	WorkingColor = max(vec3(0), mix(Luma.xxx, WorkingColor, Saturation.xyz * Saturation.w));
	WorkingColor = pow(WorkingColor * (1.0 / ContrastCenter), Contrast.xyz * Contrast.w) * ContrastCenter;
	WorkingColor = pow(WorkingColor, 1.0 / (Gamma.xyz * Gamma.w));
	WorkingColor = WorkingColor * (Gain.xyz * Gain.w) + Offset.xyz + Offset.w;
	return WorkingColor;
}

void main()
{
    vec4 src = texture(Source, uv);
    rt = vec4(Correct(src.rgb, CC.Saturation, CC.Contrast, CC.Gamma, CC.Gain, CC.Offset, CC.ContrastCenter), src.a);
}