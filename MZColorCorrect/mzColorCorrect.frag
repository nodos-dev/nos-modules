#version 450

layout(binding = 0) uniform sampler2D Source;
layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

layout(binding=1) uniform ColorCorrectParams 
{
    vec4 ColorSaturation;
    vec4 ColorContrast;
    vec4 ColorGamma;
    vec4 ColorGain;
    vec4 ColorOffset;
    vec3 ColorContrastCenter;
} CC;

const vec3 RGB2Y =
    vec3(
        0.2722287168, //AP1_2_XYZ_MAT[0][1],
        0.6740817658, //AP1_2_XYZ_MAT[1][1],
        0.0536895174  //AP1_2_XYZ_MAT[2][1]
    );

vec3 ColorCorrect(vec3 WorkingColor, vec4 ColorSaturation, vec4 ColorContrast, vec4 ColorGamma, vec4 ColorGain, vec4 ColorOffset, vec3 ColorContrastCenter)
{
	float Luma = dot(WorkingColor, RGB2Y);
	WorkingColor = max(vec3(0), mix(Luma.xxx, WorkingColor, ColorSaturation.xyz * ColorSaturation.w));
	WorkingColor = pow(WorkingColor * (1.0 / ColorContrastCenter), ColorContrast.xyz * ColorContrast.w) * ColorContrastCenter;
	WorkingColor = pow(WorkingColor, 1.0 / (ColorGamma.xyz * ColorGamma.w));
	WorkingColor = WorkingColor * (ColorGain.xyz * ColorGain.w) + ColorOffset.xyz + ColorOffset.w;
	return WorkingColor;
}

void main()
{
    rt = vec4(ColorCorrect(texture(Source, uv).rgb, CC.ColorSaturation, CC.ColorContrast, CC.ColorGamma, CC.ColorGain, CC.ColorOffset, CC.ColorContrastCenter), 1);
}