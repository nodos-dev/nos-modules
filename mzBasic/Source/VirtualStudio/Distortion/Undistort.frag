#version 450

// Ported from RealityEngine

layout(binding = 0) uniform sampler2D In;
layout(binding = 1) uniform UndistortParams
{
    vec2 k1k2;
    vec2 Center;
    float Distortion_Scale;
    uint Iteration_Count;
    vec2 Red_Offset_k1k2;
    vec2 Blue_Offset_k1k2;
    bool Enable_Chromatic_Aberration;
} Params;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

float CalculateInverseR(float TargetR, vec2 k1k2, float InitialR)
{
	float R = InitialR;
	for (int t = 0; t < Params.Iteration_Count; ++t)
	{		
		float R2 = R * R;
		float R3 = R2 * R;
		float R4 = R2 * R2;
		float R5 = R3 * R2;
		float fR = k1k2.x * R3 + k1k2.y * R5 + R - TargetR; // (K1 * R2 + K2 * R4 + 1) * R - TargetR;
		float dfR = 3 * k1k2.x * R2 + 5 * k1k2.y * R4 + 1;
		float hR = fR / dfR;
		R = R - hR;
	}
	return R;
}

vec4 Undistort()
{
    // Original point.
	vec2 P = uv;

	P -= Params.Center / 2.0;

	// Map to [-1, 1]
	vec2 P_Uniform = (P - 0.5) * 2;

	// This is our aspect ratio.
    vec2 TextureSize = vec2(textureSize(In, 0));
	vec2 Aspect = vec2(TextureSize.x / TextureSize.y, 1);
	float x = sqrt(1.0 / (Aspect.x * Aspect.x + Aspect.y * Aspect.y));	
	vec2 AspectRatio = Aspect * x;

	// Apply aspect ratio.	
	vec2 P_Uniform_Aspect = P_Uniform * AspectRatio * Params.Distortion_Scale;

	// Time to apply undistortion
	float R = length(P_Uniform_Aspect);
	float R_Ratio = CalculateInverseR(R, Params.k1k2, 1.0);
	vec2 P_Uniform_Aspect_Final = P_Uniform_Aspect * (R_Ratio / R);

	// Reapply aspect ratio.
	vec2 P_Uniform_Final = P_Uniform_Aspect_Final / AspectRatio;

	// And back to [0, 1] domain.
	vec2 P_Final = P_Uniform_Final / 2.0 + 0.5;

	P_Final += Params.Center / 2;

	P_Final = clamp(P_Final, 0, 1);

	vec4 Color = texture(In, P_Final);

	// Check outside of image domain.
	if (P_Final.x < 0 || P_Final.x > 1 || P_Final.y < 0 || P_Final.y > 1)
	{
		// Color = vec4(0, 0, 0, 1);
	}

    return Color;
}

vec4 UndistortWithAberration()
{
	// Original point.
	vec2 P = uv;

	P -= Params.Center / 2;

	// Map to [-1, 1]
	vec2 P_Uniform = (P - 0.5) * 2;

	// This is our aspect ratio.	
    vec2 TextureSize = vec2(textureSize(In, 0));
	vec2 Aspect = vec2(TextureSize.x / TextureSize.y, 1);
	float x = sqrt(1.0 / (Aspect.x * Aspect.x + Aspect.y * Aspect.y));	
	vec2 AspectRatio = Aspect * x;

	// Apply aspect ratio.	
	vec2 P_Uniform_Aspect = P_Uniform * AspectRatio * Params.Distortion_Scale;

	// Time to apply undistortion
	float R = length(P_Uniform_Aspect);
	float R_Ratio_Red = CalculateInverseR(R, Params.k1k2 + Params.Red_Offset_k1k2, 1.0);
	float R_Ratio_Green = CalculateInverseR(R, Params.k1k2, 1.0);
	float R_Ratio_Blue = CalculateInverseR(R, Params.k1k2 + Params.Blue_Offset_k1k2, 1.0);
	
	vec2 P_Uniform_Aspect_Final_Red = P_Uniform_Aspect * (R_Ratio_Red / R);
	vec2 P_Uniform_Aspect_Final_Green = P_Uniform_Aspect * (R_Ratio_Green / R);
	vec2 P_Uniform_Aspect_Final_Blue = P_Uniform_Aspect * (R_Ratio_Blue / R);

	// Reapply aspect ratio.
	vec2 P_Uniform_Final_Red = P_Uniform_Aspect_Final_Red / AspectRatio;
	vec2 P_Uniform_Final_Green = P_Uniform_Aspect_Final_Green / AspectRatio;
	vec2 P_Uniform_Final_Blue = P_Uniform_Aspect_Final_Blue / AspectRatio;

	// And back to [0, 1] domain.
	vec2 P_Final_Red = P_Uniform_Final_Red / 2.0 + 0.5;
	vec2 P_Final_Green = P_Uniform_Final_Green / 2.0 + 0.5;
	vec2 P_Final_Blue = P_Uniform_Final_Blue / 2.0 + 0.5;

	P_Final_Red += Params.Center / 2;
	P_Final_Green += Params.Center / 2;
	P_Final_Blue += Params.Center / 2;

	P_Final_Red = clamp(P_Final_Red, 0, 1);
	P_Final_Green = clamp(P_Final_Green, 0, 1);
	P_Final_Blue = clamp(P_Final_Blue, 0, 1);

	float Red = texture(In, P_Final_Red).r;
	float Green = texture(In, P_Final_Green).g;
	float Blue = texture(In, P_Final_Blue).b;
	
	// Check outside of image domain.
	// bool RedOutside = P_Final_Red.x < 0 || P_Final_Red.x > 1 || P_Final_Red.y < 0 || P_Final_Red.y > 1;
	// if (RedOutside)
	// {
	//		Red = 0;
	// }
	// bool GreenOutside = P_Final_Green.x < 0 || P_Final_Green.x > 1 || P_Final_Green.y < 0 || P_Final_Green.y > 1;
	// if (GreenOutside)
	// {
	//		Green = 0;
	// }
	// bool BlueOutside = P_Final_Blue.x < 0 || P_Final_Blue.x > 1 || P_Final_Blue.y < 0 || P_Final_Blue.y > 1;
	// if (BlueOutside)
	// {
	//		Blue = 0;
	// }
	vec4 Color = vec4(Red, Green, Blue, 1);
	// if (RedOutside && GreenOutside && BlueOutside)
	// {
	//		Color = 0;
	// }
	
	return Color;
}

void main()
{
    if (Params.Enable_Chromatic_Aberration)
    {
        rt = UndistortWithAberration();
    }
    else
    {
        rt = Undistort();
    }
}