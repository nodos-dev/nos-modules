// Copyright MediaZ AS. All Rights Reserved.

#version 450
#extension GL_EXT_scalar_block_layout : enable

layout(binding = 1)  uniform sampler2D Textures[16];

layout(binding = 0, std430) uniform MergeParams
{
	int   Blends[16];
	float Opacities[16];
	vec4 Background_Color;
	int Texture_Count;
};

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

float ScreenBlend(float s, float t)
{
	return 1.0 - (1.0 - s) * (1.0 - t);
}

vec4 ScreenBlend(vec4 Color1, vec4 Color2)
{
	vec4 Result;

	Result.r = ScreenBlend(Color1.r, Color2.r * Color2.a);
	Result.g = ScreenBlend(Color1.g, Color2.g * Color2.a);
	Result.b = ScreenBlend(Color1.b, Color2.b * Color2.a);
	Result.a = ScreenBlend(Color1.a, Color2.a);

	return Result;
}

vec4 NormalBlend(vec4 Background, vec4 Foreground)
{
	vec4 Result;
	Result.rgb = Background.rgb * (1 - Foreground.a) + Foreground.rgb * Foreground.a;

	Result.a = ScreenBlend(Background.a, Foreground.a);
	return Result;
}

vec4 AdditiveBlend(vec4 Background, vec4 Foreground)
{
	vec4 Result;
	Result.rgb = Background.rgb * (1 - Foreground.a) + Foreground.rgb;

	Result.a = ScreenBlend(Background.a, Foreground.a);
	return Result;
}

vec4 Add(vec4 Background, vec4 Foreground)
{
	return vec4(vec3(Background.rgb + Foreground.rgb * Foreground.a), Background.a);
}

vec4 Subtract(vec4 Background, vec4 Foreground)
{
	return vec4(vec3(Background.rgb - Foreground.rgb * Foreground.a), Background.a);
}

vec4 Multiply(vec4 Background, vec4 Foreground)
{
	return vec4(Background.rgb * mix(Foreground.rgb, vec3(1.0), (1.0 - Foreground.a)), Background.a);
}

vec4 Divide(vec4 Background, vec4 Foreground)
{
	return vec4(vec3(Background / mix(max(0.0001,max(max(Foreground.r, Foreground.g), Foreground.b)), 1.0, (1.0 - Foreground.a))), Background.a);
}

vec4 Min(vec4 Color1, vec4 Color2)
{
	return mix(Color1, min(Color1, Color2), Color2.a);
}

vec4 Max(vec4 Color1, vec4 Color2)
{
	return mix(Color1, max(Color1, Color2), Color2.a);
}

vec4 Blend(vec4 Color1, vec4 Color2, int BlendMode, float Opacity)
{
	vec4 Result;

	switch (BlendMode)
	{
	case 0: Result = NormalBlend(Color1, Color2); break;
	case 1: Result = AdditiveBlend(Color1, Color2); break;
	case 2: Result = Add(Color1, Color2); break;
	case 3: Result = Subtract(Color1, Color2); break;
	case 4: Result = Multiply(Color1, Color2); break;
	case 5: Result = Divide(Color1, Color2); break;
	case 6: Result = ScreenBlend(Color1, Color2); break;
	case 7: Result = Min(Color1, Color2); break;
	case 8: Result = Max(Color1, Color2); break;

	default: Result = vec4(0, 0, 0, 0); break;
	}

	return mix(Color1, Result, Opacity);
}

void main()
{
    vec4 Color = Background_Color;
	for(int i = 0; i < Texture_Count; i++)
		Color = Blend(Color, texture(Textures[i], uv), Blends[i], Opacities[i]);
	
    rt = Color;
}