#version 450

layout(binding = 0)  uniform sampler2D Texture_0;
layout(binding = 1)  uniform sampler2D Texture_1;
layout(binding = 2)  uniform sampler2D Texture_2;
layout(binding = 3)  uniform sampler2D Texture_3;
layout(binding = 4)  uniform sampler2D Texture_4;
layout(binding = 5)  uniform sampler2D Texture_5;
layout(binding = 6)  uniform sampler2D Texture_6;
layout(binding = 7)  uniform sampler2D Texture_7;
layout(binding = 8)  uniform sampler2D Texture_8;
layout(binding = 9)  uniform sampler2D Texture_9;
layout(binding = 10) uniform sampler2D Texture_10;
layout(binding = 11) uniform sampler2D Texture_11;
layout(binding = 12) uniform sampler2D Texture_12;
layout(binding = 13) uniform sampler2D Texture_13;
layout(binding = 14) uniform sampler2D Texture_14;
layout(binding = 15) uniform sampler2D Texture_15;

layout(binding = 16) uniform MergeParams
{
	int Blend_Mode_0;
	int Blend_Mode_1;
	int Blend_Mode_2;
	int Blend_Mode_3;
	int Blend_Mode_4;
	int Blend_Mode_5;
	int Blend_Mode_6;
	int Blend_Mode_7;
	int Blend_Mode_8;
	int Blend_Mode_9;
	int Blend_Mode_10;
	int Blend_Mode_11;
	int Blend_Mode_12;
	int Blend_Mode_13;
	int Blend_Mode_14;
	int Blend_Mode_15;

	float Opacity_0;
	float Opacity_1;
	float Opacity_2;
	float Opacity_3;
	float Opacity_4;
	float Opacity_5;
	float Opacity_6;
	float Opacity_7;
	float Opacity_8;
	float Opacity_9;
	float Opacity_10;
	float Opacity_11;
	float Opacity_12;
	float Opacity_13;
	float Opacity_14;
	float Opacity_15;

	vec4 Background_Color;
	int Texture_Count;
} Params;



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

vec4 TextureLookup(int TextureIndex)
{
	vec4 Result;

	switch (TextureIndex)
	{
	case 0x0: Result = texture(Texture_0, uv); break;
	case 0x1: Result = texture(Texture_1, uv); break;
	case 0x2: Result = texture(Texture_2, uv); break;
	case 0x3: Result = texture(Texture_3, uv); break;
	case 0x4: Result = texture(Texture_4, uv); break;
	case 0x5: Result = texture(Texture_5, uv); break;
	case 0x6: Result = texture(Texture_6, uv); break;
	case 0x7: Result = texture(Texture_7, uv); break;
	case 0x8: Result = texture(Texture_8, uv); break;
	case 0x9: Result = texture(Texture_9, uv); break;
	case 0xA: Result = texture(Texture_10, uv); break;
	case 0xB: Result = texture(Texture_11, uv); break;
	case 0xC: Result = texture(Texture_12, uv); break;
	case 0xD: Result = texture(Texture_13, uv); break;
	case 0xE: Result = texture(Texture_14, uv); break;
	case 0xF: Result = texture(Texture_15, uv); break;
	}
	return Result;
}

int BlendModeLookup(int TextureIndex)
{
	int Result;

	switch (TextureIndex)
	{
	case 0x0: Result = Params.Blend_Mode_0; break;
	case 0x1: Result = Params.Blend_Mode_1; break;
	case 0x2: Result = Params.Blend_Mode_2; break;
	case 0x3: Result = Params.Blend_Mode_3; break;
	case 0x4: Result = Params.Blend_Mode_4; break;
	case 0x5: Result = Params.Blend_Mode_5; break;
	case 0x6: Result = Params.Blend_Mode_6; break;
	case 0x7: Result = Params.Blend_Mode_7; break;
	case 0x8: Result = Params.Blend_Mode_8; break;
	case 0x9: Result = Params.Blend_Mode_9; break;
	case 0xA: Result = Params.Blend_Mode_10; break;
	case 0xB: Result = Params.Blend_Mode_11; break;
	case 0xC: Result = Params.Blend_Mode_12; break;
	case 0xD: Result = Params.Blend_Mode_13; break;
	case 0xE: Result = Params.Blend_Mode_14; break;
	case 0xF: Result = Params.Blend_Mode_15; break;
	}

	return Result;
}

float OpacityLookup(int TextureIndex)
{
	float Result;

	switch (TextureIndex)
	{
	case 0x0: Result = Params.Opacity_0; break;
	case 0x1: Result = Params.Opacity_1; break;
	case 0x2: Result = Params.Opacity_2; break;
	case 0x3: Result = Params.Opacity_3; break;
	case 0x4: Result = Params.Opacity_4; break;
	case 0x5: Result = Params.Opacity_5; break;
	case 0x6: Result = Params.Opacity_6; break;
	case 0x7: Result = Params.Opacity_7; break;
	case 0x8: Result = Params.Opacity_8; break;
	case 0x9: Result = Params.Opacity_9; break;
	case 0xA: Result = Params.Opacity_10; break;
	case 0xB: Result = Params.Opacity_11; break;
	case 0xC: Result = Params.Opacity_12; break;
	case 0xD: Result = Params.Opacity_13; break;
	case 0xE: Result = Params.Opacity_14; break;
	case 0xF: Result = Params.Opacity_15; break;
	}

	return Result;
}

void main()
{
    vec4 Color = Params.Background_Color;

	for(int i = 0; i < Params.Texture_Count; i++)
	{
		Color = Blend(Color, TextureLookup(i), BlendModeLookup(i), OpacityLookup(i));
	}

	rt = Color;
}