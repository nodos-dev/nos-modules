#version 450

layout(binding = 0) uniform sampler2D Input;
layout(binding = 1) uniform SharpenParams
{
	float Radius;
	float Sharpness;
	int PreMultiplyAlpha;
} Params;


layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

void main()
{
	vec2 TextureSize = textureSize(Input, 0);
    vec2 Step  = 1.0 / TextureSize;

    vec4 ColorA = texture(Input,  uv + vec2(-Step.x, -Step.y) * Params.Radius);
	vec4 ColorB = texture(Input,  uv + vec2(Step.x, -Step.y)  * Params.Radius);
	vec4 ColorC = texture(Input,  uv + vec2(-Step.x, Step.y)  * Params.Radius);
	vec4 ColorD = texture(Input,  uv + vec2(Step.x, Step.y)   * Params.Radius);

    if (Params.PreMultiplyAlpha > 0)
	{
		ColorA.rgb = ColorA.rgb * ColorA.a;
		ColorB.rgb = ColorB.rgb * ColorB.a;
		ColorC.rgb = ColorC.rgb * ColorC.a;
		ColorD.rgb = ColorD.rgb * ColorD.a;
	}

    vec4 Around = 0.25 * (ColorA + ColorB + ColorC + ColorD);
	vec4 Center  = texture(Input, uv);

    if (Params.PreMultiplyAlpha > 0)
	{
		Center.rgb = Center.rgb * Center.a;
	}

    vec4 Color = vec4(Center.rgb + clamp(Center.rgb - Around.rgb, 0.0, 1.0) * Params.Sharpness, Center.a);

    if (Params.PreMultiplyAlpha > 0)
	{
		Color.rgb = Color.rgb / Color.a;
	}

    rt = Color;
}
