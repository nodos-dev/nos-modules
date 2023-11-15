#version 450

layout(binding = 0) uniform sampler2D Input1;
layout(binding = 1) uniform sampler2D Input2;
layout(binding = 2) uniform sampler2D Input3;
layout(binding = 3) uniform sampler2D Input4;

layout(binding = 4) uniform SwizzleParams
{
	int RedSource;
	int RedChannel;
	float RedMultiplier;
	float RedOffset;
	
    int GreenSource;
	int GreenChannel;
	float GreenMultiplier;
	float GreenOffset;

    int BlueSource;
	int BlueChannel;
	float BlueMultiplier;
	float BlueOffset;

    int AlphaSource;
	int AlphaChannel;
	float AlphaMultiplier;
	float AlphaOffset;
} Params;


layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

void main()
{

    vec4 col1 = texture(Input1, uv);
    vec4 col2 = texture(Input2, uv);
    vec4 col3 = texture(Input3, uv);
    vec4 col4 = texture(Input4, uv);

    vec4 colors[4] = {col1, col2, col3, col4};

    vec4 finalColor = vec4( colors[Params.RedSource]    [Params.RedChannel], 
                            colors[Params.GreenSource]  [Params.GreenChannel], 
                            colors[Params.BlueSource]   [Params.BlueChannel], 
                            colors[Params.AlphaSource]  [Params.AlphaChannel]);

    vec4 multipliers = {Params.RedMultiplier, Params.GreenMultiplier, Params.BlueMultiplier, Params.AlphaMultiplier};
    vec4 offsets = {Params.RedOffset, Params.GreenOffset, Params.BlueOffset, Params.AlphaOffset};
	rt = finalColor * multipliers + offsets;
}
