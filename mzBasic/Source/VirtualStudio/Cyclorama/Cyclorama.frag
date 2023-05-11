#version 450

layout(location = 0) in vec4 pos;
layout(location = 0) out vec4 rt;
layout(binding = 1) uniform sampler2D Source;

layout (binding = 0) uniform UBO 
{
	mat4 MVP;
	mat4 CP_MVP;
    vec3 VOffset;
    float SmoothnessCurve;
    float UVSmoothness;
} ubo;

vec4 GetProjectedColor()
{
	vec4 Color = vec4(0,0,0,0);
    const float UVSmoothness = ubo.UVSmoothness;
    const vec2 UV = vec2(0.5) + pos.xy / pos.w * 0.5;
    if((UV.x > 0 && UV.x < 1) && (UV.y > 0 && UV.y < 1))
    {
        float N = mix(2, 16, ubo.SmoothnessCurve);
        Color = texture(Source, UV);
        if(UVSmoothness > 0.001)
        {
            vec2 uv = abs(UV * 2 - 1);
            uv = clamp((uv - vec2(1-UVSmoothness)) / UVSmoothness, 0, 1);
            Color.a = 1 - pow(pow(uv.x, N) + pow(uv.y, N), 1.0 / N);
        }
    }
	return Color;
}

void main()
{   
    rt = GetProjectedColor();
}