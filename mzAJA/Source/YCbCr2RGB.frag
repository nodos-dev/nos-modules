#version 450
#include "Common.glsl"

layout(binding = 0) uniform usampler2D Source;
layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

layout (binding = 1, rgba8)  uniform writeonly image2D Output;


void main()
{
    const bool Interlaced  = (InterlacedFlags & 3) != 0;
    const bool OddFrame    = (InterlacedFlags & 2) != 0;
    const bool EvenFrame   = (InterlacedFlags & 1) != 0;
    const vec2 Size        = textureSize(Source, 0);
    const vec2 InvSize     = 1.0 / Size;
    const bool OddLine     = 0 != (uint(gl_FragCoord.y) & 1);
    const bool OddColumn   = 0 != (uint(gl_FragCoord.x) & 1);

	vec2 uv0 = gl_FragCoord.xy * InvSize * vec2(0.5,1);
    //uv0 += InvSize * vec2(1, 0.5);

    if(OddFrame) uv0.y -= InvSize.y;
    
    float X = fract(floor(uv0.x * Size.x * 2.0) / 2.0);
    vec4 C0 = texture(Source, uv0).bgra / 255.0;

    if(!(X > 0.0))
    {
        rt = SDR_In_8(uvec3((OddColumn ? C0.gbr : C0.abr) * 255));
        return;
    }
    
    vec2 uv1 = (gl_FragCoord.xy + vec2(1, 0)) * InvSize * vec2(0.5,1);
    vec4 C2 = texture(Source, uv1).bgra / 255.0;

    const float Y0 = C0.g;
    const float Y1 = C0.a;
    const float Y2 = C2.g;
    const vec4 YY = IDX4(uvec4(round(vec4(Y0, Y2, Y1, Y1) * 255))) / 255.0;
	const vec2 Diff = abs(YY.xy - YY.zw);
    const float TotalDiff = Diff.x + Diff.y;
    const vec2 CbCr = mix(C0.br, C2.br, TotalDiff > 0.0 ? clamp(Diff.x / TotalDiff, 0, 1) : 0.5);

    rt =  SDR_In_8(uvec3(Y1  * 255, CbCr  * 255));
}