// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#version 450

layout(binding = 0) uniform sampler2D A;
layout(binding = 1) uniform sampler2D B;
layout(binding = 2) uniform Block
{
    vec2 Center;
    vec2 Direction;
} Params;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

vec2 rotate90CCW(vec2 v) {
    return vec2(-v.y, v.x);
}

float getAspectRatio(sampler2D tex) {
    ivec2 size = textureSize(tex, 0); // Get texture dimensions at mip level 0
    return float(size.x) / float(size.y);
}

void main()
{
	if(textureSize(A, 0) != textureSize(B, 0))
	{
		rt = vec4(0,0,0,0);
		return;
	}
	vec2 UV = vec2(getAspectRatio(A) * uv.x, uv.y);
	vec2 normal = normalize(rotate90CCW(Params.Direction));
	vec2 center = Params.Center;
	center.x *= getAspectRatio(A);
    center.y = 1.0f - center.y;
    vec2 center_to_pixel = normalize(UV - center);
    float up_angle = dot(normal, center_to_pixel);
    if(up_angle < 0.0f)
    	rt = texture(B, uv);
    else
    	rt = texture(A, uv);
}










