// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

struct Input
{
[[vk::location(0)]] float2 uv : UV0;
};


[[vk::combinedImageSampler]][[vk::binding(0, 0)]] Texture2D<float4> Source;
[[vk::combinedImageSampler]][[vk::binding(0, 0)]] SamplerState SourceSampler;

[[vk::binding(1, 0)]] 
cbuffer ubo { 
    float4 Saturation;
    float4 Contrast;
    float4 Gamma;
    float4 Gain;
    float4 Offset;
    float3 ContrastCenter;
};

float3 Correct(float3 WorkingColor, float4 Saturation, float4 Contrast, float4 Gamma, float4 Gain, float4 Offset, float3 ContrastCenter)
{
	float Luma = dot(WorkingColor, float3(0.2722287168,0.6740817658,0.0536895174));
	WorkingColor = max(float3(0,0,0), lerp(Luma.xxx, WorkingColor, Saturation.xyz * Saturation.w));
	WorkingColor = pow(WorkingColor * (1.0 / ContrastCenter), Contrast.xyz * Contrast.w) * ContrastCenter;
	WorkingColor = pow(WorkingColor, 1.0 / (Gamma.xyz * Gamma.w));
	WorkingColor = WorkingColor * (Gain.xyz * Gain.w) + Offset.xyz + Offset.w;
	return WorkingColor;
}


float4 main(Input input) : SV_TARGET0
{
    float4 src = Source.Sample(SourceSampler, input.uv);
    return float4(Correct(src.rgb, Saturation, Contrast, Gamma, Gain, Offset, ContrastCenter), src.a);
}
