#version 450

layout(binding = 0) uniform sampler2D Input;
layout(binding = 1) uniform sampler2D Clean_Plate;

layout(binding = 2) uniform KeyPass1Params
{
    float Key_High_Brightness;
    float Core_Matte_Clean_Plate_Gain;
    float Core_Matte_Gamma_1;
    float Core_Matte_Gamma_2;
    float Core_Matte_Red_Weight;
    float Core_Matte_Blue_Weight;
    float Core_Matte_Black_Point;
    float Core_Matte_White_Point;
} Params;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

#include "IBKCommon.glsl"

float CalculateIBK(vec4 KeyPixel, vec4 CleanPixel, float Gamma, float RedWeight, float BlueWeight)
{
    vec4 FinalGamma = vec4(Gamma, Gamma, Gamma, 1.0);
    KeyPixel = pow(KeyPixel, FinalGamma);
    CleanPixel = pow(CleanPixel, FinalGamma);
    float Alpha = (KeyPixel.g - (KeyPixel.r * RedWeight + KeyPixel.b * BlueWeight)) 
                / (CleanPixel.g - (CleanPixel.r * RedWeight + CleanPixel.b * BlueWeight));
    return mix(clamp(abs(1.0 - Alpha), 0.0, 1.0), clamp(1.0 - Alpha, 0.0, 1.0), Params.Key_High_Brightness);
}

void main()
{
    vec4 InputColor = texture(Input, uv);
    vec4 CleanPlateColor = texture(Clean_Plate, uv);

    float SoftAlpha1 = CalculateIBK(InputColor, CleanPlateColor * Params.Core_Matte_Clean_Plate_Gain,
                                    Params.Core_Matte_Gamma_1, Params.Core_Matte_Red_Weight, Params.Core_Matte_Blue_Weight);
    float SoftAlpha2 = CalculateIBK(InputColor, CleanPlateColor * Params.Core_Matte_Clean_Plate_Gain,
                                    Params.Core_Matte_Gamma_2, Params.Core_Matte_Red_Weight, Params.Core_Matte_Blue_Weight);
    float SoftAlpha = max(SoftAlpha1, SoftAlpha2);
    float HardAlpha = BlackWhitePointClamped(SoftAlpha, Params.Core_Matte_Black_Point, Params.Core_Matte_White_Point);

    rt = vec4(HardAlpha, HardAlpha, HardAlpha, HardAlpha);
}