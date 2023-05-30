#version 450

layout(binding = 0) uniform sampler2D Input;
layout(binding = 1) uniform sampler2D Clean_Plate;
layout(binding = 2) uniform sampler2D Clean_Plate_Mask;
layout(binding = 3) uniform sampler2D Core_Matte;
layout(binding = 4) uniform sampler2D Unblurred_Core_Matte;

layout(binding = 5) uniform KeyPass2Params
{
    vec2 Core_Matte_Texture_Size;
    float Erode;
    float Softness;
    float Soft_Matte_Red_Weight;
    float Soft_Matte_Blue_Weight;
    float Soft_Matte_Gamma_1;
    float Soft_Matte_Gamma_2;
    float Soft_Matte_Clean_Plate_Gain;
    vec2 Soft_Matte_422_Filtering;
    float Key_High_Brightness;
    float Core_Matte_Blend;
    vec3 Edge_Spill_Replace_Color;
    vec3 Core_Spill_Replace_Color;
    float Spill_Matte_Gamma;
    float Spill_Matte_Red_Weight;
    float Spill_Matte_Blue_Weight;
    float Spill_Matte_Gain;
    float Spill_RB_Weight;
    float Spill_Suppress_Weight;
    vec2 Spill_422_Filtering;
    float Screen_Subtract_Edge;
    float Screen_Subtract_Core;
    float Keep_Edge_Luma;
    float Keep_Core_Luma;
    float Final_Matte_Black_Point;
    float Final_Matte_White_Point;
    float Final_Matte_Gamma;
    vec3 Gamma;
    vec3 Exposure;
    vec3 Offset;
    vec3 Saturation;
    vec3 Contrast;
    vec3 Contrast_Center;
    uint Output_Type;
}
Params;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

#include "../../Shaders/ShaderCommon.glsl"
#include "IBKCommon.glsl"

vec3 Pow3(vec3 x, vec3 y, float clmp)
{
    return pow(max(abs(x), vec3(clmp, clmp, clmp)), y);
}

float LumaKey(vec4 A, vec4 B)
{
    return abs(dot(A.rgb, REC709)) - dot(B.rgb, REC709);
}

float CalculateIBK(vec3 KeyPixel, vec3 CleanPixel, float Gamma, float RedWeight, float BlueWeight)
{
    vec4 FinalGamma = vec4(Gamma, Gamma, Gamma, 1.0);
    KeyPixel = pow(KeyPixel, FinalGamma.rgb);
    CleanPixel = pow(CleanPixel, FinalGamma.rgb);
    float Alpha = (KeyPixel.g - (KeyPixel.r * RedWeight + KeyPixel.b * BlueWeight)) 
                              / (CleanPixel.g - (CleanPixel.r * RedWeight + CleanPixel.b * BlueWeight));
    vec3 AlphaKey = Alpha * CleanPixel;
    AlphaKey = KeyPixel - clamp(AlphaKey, vec3(0), AlphaKey);
    float Alpha1 = mix(clamp(abs(1.0 - Alpha), 0.0, 1.0), clamp(1.0 - Alpha, 0.0, 1.0), Params.Key_High_Brightness);
    float Alpha2 = max(AlphaKey.r, AlphaKey.g);
    Alpha = max(Alpha1, Alpha2);
    return Alpha;
}

vec3 HorizontalYUVBlur(vec3 InputColor, vec2 TexelSize, vec2 UV, vec2 Weights)
{
    vec2 Offset = vec2(TexelSize.x, 0);
    vec3 Pre = RGB2YUV(texture(Input, UV - Offset).xyz);
    vec3 Nex = RGB2YUV(texture(Input, UV + Offset).xyz);
    vec3 Cur = RGB2YUV(InputColor);
	
    vec2 Average = Pre.gb * Weights.y + Cur.gb * Weights.x + Nex.gb * Weights.y;

    return max(vec3(0), YUV2RGB(vec3(Cur.r, Average.x, Average.y)));
}

void main()
{
    // Read texel size of the first pass.
    vec2 TexelSize = 1.0 / Params.Core_Matte_Texture_Size;

    float BlurRadius = Params.Erode + Params.Softness;
    int FlooredBlurRadius = int(BlurRadius);
    float BlurredMatte = 0.0;
    float TotalWeight = 0.0;

    // Blur the Core Matte using neighbor texels with a triangle filter.
    for (int Y = -FlooredBlurRadius; Y <= FlooredBlurRadius; ++Y)
    {
        float KernelY = float(Y) / max(BlurRadius, 1.0);
        // for (int X = -FlooredBlurRadius; X <= FlooredBlurRadius; ++X)
        // {
        //    float KernelX = float(X) / max(BlurRadius, 1.0);
        float Weight = KernelTriangle1D(KernelY, 0.0, 1.0);
        BlurredMatte += texture(Core_Matte, uv + vec2(0, Y) * TexelSize).r * Weight;
        TotalWeight += Weight;
        // }
    }
    // Normalize blur result
    BlurredMatte /= TotalWeight;
    float FinalBlurMatte = BlurredMatte;

    // Set Erode and Softness values by clamping black and white levels
    float Softness = max(Params.Softness, 0.0001);
    float BlackPoint = 1.0 - Softness / BlurRadius / 2;
    float WhitePoint = 1.0;
    BlurredMatte = BlackWhitePointClamped(BlurredMatte, BlackPoint, WhitePoint);

    // Read values for Soft IBk
    vec4 InputColor = texture(Input, uv);
    vec4 CleanPlateColor = texture(Clean_Plate, uv);
    vec4 CleanPlateMaskColor = texture(Clean_Plate_Mask, uv);

    // Calculate 2 IBK passes for soft matte, and max them
    vec3 SoftMatteCorrectedColor = HorizontalYUVBlur(InputColor.rgb, TexelSize, uv, Params.Soft_Matte_422_Filtering);
    float SoftAlpha1 = CalculateIBK(SoftMatteCorrectedColor, CleanPlateColor.rgb * Params.Soft_Matte_Clean_Plate_Gain,
                                    Params.Soft_Matte_Gamma_1, Params.Soft_Matte_Red_Weight, Params.Soft_Matte_Blue_Weight);
    float SoftAlpha2 = CalculateIBK(SoftMatteCorrectedColor, CleanPlateColor.rgb * Params.Soft_Matte_Clean_Plate_Gain,
                                    Params.Soft_Matte_Gamma_2, Params.Soft_Matte_Red_Weight, Params.Soft_Matte_Blue_Weight);
    float SoftAlpha = max(SoftAlpha1, SoftAlpha2);

    // Key also luma and max it
    float KeyedLuma = LumaKey(InputColor * Params.Soft_Matte_Clean_Plate_Gain, CleanPlateColor * Params.Soft_Matte_Clean_Plate_Gain);
    SoftAlpha = max(KeyedLuma, SoftAlpha);

    // Generate spill subtraction matte
    vec3 SpillCorrectedColor = HorizontalYUVBlur(InputColor.rgb, TexelSize, uv, Params.Spill_422_Filtering);
    float SpillAlpha = CalculateIBK(SpillCorrectedColor, CleanPlateColor.rgb * Params.Soft_Matte_Clean_Plate_Gain, 1.0 / Params.Spill_Matte_Gamma,
                                    Params.Spill_Matte_Red_Weight, Params.Spill_Matte_Blue_Weight);
    SpillAlpha = pow(SpillAlpha, Params.Spill_Matte_Gain);

    // Combine Soft and Core Matte, using Screen blending
    float FinalAlpha = 1.0 - ((1.0 - BlurredMatte * Params.Core_Matte_Blend) * (1.0 - SoftAlpha));
    FinalAlpha *= InputColor.a;
    float FinalAlphaAdjusted = BlackWhitePointClamped(FinalAlpha, Params.Final_Matte_Black_Point, Params.Final_Matte_White_Point);
    FinalAlphaAdjusted = pow(FinalAlphaAdjusted, 1.0 / Params.Final_Matte_Gamma) * CleanPlateMaskColor.r;

    // Unmix Screen Color
    SpillAlpha = clamp(SpillAlpha, 0.0, 1.0);
    vec3 FinalColor = (InputColor.rgb - CleanPlateColor.rgb * (1.0 - SpillAlpha) * (1.0 - BlurredMatte) * (Params.Screen_Subtract_Edge) 
                                      - CleanPlateColor.rgb * (1.0 - SpillAlpha) * BlurredMatte * Params.Screen_Subtract_Core);
    // Preserving luma after screen subtraction and fine tune with color exposure
    float DeltaLuma = dot(REC709, InputColor.rgb) - dot(REC709, FinalColor);
    vec3 EdgeExposureAdjust = DeltaLuma * Params.Edge_Spill_Replace_Color;
    vec3 CoreExposureAdjust = DeltaLuma * Params.Core_Spill_Replace_Color;
    FinalColor += (1.0 - BlurredMatte) * Params.Keep_Edge_Luma * EdgeExposureAdjust;
    FinalColor += BlurredMatte * Params.Keep_Core_Luma * CoreExposureAdjust;

    // TODO: Classic Spill Suppressor
    // TODO(Samil): Blue key type
    FinalColor.g = mix(FinalColor.g, min(FinalColor.g, mix(FinalColor.r, FinalColor.b, Params.Spill_RB_Weight)), Params.Spill_Suppress_Weight);

    FinalColor = ColorCorrect(
        FinalColor,
        vec4(Params.Saturation, 1.0),
        vec4(Params.Contrast, 1.0),
        vec4(Params.Gamma, 1.0),
        vec4(Params.Exposure, 1.0),
        vec4(Params.Offset, 0.0),
        Params.Contrast_Center
    );

    // Combine Color and Alpha (Not multiplying with InputColor.a only needed for After Effects)
    FinalColor *= CleanPlateMaskColor.r;
    vec4 Color = vec4(FinalColor, FinalAlphaAdjusted /* * InputColor.a */);
    Color = max(Color, vec4(0));

    // To multiply the mask with different output types.
    vec4 CleanPlateMask = vec4(CleanPlateMaskColor.r, CleanPlateMaskColor.r, CleanPlateMaskColor.r, 1.0);

    if (Params.Output_Type == 0) // Final
    {
        Color.a = mix(Color.a, 1.0, CleanPlateMaskColor.g);
        Color.rgb = mix(Color.rgb, InputColor.rgb, CleanPlateMaskColor.b);
        rt = Color;
    }
    else if (Params.Output_Type == 1) // Input
    {
        rt = InputColor;
    }
    else if (Params.Output_Type == 2) // CleanPlate
    {
        rt = CleanPlateColor;
    }
    else if (Params.Output_Type == 3) // CleanPlateMask
    {
        rt = CleanPlateMaskColor;
    }
    else if (Params.Output_Type == 4) // SoftMatte1
    {
        rt = pow(vec4(SoftAlpha1, SoftAlpha1, SoftAlpha1, 1.0) * CleanPlateMask, vec4(2.2));
    }
    else if (Params.Output_Type == 5) // SoftMatte2
    {
        rt = pow(vec4(SoftAlpha2, SoftAlpha2, SoftAlpha2, 1.0) * CleanPlateMask, vec4(2.2));
    }
    else if (Params.Output_Type == 6) // CombinedSoftMatte
    {
        rt = pow(vec4(SoftAlpha, SoftAlpha, SoftAlpha, 1.0) * CleanPlateMask, vec4(2.2));
    }
    else if (Params.Output_Type == 7) // HardMatte
    {
        float UnblurredHardMatte = texture(Unblurred_Core_Matte, uv).r;
        // float UnblurredHardMatte = Texture2DSample(InHardMatteTexture, InHardMatteTextureSampler, Input.UV).r;
        rt = pow(vec4(UnblurredHardMatte, UnblurredHardMatte, UnblurredHardMatte, 1.0) * CleanPlateMask, vec4(2.2));
        // rt = pow(vec4(FinalBlurMatte, FinalBlurMatte, FinalBlurMatte, 1.0) * CleanPlateMask, 2.2);
    }
    else if (Params.Output_Type == 8) // ShrinkedMatte
    {
        rt = pow(vec4(BlurredMatte, BlurredMatte, BlurredMatte, 1.0) * CleanPlateMask, vec4(2.2));
    }
    else if (Params.Output_Type == 9) // SpillMatte
    {
        // rt = pow(vec4((1 - SpillAlpha) * BlurredMatte, (1 - SpillAlpha) * BlurredMatte, (1 - SpillAlpha) * BlurredMatte, 1.0) * CleanPlateMask, 2.2);
        rt = pow(vec4((1 - SpillAlpha), (1 - SpillAlpha), (1 - SpillAlpha), 1.0) * CleanPlateMask, vec4(2.2));
    }
    else if (Params.Output_Type == 10) // FinalMatte
    {
        rt = pow(vec4(Color.a, Color.a, Color.a, 1.0) * CleanPlateMask, vec4(2.2));
    }
    else if (Params.Output_Type == 11) // FinalColor
    {
        rt = vec4(FinalColor, 1.0) * CleanPlateMask;
    }
}
