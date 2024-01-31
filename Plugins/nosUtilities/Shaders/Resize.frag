#version 450

layout(binding = 0) uniform sampler2D Input;

layout(binding = 1) uniform Block
{
    uint Method;
}
Params;

layout(push_constant) uniform constants
{
    uvec2 ViewportSize;
}
PushConstants;

layout(location = 0) out vec4 OutColor;
layout(location = 0) in vec2 uv;

#define PI 3.14159265359
#define PI2_3 2.09439510239
#define PI_3 1.0471975512

float Luma(vec3 Color)
{
    // Rec 709 function for luma.
    return dot(Color, vec3(0.2126, 0.7152, 0.0722));
}

float Gaussian(float Scale, vec2 Offset)
{
    return exp2(Scale * dot(Offset, Offset));
}

bool Test(float x, float lo, float hi)
{
    x = fract(x);
    return x < hi && x > lo;
}

bool TestE(float x, float e)
{
    x = fract(abs(x));
    return x <= fract(abs(e));
}

vec4 BilinearSample(ivec2 uv[4], vec2 F)
{
    return mix(mix(texelFetch(Input, uv[0], 0), texelFetch(Input, uv[1], 0), F.x), mix(texelFetch(Input, uv[2], 0), texelFetch(Input, uv[3], 0), F.x), F.y);
}

vec4 BiCubicSample(vec2 texPos1, vec2 f)
{
    vec2 w0 = f * ( -0.5 + f * (1.0 - 0.5*f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5*f);
    vec2 w2 = f * ( 0.5 + f * (2.0 - 1.5*f) );
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / w12;

    vec2 texPos0 = texPos1 - vec2(1.0);
    vec2 texPos3 = texPos1 + vec2(2.0);
    vec2 texPos12 = texPos1 + offset12;
    
    vec4 result = vec4(0.0);
    result += texelFetch(Input, ivec2(round(vec2(texPos0.x,  texPos0.y) - .5)), 0) * w0.x * w0.y;
    result += texelFetch(Input, ivec2(round(vec2(texPos12.x, texPos0.y) - .5)), 0) * w12.x * w0.y;
    result += texelFetch(Input, ivec2(round(vec2(texPos3.x,  texPos0.y) - .5)), 0) * w3.x * w0.y;
    result += texelFetch(Input, ivec2(round(vec2(texPos0.x,  texPos12.y) - .5)), 0) * w0.x * w12.y;
    result += texelFetch(Input, ivec2(round(vec2(texPos12.x, texPos12.y) - .5)), 0) * w12.x * w12.y;
    result += texelFetch(Input, ivec2(round(vec2(texPos3.x,  texPos12.y) - .5)), 0) * w3.x * w12.y;
    result += texelFetch(Input, ivec2(round(vec2(texPos0.x,  texPos3.y) - .5)), 0) * w0.x * w3.y;
    result += texelFetch(Input, ivec2(round(vec2(texPos12.x, texPos3.y) - .5)), 0) * w12.x * w3.y;
    result += texelFetch(Input, ivec2(round(vec2(texPos3.x,  texPos3.y) - .5)), 0) * w3.x * w3.y;

    return clamp(result, vec4(0), vec4(1));
}


void main()
{
    const vec2 UVAndScreenPos = uv;
    const vec2 Size = textureSize(Input, 0);
    const vec2 InvSize = 1.0 / Size;
    
    switch (Params.Method)
    {
    case 1:
    {
        vec2 ID =  (uv - .5 * InvSize) * Size;
        vec2 F  = fract(ID);
        ivec2 UV = ivec2(floor(ID));
        ivec2 uvs[4] = ivec2[4](
            UV,
            UV + ivec2(1, 0),
            UV + ivec2(0, 1),
            UV + 1
        );
        OutColor = BilinearSample(uvs, F);
    } break;
    case 0: 
        OutColor = texelFetch(Input, ivec2(round((uv - .5 * InvSize) * Size)), 0); break;
    case 2:
    {
        vec2 samplePos = uv * Size;
        vec2 texPos1 = floor(samplePos - 0.5) + 0.5;
        vec2 f = samplePos - texPos1;
        OutColor = BiCubicSample(texPos1, f);
    } break;
    case 3: {
        // Directional blur with unsharp mask upsample.
        float X = 0.5;
        vec3 ColorNW = texture(Input, uv + vec2(-X, -X) * InvSize).rgb;
        vec3 ColorNE = texture(Input, uv + vec2(+X, -X) * InvSize).rgb;
        vec3 ColorSW = texture(Input, uv + vec2(-X, +X) * InvSize).rgb;
        vec3 ColorSE = texture(Input, uv + vec2(+X, +X) * InvSize).rgb;

        OutColor.rgb = (ColorNW + ColorNE + ColorSW + ColorSE) * .25;
        float LumaNW = Luma(ColorNW);
        float LumaNE = Luma(ColorNE);
        float LumaSW = Luma(ColorSW);
        float LumaSE = Luma(ColorSE);

        float DirSWMinusNE = LumaSW - LumaNE;
        float DirSEMinusNW = LumaSE - LumaNW;

        vec2 IsoBrightnessDir = vec2(DirSWMinusNE + DirSEMinusNW, DirSWMinusNE - DirSEMinusNW);

        // avoid NaN on zero vectors by adding 2^-24 (float ulp when length==1, and also minimum representable half)
        IsoBrightnessDir *= (0.125 / sqrt(dot(IsoBrightnessDir, IsoBrightnessDir) + 6e-8));

        vec3 ColorN = texture(Input, uv - IsoBrightnessDir * InvSize, 0).rgb;
        vec3 ColorP = texture(Input, uv + IsoBrightnessDir * InvSize, 0).rgb;

        float UnsharpMask = 0.25;
        OutColor.rgb = (ColorN + ColorP) * ((UnsharpMask + 1.0) * 0.5) - (OutColor.rgb * UnsharpMask);
        OutColor.w = 1;
    }
    break;
    case 4: {
        // Lanczos 3
        const vec2 UV = uv * Size + 0.5;

        vec2 tc = floor(UV - 0.5) + 0.5;
        vec2 f = UV - tc + 2;

        // compute at f, f-1, f-2, f-3, f-4, and f-5 using trig angle addition
        vec2 fpi = f * PI, fpi3 = f * (PI / 3.0);
        vec2 s = sin(fpi), s3 = sin(fpi3), c3 = cos(fpi3);
        const float r3 = sqrt(3.0);
        vec2 w0 = (s * s3) / (f * f);
        vec2 w1 = (-s * (s3 - r3 * c3)) / ((f - 1.0) * (f - 1.0));
        vec2 w2 = (s * (-s3 - r3 * c3)) / ((f - 2.0) * (f - 2.0));
        vec2 w3 = (-s * (-2.0 * s3)) / ((f - 3.0) * (f - 3.0));
        vec2 w4 = (s * (-s3 + r3 * c3)) / ((f - 4.0) * (f - 4.0));
        vec2 w5 = (-s * (s3 + r3 * c3)) / ((f - 5.0) * (f - 5.0));

        // use bilinear texture weights to merge center two samples in each dimension
        vec2 W[5];
        W[0] = w0;
        W[1] = w1;
        W[2] = w2 + w3;
        W[3] = w4;
        W[4] = w5;

        vec2 S[5];
        S[0] = InvSize * (tc - 2);
        S[1] = InvSize * (tc - 1);
        S[2] = InvSize * (tc + w3 / W[2]);
        S[3] = InvSize * (tc + 2);
        S[4] = InvSize * (tc + 3);

        OutColor = vec4(0);

        // 5x5 footprint with corners dropped to give 13 texture taps
        OutColor += vec4(texture(Input, vec2(S[0].x, S[2].y), 0).rgb, 1) * W[0].x * W[2].y;

        OutColor += vec4(texture(Input, vec2(S[1].x, S[1].y), 0).rgb, 1) * W[1].x * W[1].y;
        OutColor += vec4(texture(Input, vec2(S[1].x, S[2].y), 0).rgb, 1) * W[1].x * W[2].y;
        OutColor += vec4(texture(Input, vec2(S[1].x, S[3].y), 0).rgb, 1) * W[1].x * W[3].y;

        OutColor += vec4(texture(Input, vec2(S[2].x, S[0].y), 0).rgb, 1) * W[2].x * W[0].y;
        OutColor += vec4(texture(Input, vec2(S[2].x, S[1].y), 0).rgb, 1) * W[2].x * W[1].y;
        OutColor += vec4(texture(Input, vec2(S[2].x, S[2].y), 0).rgb, 1) * W[2].x * W[2].y;
        OutColor += vec4(texture(Input, vec2(S[2].x, S[3].y), 0).rgb, 1) * W[2].x * W[3].y;
        OutColor += vec4(texture(Input, vec2(S[2].x, S[4].y), 0).rgb, 1) * W[2].x * W[4].y;

        OutColor += vec4(texture(Input, vec2(S[3].x, S[1].y), 0).rgb, 1) * W[3].x * W[1].y;
        OutColor += vec4(texture(Input, vec2(S[3].x, S[2].y), 0).rgb, 1) * W[3].x * W[2].y;
        OutColor += vec4(texture(Input, vec2(S[3].x, S[3].y), 0).rgb, 1) * W[3].x * W[3].y;

        OutColor += vec4(texture(Input, vec2(S[4].x, S[2].y), 0).rgb, 1) * W[4].x * W[2].y;

        OutColor /= OutColor.w;
        break;
    }
    case 5: {
        // Gaussian filtered unsharp mask
        vec2 UV = uv * Size;
        vec2 tc = floor(UV) + 0.5;

        // estimate pixel value and derivatives
        OutColor = vec4(0);
        vec3 Laplacian = vec3(0);

        for (int i = -3; i <= 2; ++i)
        {
            for (int j = -3; j <= 2; ++j)
            {
                vec2 TexelOffset = vec2(i, j) + 0.5;

                // skip corners: eliminated entirely by UNROLL
                if (dot(TexelOffset, TexelOffset) > 9)
                    continue;

                vec2 Texel = tc + TexelOffset;
                vec2 Offset = UV - Texel;
                float OffsetSq = 2 * dot(Offset, Offset); // texel loop is optimized for variance = 0.5
                float Weight = exp(-0.5 * OffsetSq);
                vec4 Sample = Weight * vec4(texture(Input, Texel * InvSize).rgb, 1);
                OutColor += Sample;
                Laplacian += Sample.rgb * (OffsetSq - 2);
            }
        }
        OutColor /= OutColor.a;
        Laplacian /= OutColor.a;

        const vec2 RT = PushConstants.ViewportSize;
        float UnsharpScale = 0.5 * (1 - Size.x * Size.y / (RT.x * RT.y));
        OutColor.rgb -= UnsharpScale * Laplacian;
    }
    break;
    }
}
