#version 450

layout(binding = 0) uniform sampler2D Input;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

#include "../../Shaders/ShaderCommon.glsl"

#define KERNEL_WIDTH 11
#define KERNEL_SIZE (KERNEL_WIDTH * KERNEL_WIDTH)
#define QUADRANT_WIDTH (KERNEL_WIDTH / 2)
#define QUADRANT_SIZE (QUADRANT_WIDTH * QUADRANT_WIDTH)

vec2 MeanAndStdDev(float[QUADRANT_SIZE] Data)
{
    float Mean = 0.0;
    for (uint i = 0; i < QUADRANT_SIZE; i++)
    {
        Mean += Data[i];
    }
    Mean /= float(QUADRANT_SIZE);
    float Variance = 0.0;
    for (uint i = 0; i < QUADRANT_SIZE; i++)
    {
        float Dev = Data[i] - Mean;
        Variance += Dev * Dev;
    }
    Variance /= float(QUADRANT_SIZE);
    return vec2(Mean, sqrt(Variance));
}

void main()
{
    vec2 TexelSize = 1.0 / vec2(textureSize(Input, 0));

    float VQuadrant1[QUADRANT_SIZE];
    float VQuadrant2[QUADRANT_SIZE];
    float VQuadrant3[QUADRANT_SIZE];
    float VQuadrant4[QUADRANT_SIZE];

    for (int i = -QUADRANT_WIDTH; i <= QUADRANT_WIDTH; i++)
    {
        for (int j = -QUADRANT_WIDTH; j <= QUADRANT_WIDTH; j++)
        {
            vec4 Color = texture(Input, uv + vec2(i, j) * TexelSize);
            float V = RGB2HSV(Color.rgb).z;
            int IndexInQuadrant = (abs(i) - 1) + (abs(j) - 1) * QUADRANT_WIDTH;
            if (i > 0 && j > 0)
            {
                VQuadrant1[IndexInQuadrant] = V;
            }
            else if (i < 0 && j > 0)
            {
                VQuadrant2[IndexInQuadrant] = V;
            }
            else if (i < 0 && j < 0)
            {
                VQuadrant3[IndexInQuadrant] = V;
            }
            else if (i > 0 && j < 0)
            {
                VQuadrant4[IndexInQuadrant] = V;
            }
        }
    }

    vec2 MeanAndStdDevQ1 = MeanAndStdDev(VQuadrant1);
    vec2 MeanAndStdDevQ2 = MeanAndStdDev(VQuadrant2);
    vec2 MeanAndStdDevQ3 = MeanAndStdDev(VQuadrant3);
    vec2 MeanAndStdDevQ4 = MeanAndStdDev(VQuadrant4);

    float VQuadrant1_Mean = MeanAndStdDevQ1.x;
    float VQuadrant2_Mean = MeanAndStdDevQ2.x;
    float VQuadrant3_Mean = MeanAndStdDevQ3.x;
    float VQuadrant4_Mean = MeanAndStdDevQ4.x;

    float VQuadrant1_StdDev = MeanAndStdDevQ1.y;
    float VQuadrant2_StdDev = MeanAndStdDevQ2.y;
    float VQuadrant3_StdDev = MeanAndStdDevQ3.y;
    float VQuadrant4_StdDev = MeanAndStdDevQ4.y;
    
    // Select the mean of the quadrant with the lowest standard deviation
    float MinStdDev = VQuadrant1_StdDev;
    float SelectedMean = VQuadrant1_Mean;
    if (VQuadrant2_StdDev < MinStdDev)
    {
        MinStdDev = VQuadrant2_StdDev;
        SelectedMean = VQuadrant2_Mean;
    }
    if (VQuadrant3_StdDev < MinStdDev)
    {
        MinStdDev = VQuadrant3_StdDev;
        SelectedMean = VQuadrant3_Mean;
    }
    if (VQuadrant4_StdDev < MinStdDev)
    {
        MinStdDev = VQuadrant4_StdDev;
        SelectedMean = VQuadrant4_Mean;
    }

    vec4 Sample = texture(Input, uv);
    vec3 HSV = RGB2HSV(Sample.rgb);
    HSV.z = SelectedMean;
    rt = vec4(HSV2RGB(HSV), Sample.a);
}