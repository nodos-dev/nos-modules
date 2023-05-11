#version 450

// Ported from RealityEngine

layout(binding = 0) uniform sampler2D In;
layout(binding = 1) uniform DistortParams
{
    vec2 k1k2;
    vec2 Center;
    float Distortion_Scale;
    uint Distort_Mode;
} Params;

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

float CalculateRFour(float R, vec2 k1k2)
{
	float R2 = R * R;
	float R4 = R2 * R2;
	return (1.0 + k1k2.x * R2 + k1k2.y * R4);
}

void main()
{
    vec2 TextureSize = vec2(textureSize(In, 0));
    vec2 P = uv; // Original point
    int X = int(P.x * TextureSize.x);
    int Y = int(P.y * TextureSize.y);

    int XMin = 0;
    int XMax = int(TextureSize.x - 1);
    int YMin = 0;
    int YMax = int(TextureSize.y - 1);

    bool ShouldCalculate = true;
    if (Params.Distort_Mode == 1) // Crop
    {
        if (X == XMin || X == XMax || Y == YMin || Y == YMax)
        {
            rt = vec4(0.0, 0.0, 0.0, 1.0);
            ShouldCalculate = false;
        }
    }
    else if (Params.Distort_Mode == 2) // Repeat
    {
        if (X == XMax)
        {
            X = XMax - 1;
            P.x = (X + 0.5) / TextureSize.x;
        }
        if (X == XMin)
        {
            X = XMin + 1;
            P.x = (X + 0.5) / TextureSize.x;
        }
        if (Y == YMax)
        {
            Y = YMax - 1;
            P.y = (Y + 0.5) / TextureSize.y;
        }
        if (Y == YMin)
        {
            Y = YMin + 1;
            P.y = (Y + 0.5) / TextureSize.y;
        }
    }

    if (ShouldCalculate)
    {
        P -= Params.Center;
        // Map to [-1, 1]
        vec2 PUniform = (P - 0.5) * 2.0;

        // This is out aspect ratio
        vec2 Aspect = vec2(TextureSize.x / TextureSize.y, 1.0);
        float x = sqrt(1.0 / (Aspect.x * Aspect.x + Aspect.y * Aspect.y));
        vec2 AspectRatio = Aspect * x;

		// Apply aspect ratio.
        vec2 PUniformAspect  = PUniform * AspectRatio;

		// Time to apply distortion
        float R = length(PUniformAspect);
        float RRatio = CalculateRFour(R, Params.k1k2);

        vec2 PUniformAspectFinal = PUniformAspect * RRatio / Params.Distortion_Scale;

		// Reapply aspect ratio.
        vec2 PUniformFinal = PUniformAspectFinal / AspectRatio;

		// And back to [0, 1] domain.
        vec2 PFinal  = PUniformFinal / 2.0 + 0.5;

        PFinal += Params.Center / 2;

        vec4 Color = texture(In, PFinal);
        // Check outside of image domain
        if (PFinal.x < 0.0 || PFinal.x > 1.0 || PFinal.y < 0.0 || PFinal.y > 1.0)
        {
            Color = vec4(0.0);
        }
        rt = Color;
    }
}