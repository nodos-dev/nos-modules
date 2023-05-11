const vec3 REC709 = vec3(0.2126, 0.7152, 0.0722);
const vec3 RGB2Y =
    vec3(
        0.2722287168, // AP1_2_XYZ_MAT[0][1],
        0.6740817658, // AP1_2_XYZ_MAT[1][1],
        0.0536895174  // AP1_2_XYZ_MAT[2][1]
    );

const float PI_FLOAT = 3.1415926535897932384626433832795;

vec3 RGB2YUV(vec3 Color)
{
    float Y = dot(Color, REC709);
    float Cb = ((Color.b - Y) / (1 - REC709.b)) * .5;
	float Cr = ((Color.r - Y) / (1 - REC709.r)) * .5;
    return vec3(Y, Cb, Cr);
}

vec3 YUV2RGB(vec3 Color)
{
    float B = (1 - REC709.b) * Color.g * 2 + Color.r;
	float R = (1 - REC709.r) * Color.b * 2 + Color.r;
    float G = (Color.r - (REC709.r * R) - (REC709.b * B)) / REC709.g;
    return vec3(R, G, B);
}

vec3 HSV2RGB(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// From Sam Hocevar: http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
vec3 RGB2HSV(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 ColorCorrect(vec3 WorkingColor, vec4 Saturation, vec4 Contrast, vec4 Gamma, vec4 Gain, vec4 Offset, vec3 ContrastCenter)
{
	float Luma = dot(WorkingColor, RGB2Y);
	WorkingColor = max(vec3(0), mix(Luma.xxx, WorkingColor, Saturation.xyz * Saturation.w));
	WorkingColor = pow(WorkingColor * (1.0 / ContrastCenter), Contrast.xyz * Contrast.w) * ContrastCenter;
	WorkingColor = pow(WorkingColor, 1.0 / (Gamma.xyz * Gamma.w));
	WorkingColor = WorkingColor * (Gain.xyz * Gain.w) + Offset.xyz + Offset.w;
	return WorkingColor;
}

float KernelTriangle1D(float X, float WhiteRadius, float SoftnessRadius)
{
    float CircleRadius = WhiteRadius + SoftnessRadius;
    float Value = 1 - abs(X);
    Value = Value - (1 - CircleRadius);
    Value = Value / max(SoftnessRadius, 0.00001);
    Value = clamp(Value, 0, 1);
    return Value;
}

float Gaussian2D(float X, float Y, float Sigma)
{
    return 1.0 / (2.0 * PI_FLOAT * Sigma * Sigma) * exp(-(X * X + Y * Y) / (2.0 * Sigma * Sigma));
}

float Linear2SRGBSingle(float Value)
{
    float Result;
    if (Value <= 0.0031308)
        Result = Value * 12.92;
    else
        Result = 1.055 * pow(Value, 1.0 / 2.4) - 0.055;
    return Result;
}

vec4 Linear2SRGB(vec4 Value)
{
    return vec4(
        Linear2SRGBSingle(Value.r),
        Linear2SRGBSingle(Value.g),
        Linear2SRGBSingle(Value.b),
        Value.a);
}

float SRGB2LinearSingle(float Value)
{
    float Result;
    if (Value <= 0.04045)
        Result = Value / 12.92;
    else
        Result = pow((Value + 0.055) / 1.055, 2.4);
    return Result;
}

vec4 SRGB2Linear(vec4 Value)
{
    return vec4(
        SRGB2LinearSingle(Value.r),
        SRGB2LinearSingle(Value.g),
        SRGB2LinearSingle(Value.b),
        Value.a);
}