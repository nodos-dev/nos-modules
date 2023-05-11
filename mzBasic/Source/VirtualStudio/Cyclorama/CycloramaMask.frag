#version 450

#include "CycloDefines.glsl"

layout(location = 0) in vec3 pos;

layout(location = 0) out vec4 rt;
layout(binding = 1) uniform sampler2D Source;
layout (binding = 0) uniform UBO 
{
	mat4 MVP;
    vec4 Smoothness;
    vec4 SmoothnessCrop;
    vec4 MaskColor;
    vec3 VOffset;
    uint Flags;
    float SmoothnessCurve;
    vec4 Scale;
    float Roundness;
    vec2 Angle;
    vec2 Diag;
} ubo;

#define EPSILON (0.00001)

vec4 MixColor(float c)
{
    return mix(ubo.MaskColor, vec4(0), clamp(c, 0, 1));
}

vec4 CalcSmoothness(in float c[5])
{
    const float N = mix(2, 16, ubo.SmoothnessCurve);
    float re = 0;
    for(int i = 0; i < 5; ++i) re += pow(clamp(c[i], 0, 1), N);
    return MixColor(pow(re, 1.0/N));
}

vec2 Project(in vec2 L, in vec2 R, in vec2 P)
{
    L = L - R;
    P = P - R;
    return L * dot(L, P) / dot(L, L) + R;
}

float Distance(vec2 L, vec2 R, vec2 P)
{
    return length(Project(L, R, P) - P);
}

vec2 Point(float t, vec2 L, vec2 R)
{
    return (1-t) * L + t * R;
}

bool InTriangle(vec2 p0, vec2 p1, vec2 p2, vec2 p)
{
    p  -= p0;
    p1 -= p0;
    p2 -= p0;
    const vec2 C = inverse(mat2(p1, p2)) * p;
    return C.x >= -EPSILON && C.y >= -EPSILON && C.x + C.y <= 1 + EPSILON;
}

bool InQuad(vec2 p0, vec2 p1, vec2 p2, vec2 p3, vec2 p)
{
    return InTriangle(p0, p1, p2, p) ||
           InTriangle(p1, p3, p2, p);
}

bool Sign(vec2 a, vec2 b)
{
    return a.y * b.x - a.x * b.y < 0;
}

bool Sign(vec2 p0, vec2 p1, vec2 p)
{
    return Sign(p1 - p0, p - p0);
}

vec2 R(vec2 V) { return vec2(-V.y,V.x); }



float Cross(vec2 u, vec2 v)
{
    return u.x*v.y - u.y*v.x;
}

float SmoothnessLine2(
    in vec2 P0, 
    in vec2 P1, 
    in vec2 D0,
    in vec2 D1,
    in vec2 C, 
    in vec2 S,
    in vec2 P)
{
    C = max(C, vec2(EPSILON));
    S = max(S, vec2(EPSILON));
    
    D0 = normalize(D0);
    D1 = normalize(D1);

    const vec2 O  = P0 - C.x * D0;
    const vec2 p  = P - O;
    const vec2 p0 = P1 - C.y * D1 - O;
    const vec2 p1 = S.x * D0;
    const vec2 p2 = p1 - S.y * D1;

    const float a = Cross(p1, p2);
    const float b = Cross(p, p2) + Cross(p1, p0);
    const float c = Cross(p, p0);

    
    if(dot(D0 - D1, D0 - D1) < EPSILON)
    {
        return 1 - clamp(-c/b, 0, 1);
    }
    return 1 - clamp((-b - sqrt(b*b - 4*a*c))/(2*a), 0, 1);
}

float SmoothnessLine(
    vec2 P0, 
    vec2 P01, 
    vec2 P1, 
    vec2 P10,
    vec2 C, 
    vec2 S,
    vec2 P)
{
    return SmoothnessLine2(P0, P1, P0 - P01, P1 - P10, C, S, P);
}

float DiagSmoothness(vec2 X, vec2 Y, vec2 O, vec3 P)
{
    const vec2 Diag  = 4*ubo.Diag;
    X = normalize(X-O);
    Y = normalize(Y-O);
    const float A = dot(X, Y);
    const vec2 B = 0.5 * (X + Y);
    //return (SmoothnessLine2(O + X - B, O + Y - B, -B, -B, Diag.xx, Diag.yy, P.xy));
    return int(P.z == 0)*(SmoothnessLine2(O + X - B, O + Y - B, -B, -B, Diag.xx, Diag.yy, P.xy));
}

void main()
{   
    const bool HasLeftWing = GetBit(ubo.Flags, HAS_LEFT_WING_BIT);
    const bool HasRightWing = GetBit(ubo.Flags, HAS_RIGHT_WING_BIT);
    const bool SharpEdges = GetBit(ubo.Flags, SHARP_EDGES_BIT);
    
    const vec4 S  = ubo.Scale;
    const vec4 C  = ubo.SmoothnessCrop;
    const vec4 s  = ubo.Smoothness;
    const vec2 A  = ubo.Angle;
    const float R  = ubo.Roundness;

    vec3 POS = pos.xyz;

    const vec2 SIN = sin(A);
    const vec2 COS = cos(A);
    const vec2 WIN = R * vec2(HasLeftWing, HasRightWing);
    const vec2 COT  = WIN * abs(1.0 / tan(A * .5));
    
    const vec2 LL = S.yw - R + WIN - COT;
    const vec2 LX = LL * SIN + R;
    const vec2 LY = LL * COS + COT - .5 * S.x;
    
    const vec2 dr = vec2(SIN.y, -COS.y);
    const vec2 dl = vec2(SIN.x, +COS.x);

    const vec2 OL = vec2(R, +COT.x -.5 * S.x);
    const vec2 OR = vec2(R, -COT.y +.5 * S.x);
    
    const vec2 BR = vec2(LX.y, -LY.y);
    const vec2 BL = vec2(LX.x, +LY.x);
    
    bool RHS = Sign(OR, BR, POS.xy) && Sign(vec2(0, OR.y), OR, POS.xy);
    bool LHS = Sign(BL, OL, POS.xy) && Sign(OL, vec2(0, OL.y), POS.xy);

    if(!InQuad(OL, OR, BL, BR, POS.xy))
    {
        if(RHS) POS.xy = Project(BR, OR, POS.xy);
        if(LHS) POS.xy = Project(BL, OL, POS.xy);
        if(!LHS && !RHS) POS.xy = Project(OL, OR, POS.xy);
    }
    
    vec2 SC;

    {
        const mat2x3 M = mat2x3(
            vec3(!HasLeftWing && !HasRightWing, HasLeftWing,  !HasLeftWing && HasRightWing),
            vec3(!HasLeftWing && !HasRightWing, HasLeftWing && !HasRightWing, HasRightWing)
        );
        const vec2 CL = C.xyw * M;
        const vec2 SR = CL + s.xyw * M;
        const mat2 T0 = transpose(mat2(BL, BL) - mat2(CL.x * dl, SR.x * dl));
        const mat2 T1 = transpose(mat2(BR, BR) - mat2(CL.y * dr, SR.y * dr));
        const vec2 m = (T1[0] - T0[0]) / (T1[1] - T0[1]);
        SC = m * (POS.y - T1[1]) + T1[0];
    }
    
    float Coeff[5] = float[5](0,0,0,0,0);
    Coeff[0] = SmoothnessLine(BR, OR, BL, OL, ubo.SmoothnessCrop.wy, ubo.Smoothness.wy, POS.xy);
    Coeff[1] = (POS.z + C.z + s.z - S.z) / s.z;
    Coeff[2] = int(!HasLeftWing) * (1 - clamp((Distance(OL, BL, POS.xy) - C.y) / s.y, 0, 1)); 
    Coeff[3] = int(!HasRightWing) * (1 - clamp((Distance(OR, BR, POS.xy) - C.w) / s.w, 0, 1));

    if(HasRightWing ^^ HasLeftWing)
    {
        Coeff[4] = HasLeftWing ? DiagSmoothness(OR, BL, BR, POS) : DiagSmoothness(OL, BR, BL, POS);
    }
    
    rt = CalcSmoothness(Coeff);
    rt.a = 1;
}