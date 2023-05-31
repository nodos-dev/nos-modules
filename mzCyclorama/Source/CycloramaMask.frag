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
    float re = 1;
    for(int i = 0; i < 5; ++i) re *= 1 - clamp(c[i], 0, 1); return MixColor(1-re);
    for(int i = 0; i < 5; ++i) re += pow(clamp(c[i], 0, 1), N); return MixColor(pow(re - 1, 1.0/N));
}

vec2 Project(in vec2 L, in vec2 R, in vec2 P)
{
    L -= R;
    P -= R;
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
    const vec2 Diag = ubo.Diag;
    X = normalize(X-O);
    Y = normalize(Y-O);
    const float A = dot(X, Y);
    const vec2 B = 0.5 * (X + Y);
    //return (SmoothnessLine2(O + X - B, O + Y - B, -B, -B, Diag.xx, Diag.yy, P.xy));
    return int(P.z == 0)*(SmoothnessLine2(O + X - B, O + Y - B, -B, -B, Diag.xx, Diag.yy, P.xy));
}


void swap(inout vec2 a, inout vec2 b)
{
    vec2 tmp = a;
    a = b;
    b = tmp;
}

float DiagCoeff(vec2 B0, vec2 B1, vec2 O0, vec2 O1, vec2 C, vec2 pos, bool flip)
{
    if(flip)
    {
        swap(B0, B1);
        swap(O0, O1);
        C.xy = C.yx;
    }

    const vec2 d0 = B0 - normalize(B0 - O0) * ((ubo.Diag.x *2/ sqrt(2)) + C.y) - normalize(B0 - B1) * C.x;
    const vec2 d1 = B0 - normalize(B0 - B1) * ((ubo.Diag.x *2/ sqrt(2)) + C.x) - normalize(B0 - O0) * C.y;
    const vec2 d2 = B0 - normalize(B0 - O0) * (((ubo.Diag.x + ubo.Diag.y)*2 / sqrt(2)) + C.y) - normalize(B0 - B1) * (C.x);
    const vec2 d3 = B0 - normalize(B0 - B1) * (((ubo.Diag.x + ubo.Diag.y)*2 / sqrt(2)) + C.x) - normalize(B0 - O0) * (C.y);
    const vec2 d0x = d2 + normalize(B0 - O1) * ubo.Diag.y;
    const vec2 d1x = d3 + normalize(B0 - O1) * ubo.Diag.y;
    const vec2 a = Project(d0x, d1x, pos);
    const vec2 b = Project(d2, d3, pos);
    
    if((ubo.Diag.x > EPSILON || ubo.Diag.y > EPSILON) && (((flip ? -1 : 1)*Cross(d1x-d0x, pos-d0x))<0))
    {
        return -1;
    }

    if(((flip ? -1 : 1)*Cross(d3-d2, pos-d2))<0)
    {
        float x0 = length(pos - a);
        float x1 = length(pos - b);
        float x2 = length(a-b);
        return 1-clamp(x0 / x2, 0, 1);
    }

    return 0;
}

void main()
{   
    const bool HasLeftWing = GetBit(ubo.Flags, HAS_LEFT_WING_BIT);
    const bool HasRightWing = GetBit(ubo.Flags, HAS_RIGHT_WING_BIT);
    const bool SharpEdges = GetBit(ubo.Flags, SHARP_EDGES_BIT);
    
    const vec4  S = ubo.Scale.xwzy;
    const vec4  C = ubo.SmoothnessCrop.xwzy;
    const vec4  s = ubo.Smoothness.xwzy;
    const vec2  A = ubo.Angle.yx;
    const float R = ubo.Roundness;

    vec3 POS = pos.xyz;
    
    const vec2 SIN = sin(A);
    const vec2 COS = cos(A);
    const vec2 WIN = R * vec2(HasLeftWing, HasRightWing).yx;
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

    if(!InQuad(OL, OR, BL, BR, POS.xy))
    {
        const bool RHS = Sign(OR, BR, POS.xy) && Sign(vec2(0, OR.y), OR, POS.xy);
        const bool LHS = Sign(BL, OL, POS.xy) && Sign(OL, vec2(0, OL.y), POS.xy);
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

    Coeff[0] = (POS.x - SC.y) / abs(SC.x - SC.y);
    Coeff[1] = (POS.z + C.z + s.z - S.z)/ s.z;
    if(!HasLeftWing  ) Coeff[2] = 1 - clamp((Distance(OL, BL, POS.xy) - C.y) / s.y, 0, 1); 
    if(!HasRightWing ) Coeff[3] = 1 - clamp((Distance(OR, BR, POS.xy) - C.w) / s.w, 0, 1); 

    if((HasRightWing ^^ HasLeftWing))
    {

#if 0  // Debug
        if(
            length(d0x - POS.xy) < 0.025 || length(d1x - POS.xy) < 0.025
        )
        {
            rt = vec4(0, 0, 1, 1);
            return;
        }

        if(
            length(d0 - POS.xy) < 0.025 || length(d1 - POS.xy) < 0.025 ||
            length(d2 - POS.xy) < 0.025 || length(d3 - POS.xy) < 0.025 ||
            length(d0x - POS.xy) < 0.025 || length(d1x - POS.xy) < 0.025
        )
        {
            rt = vec4(1, 1, 1, 1);
            return;
        }
        if(length(POS.xy - b) < 0.005){ rt = vec4(1, 1, 0, 1); return;}
        if(length(POS.xy - a) < 0.005){ rt = vec4(1, 0, 1, 1); return;}
#endif

        if((Coeff[4] = DiagCoeff(BL, BR, OL, OR, C.yw, POS.xy, HasLeftWing)) < 0)
        {
            rt = vec4(0, 0, 0, 1);
            return;
        }
    }

    rt = CalcSmoothness(Coeff);
    rt.a = 1;
}