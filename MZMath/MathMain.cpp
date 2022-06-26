#include "mzMathConfig.h"
#include "stdio.h"
#include "stdlib.h"

template<class T, T(Op)(T,T)>
T FOp(size_t szIn, T** in, T identity)
{
    for(size_t i = 0; i < szIn; ++i) 
        identity = Op(identity, *in[i]);
    return identity;
}

extern "C"
{

void mzMath_API FAdd(size_t szIn, void* in, void* out)
{
    **(float**)out = FOp<float, [](float a, float b) { return a+b; }>(szIn, (float**)in, 0);
}

void mzMath_API __stdcall FMul(size_t szIn, void* in, void* out)
{
    **(float**)out = FOp<float, [](float a, float b) { return a*b; }>(szIn, (float**)in, 1);
}

void mzMath_API __stdcall FSub(size_t szIn, void* in, void* out)
{
    **(float**)out = FOp<float, [](float a, float b) { return a-b; }>(szIn, (float**)in, 1);
}

void mzMath_API __stdcall FDiv(size_t szIn, void* in, void* out)
{
    **(float**)out = FOp<float, [](float a, float b) { return a/b; }>(szIn, (float**)in, 1);
}

void mzMath_API __stdcall MakeVec3(size_t szIn, void* in, void* out)
{
    ((float**)out)[0][0] = *((float**)in)[0];
    ((float**)out)[0][1] = *((float**)in)[1];
    ((float**)out)[0][2] = *((float**)in)[2];
}

void mzMath_API __stdcall CvtVec3ToVec3d(size_t szIn, void* in, void* out)
{
    ((double**)out)[0][0] = ((float**)in)[0][0];
    ((double**)out)[0][1] = ((float**)in)[0][1];
    ((double**)out)[0][2] = ((float**)in)[0][2];
}

}
