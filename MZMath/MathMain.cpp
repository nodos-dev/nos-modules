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
    **(float**)out = FOp<float, [](float a, float b) { return a+b; }>(szIn, (float**)in, 1);
}

void mzMath_API __stdcall FSub(size_t szIn, void* in, void* out)
{
    **(float**)out = FOp<float, [](float a, float b) { return a-b; }>(szIn, (float**)in, 1);
}

void mzMath_API __stdcall FDiv(size_t szIn, void* in, void* out)
{
    **(float**)out = FOp<float, [](float a, float b) { return a/b; }>(szIn, (float**)in, 1);
}

}
