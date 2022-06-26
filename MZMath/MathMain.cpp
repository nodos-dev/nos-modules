#include "mzMathConfig.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include <algorithm>


using i32 = int32_t;
using i64 = int64_t;
using u32 = uint32_t;
using u64 = uint64_t;
using f32 = float;
using f64 = double;

template<class T, u32 N>
struct vec 
{ 
    T val[N] = {};
    vec() = default;
    vec(T fill)  {std::fill_n(val, N, fill); }

    static vec Load(T** params) 
    {
        vec v;
        for(u32 i = 0; i < N; ++i) 
            v[i] = *params[i]; 
        return v;
    }

    template<class U>
    vec(vec<U,N> r) { for(u32 i = 0; i < N; ++i) val[i] = r[i];}
    T& operator[](u64 n) { return val[n]; }
    friend vec operator+(vec l, vec r) { for(u32 i = 0; i < N; ++i) l[i] += r[i]; return l;}
    friend vec operator-(vec l, vec r) { for(u32 i = 0; i < N; ++i) l[i] -= r[i]; return l;}
    friend vec operator*(vec l, vec r) { for(u32 i = 0; i < N; ++i) l[i] *= r[i]; return l;}
    friend vec operator/(vec l, vec r) { for(u32 i = 0; i < N; ++i) l[i] /= r[i]; return l;}
};

#define TYPEDEF_VEC(T) \
using T##x2 = vec<T,2>;\
using T##x3 = vec<T,3>;\
using T##x4 = vec<T,4>;

TYPEDEF_VEC(i32)
TYPEDEF_VEC(i64)
TYPEDEF_VEC(u32)
TYPEDEF_VEC(u64)
TYPEDEF_VEC(f32)
TYPEDEF_VEC(f64)

template<class T, T(Op)(T,T)>
T TOp(u64 szIn, T** in, T identity)
{
    for(size_t i = 0; i < szIn; ++i) 
        identity = Op(*in[i], identity);
    return identity;
}

#define DO_FOR_ALL_SCALARS(MACRO) \
        MACRO(i32)                \
        MACRO(i64)                \
        MACRO(u32)                \
        MACRO(u64)                \
        MACRO(f32)                \
        MACRO(f64)

#define GENERATE_OP(F, T, I, Op) void mzMath_API F(size_t szIn, T** in, T** out) { **out = TOp<T, [](T a, T b) -> T { return a Op b; }>(szIn, in, I);}
#define GENERATE_CVT(from, to)   void mzMath_API Cast_##from##_To_##to(size_t, from** in, to** out) { **out = **in; };
#define MAKE_VEC(T, N)           void mzMath_API MakeVec_##T##x##N(size_t szIn, T** in, vec<T,N>** out) { **out = vec<T,N>::Load(in); }

#define GENERATE_OP_ALL(T)           \
    GENERATE_OP(Add_##T, T, T(0), +) \
    GENERATE_OP(Sub_##T, T, T(0), -) \
    GENERATE_OP(Mul_##T, T, T(1), *) \
    GENERATE_OP(Div_##T, T, T(1), /) 

#define GENERATE_OP_ALL_BATCH(T) \
    GENERATE_OP_ALL(T)           \
    GENERATE_OP_ALL(T##x2)       \
    GENERATE_OP_ALL(T##x3)       \
    GENERATE_OP_ALL(T##x4)

#define GENERATE_CVT_ALL(from, to)  \
    GENERATE_CVT(from, to)          \
    GENERATE_CVT(from##x2, to##x2)  \
    GENERATE_CVT(from##x3, to##x3)  \
    GENERATE_CVT(from##x4, to##x4)

#define GENERATE_CVT_ALL_CROSS      \
        GENERATE_CVT_ALL(i32, i64)  \
        GENERATE_CVT_ALL(i32, u32)  \
        GENERATE_CVT_ALL(i32, u64)  \
        GENERATE_CVT_ALL(i32, f32)  \
        GENERATE_CVT_ALL(i32, f64)  \
        GENERATE_CVT_ALL(i64, i32)  \
        GENERATE_CVT_ALL(i64, u32)  \
        GENERATE_CVT_ALL(i64, u64)  \
        GENERATE_CVT_ALL(i64, f32)  \
        GENERATE_CVT_ALL(i64, f64)  \
        GENERATE_CVT_ALL(f32, i32)  \
        GENERATE_CVT_ALL(f32, i64)  \
        GENERATE_CVT_ALL(f32, u32)  \
        GENERATE_CVT_ALL(f32, u64)  \
        GENERATE_CVT_ALL(f32, f64)  \
        GENERATE_CVT_ALL(f64, i32)  \
        GENERATE_CVT_ALL(f64, i64)  \
        GENERATE_CVT_ALL(f64, u32)  \
        GENERATE_CVT_ALL(f64, u64)  \
        GENERATE_CVT_ALL(f64, f32) 

#define MAKE_VEC_ALL(T) \
        MAKE_VEC(T, 2)  \
        MAKE_VEC(T, 3)  \
        MAKE_VEC(T, 4)  \

extern "C"
{

DO_FOR_ALL_SCALARS(GENERATE_OP_ALL_BATCH)
DO_FOR_ALL_SCALARS(MAKE_VEC_ALL)
GENERATE_CVT_ALL_CROSS

}
