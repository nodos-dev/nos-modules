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


#define VEC_POSTFIX_i32 i
#define VEC_POSTFIX_i64 i64
#define VEC_POSTFIX_u32 u
#define VEC_POSTFIX_u64 u64
#define VEC_POSTFIX_f32 
#define VEC_POSTFIX_f64 d

#define EXPAND(...) __VA_ARGS__

#define CONCAT(X,Y)      X##Y
#define CONCAT4(X,Y,Z,W) X##Y##Z##W
#define VEC_NAME_0(N, P) EXPAND(CONCAT(vec##N, P))
#define VEC_NAME(T, N)   EXPAND(VEC_NAME_0(N, VEC_POSTFIX_##T))


#define TYPEDEF_VEC(T) \
    using VEC_NAME(T,2) = vec<T,2>;\
    using VEC_NAME(T,3) = vec<T,3>;\
    using VEC_NAME(T,4) = vec<T,4>;

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
    {
        T lhs = *in[i];
        identity = Op(lhs, identity);
    }
    return identity;
}

#define GENERATE_OP(F, T, I, Op) void mzMath_API F(size_t szIn, void* in, void* out) { **(T**)out = TOp<T, [](T a, T b) -> T { return a Op b; }>(szIn, (T**)in, I);}
#define GENERATE_CVT(from, to)   void mzMath_API CONCAT4(Cast_,from,_To_,to)(size_t, from** in, to** out) { **out = **in; };
#define MAKE_VEC(T,N,V)          void mzMath_API CONCAT(MakeVec_,V)(size_t szIn, T** in, vec<T,N>** out) { **out = vec<T,N>::Load(in); }

#define GENERATE_OP_ALL(T)           \
    EXPAND(GENERATE_OP(CONCAT(Add_, T), T, T(0), +)) \
    EXPAND(GENERATE_OP(CONCAT(Sub_, T), T, T(0), -)) \
    EXPAND(GENERATE_OP(CONCAT(Mul_, T), T, T(1), *)) \
    EXPAND(GENERATE_OP(CONCAT(Div_, T), T, T(1), /)) 

#define GENERATE_OP_ALL_BATCH(T) \
    EXPAND(GENERATE_OP_ALL(T))   \
    EXPAND(GENERATE_OP_ALL(VEC_NAME(T,2))) \
    EXPAND(GENERATE_OP_ALL(VEC_NAME(T,3))) \
    EXPAND(GENERATE_OP_ALL(VEC_NAME(T,4)))

#define GENERATE_CVT_ALL_0(from, to) EXPAND(GENERATE_CVT(from, to))
#define GENERATE_CVT_ALL(from, to) \
    EXPAND(GENERATE_CVT_ALL_0(from, to)) \
    EXPAND(GENERATE_CVT_ALL_0(VEC_NAME(from, 2), VEC_NAME(to, 2))) \
    EXPAND(GENERATE_CVT_ALL_0(VEC_NAME(from, 3), VEC_NAME(to, 3))) \
    EXPAND(GENERATE_CVT_ALL_0(VEC_NAME(from, 4), VEC_NAME(to, 4)))

#define GENERATE_CVT_ALL_CROSS      \
        EXPAND(GENERATE_CVT_ALL(i32, i64))  \
        EXPAND(GENERATE_CVT_ALL(i32, u32))  \
        EXPAND(GENERATE_CVT_ALL(i32, u64))  \
        EXPAND(GENERATE_CVT_ALL(i32, f32))  \
        EXPAND(GENERATE_CVT_ALL(i32, f64))  \
        EXPAND(GENERATE_CVT_ALL(i64, i32))  \
        EXPAND(GENERATE_CVT_ALL(i64, u32))  \
        EXPAND(GENERATE_CVT_ALL(i64, u64))  \
        EXPAND(GENERATE_CVT_ALL(i64, f32))  \
        EXPAND(GENERATE_CVT_ALL(i64, f64))  \
        EXPAND(GENERATE_CVT_ALL(f32, i32))  \
        EXPAND(GENERATE_CVT_ALL(f32, i64))  \
        EXPAND(GENERATE_CVT_ALL(f32, u32))  \
        EXPAND(GENERATE_CVT_ALL(f32, u64))  \
        EXPAND(GENERATE_CVT_ALL(f32, f64))  \
        EXPAND(GENERATE_CVT_ALL(f64, i32))  \
        EXPAND(GENERATE_CVT_ALL(f64, i64))  \
        EXPAND(GENERATE_CVT_ALL(f64, u32))  \
        EXPAND(GENERATE_CVT_ALL(f64, u64))  \
        EXPAND(GENERATE_CVT_ALL(f64, f32)) 

#define MAKE_VEC_ALL(T) \
        EXPAND(MAKE_VEC(T, 2, VEC_NAME(T, 2))) \
        EXPAND(MAKE_VEC(T, 3, VEC_NAME(T, 3))) \
        EXPAND(MAKE_VEC(T, 4, VEC_NAME(T, 4)))

#define ALL_GENERATORS(T)  \
    EXPAND(GENERATE_OP_ALL_BATCH(T)) \
    EXPAND(MAKE_VEC_ALL(T))


extern "C"
{

GENERATE_CVT_ALL_CROSS
ALL_GENERATORS(i32)
ALL_GENERATORS(i64)
ALL_GENERATORS(u32)
ALL_GENERATORS(u64)
ALL_GENERATORS(f32)
ALL_GENERATORS(f64)

void mzMath_API Square(size_t szIn, void* in, void* out)
{
    **(f32**)out = **(f32**)in * **(f32**)in;
}

void mzMath_API SquareRoot(size_t szIn, void* in, void* out)
{
    **(f32**)out = sqrtf(**(f32**)in);
}

void mzMath_API AllProtoTypesNode(size_t szIn, void* in, void* out)
{
}

void mzMath_API AllBuiltinTypesNode(size_t szIn, void* in, void* out)
{
}

void mzMath_API GodNode(size_t szIn, void* in, void* out)
{
}

void mzMath_API EmptyNode(size_t szIn, void* in, void* out)
{
}

}
