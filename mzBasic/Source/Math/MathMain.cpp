// Copyright MediaZ AS. All Rights Reserved.

#include <MediaZ/Plugin.h>
#include "BasicMain.h"
#include "Args.h"
#include "Builtins_generated.h"
#include "flatbuffers/flatbuffers.h"

#include <glm/glm.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <chrono>
#include <type_traits>

#define NO_ARG

#define DEF_OP0(o, n, t) mz::fb::vec##n##t operator o(mz::fb::vec##n##t l, mz::fb::vec##n##t r) { (glm::t##vec##n&)l += (glm::t##vec##n&)r; return (mz::fb::vec##n##t&)l; }
#define DEF_OP1(n, t) DEF_OP0(+, n, t) DEF_OP0(-, n, t) DEF_OP0(*, n, t) DEF_OP0(/, n, t)
#define DEF_OP(t) DEF_OP1(2, t) DEF_OP1(3, t) DEF_OP1(4, t)

DEF_OP(u);
DEF_OP(i);
DEF_OP(d);
DEF_OP(NO_ARG);

namespace mz
{

template<class T, T F(T,T)>
bool BinopEntryGenerator(mz::Args& args, void*)
{
     auto X = args.Get<T>("X");
     auto Y = args.Get<T>("Y");
     auto Z = args.Get<T>("Z");
     *Z = F(*X,*Y);
     return true;
}

template<class T> T Add(T x, T y) { return x+y;}
template<class T> T Sub(T x, T y) { return x-y;}
template<class T> T Mul(T x, T y) { return x*y;}
template<class T> T Div(T x, T y) { return x/y;}

template<class T, int n>
struct vec {
    T v[n];

    vec() = default;

    template<class P>
    vec(const P* p)  : v{}
    {
        v[0] = p->x();
        v[1] = p->y();
        if constexpr(n > 2) v[2] = p->z();
        if constexpr(n > 3) v[3] = p->w();
    }
    
    template<T F(T,T)>
    vec binop(vec r) const
    {
        vec result;
        for(int i = 0; i < n; i++)
            result.v[i] = F(v[i], r.v[i]);
        return result;  
    }
    
    vec operator +(vec v) const {return binop<Add>(v);}
    vec operator -(vec v) const {return binop<Sub>(v);}
    vec operator *(vec v) const {return binop<Mul>(v);}
    vec operator /(vec v) const {return binop<Div>(v);}
};

template<class T, int Dim, vec<T,Dim>F(vec<T,Dim>,vec<T,Dim>)>
bool VecBinopEntryGenerator(mz::Args& args, void*)
{
    auto X = args.Get<vec<T, Dim>>("X");
    auto Y = args.Get<vec<T, Dim>>("Y");
    auto Z = args.Get<vec<T, Dim>>("Z");
    *Z = F(*X, *Y);
    return true;
}

template<class T>
bool ToString(mz::Args& args, void* ctx)
{
    auto s = std::to_string(*args.Get<T>("in"));
    
    *args.GetBuffer("out") = mz::Buffer((u8*)s.data(), s.size()+1);
    return true;
}

bool SampleNodeAddFunc(mz::Args& pins, mz::Args& funcParams, void* ctx)
{
    auto X = funcParams.Get<f64>("X");
    auto Y = funcParams.Get<f64>("Y");
    auto Z = funcParams.Get<f64>("Z");
    *Z = *X + *Y;
    return true;
}


template<class T, u32 I, u32 F = I*2 + 4>
inline auto AddTrackField(flatbuffers::FlatBufferBuilder& fbb, flatbuffers::Table* X, flatbuffers::Table* Y)
{
    auto l = X->GetStruct<T*>(F);
    auto r = Y->GetStruct<T*>(F);
    auto c = (l ? *l : T{}) + (r ? *r : T{});
    fbb.AddStruct(F, &c);
}

template<u32 hi, class F, u32 i = 0>
void FOR(F&& f)
{
    if constexpr(i < hi)
    {
        f.template operator()<i>();
        FOR<hi, F, i +1>(std::move(f));
    }
}

template<class T, class F>
void FieldIterator(F&& f)
{
    FOR<T::Traits::fields_number>([f=std::move(f),ref=T::MiniReflectTypeTable()]<u32 i>() {
        using Type = std::remove_pointer_t<typename T::Traits::template FieldType<i>>;
        f.template operator()<i, Type>(ref->values ? ref->values[i] : 0);
    });
}

bool AddTrack(mz::Args& pins, void*)
{
    flatbuffers::FlatBufferBuilder fbb;
    fb::Track::Builder b(fbb);
    FieldIterator<fb::Track>([&fbb, X=pins.Get<flatbuffers::Table>("X"), Y = pins.Get<flatbuffers::Table>("Y")]<u32 i, class T>(auto){ AddTrackField<T, i>(fbb, X, Y); });
    fbb.Finish(b.Finish());
    *pins.GetBuffer("Z") = fbb.Release();
    return true;
}

bool AddTransform(mz::Args& pins, void*)
{
    FieldIterator<fb::Transform>([X=pins.Get<u8>("X"),Y=pins.Get<u8>("Y"),Z=pins.Get<u8>("Z")]<u32 i, class T>(auto O) { 
        if constexpr (i == 2) (T&)O[Z] = (T&)O[X] * (T&)O[Y]; 
        else (T&)O[Z] = (T&)O[X] + (T&)O[Y]; 
    });
    return true;
}

void RegisterMath(NodeActionsMap& functions)
{
    functions["mz.math.NodeWithFunction"].NodeCreated = [](mz::fb::Node const& node, mz::Args& args, void* ctx)
    {
        std::vector<std::string> demoList = { "Item1","Item2", "AAAAAAAAa", "a" };
        GServices.UpdateItemList("DemoItems", demoList);
    };

    functions["mz.math.U32ToString"].EntryPoint = ToString<u32>;
    functions["mz.math.NodeWithFunction"].NodeFunctions["NodeAsFunctionAddF64"] = SampleNodeAddFunc;
    functions["mz.math.NodeWithFunction"].NodeFunctions["NodeAsFunctionSubF64"] = SampleNodeAddFunc;
    functions["mz.math.Add_f32"].EntryPoint = BinopEntryGenerator<f32, Add>;
    functions["mz.math.Add_f64"].EntryPoint = BinopEntryGenerator<f64, Add>;
    functions["mz.math.Add_i32"].EntryPoint = BinopEntryGenerator<i32, Add>;
    functions["mz.math.Add_u32"].EntryPoint = BinopEntryGenerator<u32, Add>;
    functions["mz.math.Add_i64"].EntryPoint = BinopEntryGenerator<i64, Add>;
    functions["mz.math.Add_u64"].EntryPoint = BinopEntryGenerator<u64, Add>;
    functions["mz.math.Sub_f32"].EntryPoint = BinopEntryGenerator<f32, Sub>;
    functions["mz.math.Sub_f64"].EntryPoint = BinopEntryGenerator<f64, Sub>;
    functions["mz.math.Sub_i32"].EntryPoint = BinopEntryGenerator<i32, Sub>;
    functions["mz.math.Sub_u32"].EntryPoint = BinopEntryGenerator<u32, Sub>;
    functions["mz.math.Sub_i64"].EntryPoint = BinopEntryGenerator<i64, Sub>;
    functions["mz.math.Sub_u64"].EntryPoint = BinopEntryGenerator<u64, Sub>;
    functions["mz.math.Mul_f32"].EntryPoint = BinopEntryGenerator<f32, Mul>;
    functions["mz.math.Mul_f64"].EntryPoint = BinopEntryGenerator<f64, Mul>;
    functions["mz.math.Mul_i32"].EntryPoint = BinopEntryGenerator<i32, Mul>;
    functions["mz.math.Mul_u32"].EntryPoint = BinopEntryGenerator<u32, Mul>;
    functions["mz.math.Mul_i64"].EntryPoint = BinopEntryGenerator<i64, Mul>;
    functions["mz.math.Mul_u64"].EntryPoint = BinopEntryGenerator<u64, Mul>;
    functions["mz.math.Div_f32"].EntryPoint = BinopEntryGenerator<f32, Div>;
    functions["mz.math.Div_f64"].EntryPoint = BinopEntryGenerator<f64, Div>;
    functions["mz.math.Div_i32"].EntryPoint = BinopEntryGenerator<i32, Div>;
    functions["mz.math.Div_u32"].EntryPoint = BinopEntryGenerator<u32, Div>;
    functions["mz.math.Div_i64"].EntryPoint = BinopEntryGenerator<i64, Div>;
    functions["mz.math.Div_u64"].EntryPoint = BinopEntryGenerator<u64, Div>;

    functions["mz.math.Add_vec2"]. EntryPoint = VecBinopEntryGenerator<f32, 2, Add>;
    functions["mz.math.Add_vec2d"].EntryPoint = VecBinopEntryGenerator<f64, 2, Add>;
    functions["mz.math.Add_vec2i"].EntryPoint = VecBinopEntryGenerator<i32, 2, Add>;
    functions["mz.math.Add_vec2u"].EntryPoint = VecBinopEntryGenerator<u32, 2, Add>;
    functions["mz.math.Add_vec3"]. EntryPoint = VecBinopEntryGenerator<f32, 3, Add>;
    functions["mz.math.Add_vec3d"].EntryPoint = VecBinopEntryGenerator<f64, 3, Add>;
    functions["mz.math.Add_vec3i"].EntryPoint = VecBinopEntryGenerator<i32, 3, Add>;
    functions["mz.math.Add_vec3u"].EntryPoint = VecBinopEntryGenerator<u32, 3, Add>;
    functions["mz.math.Add_vec4"]. EntryPoint = VecBinopEntryGenerator<f32, 4, Add>;
    functions["mz.math.Add_vec4d"].EntryPoint = VecBinopEntryGenerator<f64, 4, Add>;
    functions["mz.math.Add_vec4i"].EntryPoint = VecBinopEntryGenerator<i32, 4, Add>;
    functions["mz.math.Add_vec4u"].EntryPoint = VecBinopEntryGenerator<u32, 4, Add>;
    functions["mz.math.Sub_vec2"]. EntryPoint = VecBinopEntryGenerator<f32, 2, Sub>;
    functions["mz.math.Sub_vec2d"].EntryPoint = VecBinopEntryGenerator<f64, 2, Sub>;
    functions["mz.math.Sub_vec2i"].EntryPoint = VecBinopEntryGenerator<i32, 2, Sub>;
    functions["mz.math.Sub_vec2u"].EntryPoint = VecBinopEntryGenerator<u32, 2, Sub>;
    functions["mz.math.Sub_vec3"]. EntryPoint = VecBinopEntryGenerator<f32, 3, Sub>;
    functions["mz.math.Sub_vec3d"].EntryPoint = VecBinopEntryGenerator<f64, 3, Sub>;
    functions["mz.math.Sub_vec3i"].EntryPoint = VecBinopEntryGenerator<i32, 3, Sub>;
    functions["mz.math.Sub_vec3u"].EntryPoint = VecBinopEntryGenerator<u32, 3, Sub>;
    functions["mz.math.Sub_vec4"]. EntryPoint = VecBinopEntryGenerator<f32, 4, Sub>;
    functions["mz.math.Sub_vec4d"].EntryPoint = VecBinopEntryGenerator<f64, 4, Sub>;
    functions["mz.math.Sub_vec4i"].EntryPoint = VecBinopEntryGenerator<i32, 4, Sub>;
    functions["mz.math.Sub_vec4u"].EntryPoint = VecBinopEntryGenerator<u32, 4, Sub>;
    functions["mz.math.Mul_vec2"]. EntryPoint = VecBinopEntryGenerator<f32, 2, Mul>;
    functions["mz.math.Mul_vec2d"].EntryPoint = VecBinopEntryGenerator<f64, 2, Mul>;
    functions["mz.math.Mul_vec2i"].EntryPoint = VecBinopEntryGenerator<i32, 2, Mul>;
    functions["mz.math.Mul_vec2u"].EntryPoint = VecBinopEntryGenerator<u32, 2, Mul>;
    functions["mz.math.Mul_vec3"]. EntryPoint = VecBinopEntryGenerator<f32, 3, Mul>;
    functions["mz.math.Mul_vec3d"].EntryPoint = VecBinopEntryGenerator<f64, 3, Mul>;
    functions["mz.math.Mul_vec3i"].EntryPoint = VecBinopEntryGenerator<i32, 3, Mul>;
    functions["mz.math.Mul_vec3u"].EntryPoint = VecBinopEntryGenerator<u32, 3, Mul>;
    functions["mz.math.Mul_vec4"]. EntryPoint = VecBinopEntryGenerator<f32, 4, Mul>;
    functions["mz.math.Mul_vec4d"].EntryPoint = VecBinopEntryGenerator<f64, 4, Mul>;
    functions["mz.math.Mul_vec4i"].EntryPoint = VecBinopEntryGenerator<i32, 4, Mul>;
    functions["mz.math.Mul_vec4u"].EntryPoint = VecBinopEntryGenerator<u32, 4, Mul>;
    functions["mz.math.Div_vec2"]. EntryPoint = VecBinopEntryGenerator<f32, 2, Div>;
    functions["mz.math.Div_vec2d"].EntryPoint = VecBinopEntryGenerator<f64, 2, Div>;
    functions["mz.math.Div_vec2i"].EntryPoint = VecBinopEntryGenerator<i32, 2, Div>;
    functions["mz.math.Div_vec2u"].EntryPoint = VecBinopEntryGenerator<u32, 2, Div>;
    functions["mz.math.Div_vec3"]. EntryPoint = VecBinopEntryGenerator<f32, 3, Div>;
    functions["mz.math.Div_vec3d"].EntryPoint = VecBinopEntryGenerator<f64, 3, Div>;
    functions["mz.math.Div_vec3i"].EntryPoint = VecBinopEntryGenerator<i32, 3, Div>;
    functions["mz.math.Div_vec3u"].EntryPoint = VecBinopEntryGenerator<u32, 3, Div>;
    functions["mz.math.Div_vec4"]. EntryPoint = VecBinopEntryGenerator<f32, 4, Div>;
    functions["mz.math.Div_vec4d"].EntryPoint = VecBinopEntryGenerator<f64, 4, Div>;
    functions["mz.math.Div_vec4i"].EntryPoint = VecBinopEntryGenerator<i32, 4, Div>;
    functions["mz.math.Div_vec4u"].EntryPoint = VecBinopEntryGenerator<u32, 4, Div>;
    
    functions["mz.math.Add_Track"].EntryPoint = AddTrack;
    functions["mz.math.Add_Transform"].EntryPoint = AddTransform;

    functions["mz.math.PerspectiveView"].EntryPoint = [](mz::Args& args, void*) {
        auto fov = *args.Get<f64>("FOV");

        // Sanity checks
        static_assert(alignof(glm::dvec3) == alignof(mz::fb::vec3d));
        static_assert(sizeof(glm::dvec3) == sizeof(mz::fb::vec3d));
        static_assert(alignof(glm::dmat4) == alignof(mz::fb::mat4d));
        static_assert(sizeof(glm::dmat4) == sizeof(mz::fb::mat4d));

        // glm::dvec3 is compatible with mz::fb::vec3d so it's safe to cast
        auto rot = *args.Get<glm::dvec3>("Rotation"); 
        auto pos = *args.Get<glm::dvec3>("Position"); 
        auto perspective = glm::perspective(fov, 16.0/9.0, 10.0, 10000.0);
        auto view = glm::eulerAngleXYZ(rot.x, rot.y, rot.z);
        *args.Get<glm::dmat4>("Transformation") = perspective * view;
        return true;
    };

    functions["mz.math.SineWave"].EntryPoint = [](mz::Args& args, void*) {
        float frequency = *args.Get<float>("Frequency");
        float amplitude = *args.Get<float>("Amplitude");
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        float sec = millis / 1000.f;
        *args.Get<float>("Out") = amplitude * sin(frequency * sec);
        return true;
    };

    functions["mz.math.Clamp"].EntryPoint = [](mz::Args& args, void*) {
        auto value = *args.Get<float>("In");
        auto min = *args.Get<float>("Min");
        auto max = *args.Get<float>("Max");
        max = std::max(min, max);
        *args.Get<float>("Out") = std::clamp(value, min, max);
        return true;
    };

    functions["mz.math.Absolute"].EntryPoint = [](mz::Args& args, void*) {
        auto value = *args.Get<float>("In");
        *args.Get<float>("Out") = std::abs(value);
        return true;
    };
}

}