// Copyright MediaZ AS. All Rights Reserved.


#include "glm/common.hpp"
#include <type_traits>

#include "BasicMain.h"
#include <mzCommon.h>

#include <atomic>
#include <flatbuffers/flatbuffers.h>
#include <thread>

#include <stb_image.h>
#include <stb_image_write.h>

#include "AppService_generated.h"
#include "Builtins_generated.h"
#include "CleanPlateAcc.frag.spv.dat"
#include "Cyclorama.frag.spv.dat"
#include "Cyclorama.vert.spv.dat"
#include "CycloramaMask.frag.spv.dat"
#include "CycloramaMask.vert.spv.dat"


#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext/quaternion_trigonometric.hpp"
#include "glm/ext/scalar_constants.hpp"
#include "glm/geometric.hpp"
#include "glm/glm.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/euler_angles.hpp>

typedef u32 uint;
#include "CycloDefines.glsl"

namespace mz
{
#undef near
#undef far

struct Vertex : glm::vec3
{
};

static_assert(sizeof(Vertex) == 12);

static glm::mat4 Perspective(f32 fovx, f32 pixelAspectRatio, glm::vec2 sensorSize, glm::vec2 centerShift)
{
    if(glm::vec2(0) == sensorSize)
    {
        sensorSize  = glm::vec2(1);
        centerShift = glm::vec2(0);
    }

    const f32 near = 0.1;
    const f32 far  = 10000;
    const f32  X = 1.f / tanf(glm::radians(fovx * 0.5));
    const f32  Y = -X * (sensorSize.x/sensorSize.y) * pixelAspectRatio;
    const auto S = -centerShift / sensorSize;
    const f32  Z = far / (far - near);
    return glm::mat4(
        glm::vec4(X, 0, 0, 0),
        glm::vec4(0, Y, 0, 0),
        glm::vec4(S.x, S.y, Z, 1.0f),
        glm::vec4(0, 0, -near * Z, 0)
    );
}


struct Plane
{
    glm::vec3 V[4];

    Plane(glm::vec3 P0, glm::vec3 P1, glm::vec3 P2) : Plane(P0, P1, P2, P1 + P2 - P0)
    {
    }

    Plane(glm::vec3 P0, glm::vec3 P1, glm::vec3 P2, glm::vec3 P3) : V{P0,P1,P2,P3} 
    {
        
    }

    Plane(glm::vec3 P0, glm::vec3 P1, glm::vec3 P2, glm::mat4 xf) : Plane(glm::vec4(P0, 1) * xf, glm::vec4(P1, 1) * xf, glm::vec4(P2, 1) * xf)
    {
    }

    void Add(std::vector<Vertex>& vertices, std::vector<glm::uvec3>& indices, bool flip = false)
    {
        const u32 sz = vertices.size();
        vertices.push_back({V[0]});
        vertices.push_back({V[1]});
        vertices.push_back({V[2]});
        vertices.push_back({V[3]});
        indices.push_back(glm::uvec3(sz) + (flip ? glm::uvec3{1, 0, 2} : glm::uvec3{0, 1, 2}));
        indices.push_back(glm::uvec3(sz) + (flip ? glm::uvec3{1, 2, 3} : glm::uvec3{2, 1, 3}));
    }
};

struct Cylinder
{
    std::vector<glm::vec2> V;
    
    Cylinder(f32 angle)
    {
        const u32 RES = 17;
        const f32 SCA = glm::radians(angle) / RES;
        for (u32 i = 0; i <= RES; ++i)
        {
            const f32 c = cos(i * SCA);
            const f32 s = sin(i * SCA);
            V.push_back(glm::vec2(1 - c, 1 - s));
        }
    }

    void Add(std::vector<Vertex>& vertices, std::vector<glm::uvec3>& indices, glm::mat4 xf, bool flip = false)
    {
        for(u32 i = 0; i < V.size() - 1; ++i)
        {
            auto& v0 = V[i + 0];
            auto& v1 = V[i + 1];
            Plane(glm::vec3(v0.x, -.5, v0.y) , glm::vec3(v0.x, .5, v0.y), glm::vec3(v1.x, -.5, v1.y), xf).Add(vertices, indices, flip);
        }
    }
};

struct QSphere
{
    std::vector<std::vector<glm::vec3>> V;

    QSphere(f32 angle)
    {
        const u32 RES = 17;
        const f32 SCAi = glm::radians(angle) / RES;
        const f32 SCAj = glm::pi<f32>() / 2.f / RES;
        for(u32 j = 0; j <= RES; ++j)
        {
            const f32 cy = cos(j * SCAj);
            const f32 sy = sin(j * SCAj);
            std::vector<glm::vec3> vertices;
            for(u32 i = 0; i <= RES; ++i)
            {
                const f32 cx = cos(i * SCAi);
                const f32 sx = sin(i * SCAi);
                vertices.push_back({cy*cx, cy*sx, sy});
            }
            V.push_back(std::move(vertices));
        }
    }

    void Add(std::vector<Vertex>& vertices, std::vector<glm::uvec3>& indices, glm::mat4 xf, bool flip = false)
    {
        const auto base = vertices.size();

        for(u32 j = 0; j < V.size(); ++j)
        {
            for(u32 i = 0; i < V[j].size(); ++i)
            {
                const u32 i0j0 = base +  V.size() * (j + 0) + (i + 0);
                const u32 i0j1 = base +  V.size() * (j + 1) + (i + 0);
                const u32 i1j0 = base +  V.size() * (j + 0) + (i + 1);
                const u32 i1j1 = base +  V.size() * (j + 1) + (i + 1);
                vertices.push_back({glm::vec4(V[j][i], 1) * xf});
                if(i < V[j].size() - 1 && j < V.size() - 1)
                {
                    indices.push_back(!flip ? glm::uvec3{i1j1, i0j0, i1j0} : glm::uvec3{i0j0, i1j1, i1j0});
                    indices.push_back(!flip ? glm::uvec3{i0j1, i0j0, i1j1} : glm::uvec3{i0j0, i0j1, i1j1});
                }
            }
        }
    }
};

static glm::mat4 MakeTransform(glm::vec3 pos, glm::vec3 rot)
{
    rot = glm::radians(rot);
    auto mat = (glm::mat3)glm::eulerAngleZYX(rot.z, -rot.y, -rot.x);
    return glm::lookAtLH(pos, pos + mat[0], mat[2]);
}

template<class T, class U = T>
void AddParam(std::vector<std::unique_ptr<mz::app::TShaderBinding>>& inputs, mz::Args& pins, const char* name)
{
    if(auto val = pins.Get<T>(name))
    {
        inputs.push_back(
            std::make_unique<mz::app::TShaderBinding>(
            mz::app::TShaderBinding{
                .var = name,
                .val = mz::Buffer::From<U>(*val)
            }));
    }
}

template<class T>
void AddParamXF(std::vector<std::unique_ptr<mz::app::TShaderBinding>>& inputs, mz::Args& pins, const char* name, std::function<T(T const&)>&& xf = [](T const& t) { return U(t); })
{
    if(auto val = pins.Get<T>(name))
    {
        inputs.push_back(
            std::make_unique<mz::app::TShaderBinding>(
            mz::app::TShaderBinding{
                .var = name,
                .val = mz::Buffer::From<T>(xf(*val))
            }));
    }
}

template<class T>
void ChangeVal(std::vector<std::unique_ptr<mz::app::TShaderBinding>>& inputs, std::string const& name, T val )
{
    for (auto& in : inputs)
    {
        if (in->var == name)
        {
            in->val = mz::Buffer::From(val);
            return;
        }
    }
}

template<class T>
void AddParam(std::vector<std::unique_ptr<mz::app::TShaderBinding>>& inputs,  const char* name, T val)
{
    inputs.push_back(
        std::make_unique<mz::app::TShaderBinding>(
            mz::app::TShaderBinding{
                .var = name,
                .val = mz::Buffer::From(val)
            }));
}

template<class T>
bool GetValue(std::map<std::string, const mz::fb::Pin*>& pins, std::string const& name, T& dst)
{
    if(auto pin = pins[name])
    {
        if(flatbuffers::IsFieldPresent(pin, mz::fb::Pin::VT_DATA))
        {
            dst = *(T*)pin->data()->Data();
            return true;
        }
    }
    return false;
}

template<class T>
bool GetValue(std::map<std::string, const mz::fb::Pin*>& pins, std::string const& name, std::function<void(T*)>&& cb)
{
    if(auto pin = pins[name])
    {
        if(flatbuffers::IsFieldPresent(pin, mz::fb::Pin::VT_DATA) && pin->data()->size())
        {
            cb((T*)pin->data()->Data());
            return true;
        }
    }
    return false;
}

struct Cyclorama : PinMapping
{
    mz::app::VertexData Verts;

    std::atomic_bool Capturing = false;

    std::atomic_uint CapturedFrameCount = 0;

    mz::fb::TTexture lhs = {};
    mz::fb::TTexture rt = {};
    
    struct FrameData
    {
        std::string Path;
        mz::fb::TTexture Texture;
        mz::fb::TTrack Track;
        glm::vec3 Pos;
        glm::vec3 Rot;

        operator ru<fb::TCaptureData>() const 
        {
            auto re = MakeUnique<fb::TCaptureData>();
            re->path = Path;
            re->track.reset(new fb::TTrack(Track));
            re->pos = (fb::vec3&)Pos;
            re->rot = (fb::vec3&)Rot;
            return re;
        }

        static bool New(fb::TCaptureData&& dat, FrameData& out)
        {
			i32 w, h, n;
			if (auto raw = stbi_load(dat.path.c_str(), &w, &h, &n, 4))
            {
				auto tex = GServices.LoadImage(raw, fb::vec2u(w, h), fb::Format::R8G8B8A8_SRGB, fb::vec2u(w, h), fb::Format::R8G8B8A8_UNORM);
                free(raw);
                out = {
                    .Path = std::move(dat.path),
                    .Texture = tex,
                    .Track = *dat.track,
                    .Pos = (glm::vec3&)dat.pos,
                    .Rot = (glm::vec3&)dat.rot,
                };
                return true;
            }
            return false;
        }
    };

    std::vector<FrameData> CleanPlates;
    // std::map<u32, FrameData> CleanPlates;

    mz::fb::TTrack Track;
    
    f64 Width = 100;
    f64 Height = 100;
    f64 LeftWingLength = 100;
    f64 RightWingLength = 100;

    glm::dvec3 Position =  {};
    glm::dvec3 Rotation =  {};
    glm::dvec3 BL, BR;

    f64 EdgeRoundness = 0.2f;
    bool HasLeftWing  = 1;
    bool HasRightWing = 1;
    f64 LeftWingAngle = 0;
    f64 RightWingAngle = 0;

    void Clear()
    {
        app::TClearTexture clear1, clear2;
        clear1.texture.reset(&rt);
        clear2.texture.reset(&lhs);
        clear1.color = clear2.color = fb::vec4(0,0,0,1);
        GServices.MakeAPICalls(true, clear1, clear2);
        clear1.texture.release();
        clear2.texture.release();
    }

    void CleanCleanPlates()
    {
        for (auto &cp : CleanPlates) GServices.Destroy(cp.Texture);
        CleanPlates.clear();
    }

    void LoadCleanPlates(std::vector<std::unique_ptr<mz::fb::TCaptureData>> data)
    {
        CleanCleanPlates();
        for (auto& capture : data)
        {
            FrameData fd;
            if (FrameData::New(std::move(*capture), fd))
                CleanPlates.emplace_back(std::move(fd));
        }
    }

    void LoadCleanPlates(const mz::fb::CaptureDataArray &data)
    {
        CleanCleanPlates();
        auto captureDataArray = data.data();
        if (!captureDataArray)
        {
            return;
        }
        fb::TCaptureDataArray arr;
        data.UnPackTo(&arr);
        LoadCleanPlates(std::move(arr.data));
    }

    void UpdateCleanPlatesValue()
    {
        fb::TCaptureDataArray arr;
        std::transform(CleanPlates.begin(), CleanPlates.end(), std::back_inserter(arr.data), [](auto& cp) { return cp; });
        GServices.TEvent(TPinValueChanged{ 
            .pin_id = GetPinId("CleanPlates"),
            .value = mz::Buffer::From(arr)
        });
    }

    void GenerateVertices2(std::vector<Vertex>& vertices, std::vector<glm::uvec3>& indices);
    void LoadVertices()
    {
        if(Verts.buf().pid())
        {
            GServices.Destroy(Verts.buf());
        }

        std::vector<Vertex> vertices;
        std::vector<glm::uvec3> indices;

        GenerateVertices2(vertices, indices);

        u32 vsz = vertices.size() * sizeof(vertices[0]);
        u32 isz = indices.size() * sizeof(indices[0]);
        Verts.mutable_buf().mutate_size(vsz + isz);
        Verts.mutable_buf().mutate_usage(mz::fb::BufferUsage::VERTEX_BUFFER | mz::fb::BufferUsage::INDEX_BUFFER);
        Verts.mutate_vertex_offset(0);
        Verts.mutate_index_offset(vsz);
        Verts.mutate_num_indices(indices.size() * 3);
        // Verts.mutate_depth_func(mz::app::DepthFunction::LESS);
        // Verts.mutate_depth_test(true);
        // Verts.mutate_depth_write(true);

        GServices.Create(Verts.mutable_buf());
        u8 *mapping = GServices.Map(Verts.mutable_buf());
        memcpy(mapping, vertices.data(), vsz);
        memcpy(mapping + vsz, indices.data(), isz);
    }
    
    glm::vec3 GetOrigin(u32 OriginPreset)
    {
        switch (OriginPreset)
        {
        default: // Center Back
            return {};
        case 1: // Left Wing Front
            return BL;
        case 2: // Right Wing Front
            return BR;
        }
    }

    void Load(mz::fb::Node const &node)
    {
        auto name2pin = PinMapping::Load(node);
        DestroyTransientResources();

        CleanCleanPlates();

        GetValue<void>(name2pin, "CleanPlates", [this](auto captures) 
        { 
            LoadCleanPlates(*flatbuffers::GetRoot<mz::fb::CaptureDataArray>(captures));
        });

        mz::fb::TTexture src;
        if (GetValue<void>(name2pin, "Video", 
            [&src](auto* bufStart) { flatbuffers::GetRoot<mz::fb::Texture>(bufStart)->UnPackTo(&src); }))
        {
            lhs.height = src.height;
            lhs.width  = src.width;
			lhs.usage  = mz::fb::ImageUsage::SAMPLED | mz::fb::ImageUsage::RENDER_TARGET | mz::fb::ImageUsage::TRANSFER_SRC | mz::fb::ImageUsage::TRANSFER_DST;
            lhs.format = mz::fb::Format::R16G16B16A16_UNORM;
            rt = lhs;
            GServices.Create(lhs);
            GServices.Create(rt);
        }
        GetValue(name2pin, "EdgeRoundness",  EdgeRoundness);
        GetValue(name2pin, "HasLeftWing",    HasLeftWing);
        GetValue(name2pin, "HasRightWing",   HasRightWing);
        GetValue(name2pin, "RightWingAngle", RightWingAngle);
        GetValue(name2pin, "LeftWingAngle",  LeftWingAngle);
        GetValue(name2pin, "LeftWingLength",   LeftWingLength);
        GetValue(name2pin, "RightWingLength",  RightWingLength);
        GetValue(name2pin, "Width",  Width);
        GetValue(name2pin, "Height",  Height);

        LoadVertices();
        RegisterPasses();
    }

    void RegisterPasses()
    {
        GServices.MakeAPICalls(true,
                              app::TRegisterPass{
                                  .key = "Cyclorama_CleanPlateAccumulator" + UUID2STR(NodeId),
                                  .shader = "Cyclorama_CleanPlateAccumulator",
                                  .blend = false,
                              },
                              app::TRegisterPass{
                                  .key = "Cyclorama_" + UUID2STR(NodeId),
                                  .shader = "Cyclorama_Frag",
                                  .vertex_shader = "Cyclorama_Vert",
                                  .blend = true,
                              },
                              app::TRegisterPass{
                                  .key = "Cyclorama_Mask_" + UUID2STR(NodeId),
                                  .shader = "Cyclorama_Mask_Frag",
                                  .vertex_shader = "Cyclorama_Mask_Vert",
                                  .blend = false,
                              });
    }
    void DestroyTransientResources()
    {
        if (lhs.pid)
        {
            GServices.Destroy(lhs);
            lhs = {};
        }
        if (rt.pid)
        {
            GServices.Destroy(rt);
            rt = {};
        }
    }

    Cyclorama() 
    {
        auto defaultTrackBuf = GServices.GetDefaultDataOfType("mz.fb.Track");
        defaultTrackBuf->As<mz::fb::Track>()->UnPackTo(&Track);
    }

    ~Cyclorama()
    {
        DestroyTransientResources();
        GServices.Destroy(Verts.buf());
        for (auto& c : CleanPlates)
        {
            GServices.Destroy(c.Texture);
        }
    }
};

template<class T>
bool NotSame(T a, T b)  {
    if constexpr(std::is_floating_point_v<T>)
    {
        return glm::abs(a-b) > glm::epsilon<T>();
    }
    else
    {
        return a != b;
    }
}


template<class...T>
u32 MakeFlags(mz::Args &pins, const char* var, u32 idx, T&&... tail)
{
    return MakeFlags(pins, var, idx) | MakeFlags(pins, tail...);
}

template<>
u32 MakeFlags<>(mz::Args &pins, const char* var, u32 idx)
{
    auto val = pins.Get<bool>(var);
    return val ? (*val << idx) : 0;
}

void RegisterCyclorama(NodeActionsMap& functions)
{
    auto &actions = functions["Cyclorama.Cyclorama"];
    actions.NodeCreated = [](fb::Node const &node, mz::Args &args, void **ctx) {
        static bool registered = false;
        if (!registered)
        {
            GServices.MakeAPICalls(
                true,
                app::TRegisterShader{.key = "Cyclorama_Frag",
                                     .spirv = ShaderSrc<sizeof(Cyclorama_frag_spv)>(Cyclorama_frag_spv)},
                app::TRegisterShader{.key = "Cyclorama_Vert",
                                     .spirv = ShaderSrc<sizeof(Cyclorama_vert_spv)>(Cyclorama_vert_spv)},
                app::TRegisterShader{.key = "Cyclorama_Mask_Frag",
                                     .spirv = ShaderSrc<sizeof(CycloramaMask_frag_spv)>(CycloramaMask_frag_spv)},
                app::TRegisterShader{.key = "Cyclorama_Mask_Vert",
                                     .spirv = ShaderSrc<sizeof(CycloramaMask_vert_spv)>(CycloramaMask_vert_spv)},
                app::TRegisterShader{.key = "Cyclorama_CleanPlateAccumulator",
                                     .spirv = ShaderSrc<sizeof(CleanPlateAcc_frag_spv)>(CleanPlateAcc_frag_spv)});
            registered = true;
        }
        Cyclorama *c = new Cyclorama();
        c->Load(node);
        *ctx = c;
        auto id = c->GetPinId("Video");
    };

    actions.PinValueChanged = [](void *ctx, mz::fb::UUID const &id, mz::Buffer* value) {
        auto c = static_cast<Cyclorama*>(ctx);
        auto PinName = c->GetPinName(id);

        #define CHECK_DIFF_AND_LOAD_VERTS(name) \
        if(#name == PinName) { \
            auto& val = *(std::remove_reference_t<decltype(c->name)>*)(value->data());\
            if(NotSame(c->name, val)) { \
                c->name = val; \
                c->LoadVertices(); \
            } \
            return; \
        }

        #define CHECK_AND_SET(name) \
        if(#name == PinName) { \
            c->name = *(std::remove_reference_t<decltype(c->name)>*)(value->data());\
            return;\
        }

        #define CHECK_AND_SET_TABLE(name, type) \
        if(#name == PinName) { \
            auto root = value->As<type>();\
            root->UnPackTo(&c->name);\
            return;\
        }

        CHECK_DIFF_AND_LOAD_VERTS(EdgeRoundness);
        CHECK_DIFF_AND_LOAD_VERTS(LeftWingAngle);
        CHECK_DIFF_AND_LOAD_VERTS(LeftWingLength);
        CHECK_DIFF_AND_LOAD_VERTS(RightWingLength);
        CHECK_DIFF_AND_LOAD_VERTS(RightWingAngle);
        CHECK_DIFF_AND_LOAD_VERTS(HasLeftWing);
        CHECK_DIFF_AND_LOAD_VERTS(HasRightWing);

        // CHECK_DIFF_AND_LOAD_VERTS(Size)
        CHECK_DIFF_AND_LOAD_VERTS(Width)
        CHECK_DIFF_AND_LOAD_VERTS(Height)

        CHECK_AND_SET_TABLE(Track, mz::fb::Track)
        CHECK_AND_SET(Position)
        CHECK_AND_SET(Rotation)

        if ("CleanPlates" == PinName)
        {
            // c->LoadCleanPlates(*value->As<mz::fb::CaptureDataArray>());
            return;
        }

        if (!c->Capturing)
        {
            return;
        }
        
        if ("Video" == c->GetPinName(id))
        {
            auto video = value->As<mz::fb::TTexture>();
            if (c->CapturedFrameCount++ < 50)
            {
                app::TRunPass pass;
                pass.pass = "Cyclorama_CleanPlateAccumulator" + UUID2STR(c->NodeId);
                pass.output = std::make_unique<mz::fb::TTexture>(c->rt);
                pass.inputs.emplace_back(new mz::app::TShaderBinding{.var = "lhs", .val = mz::Buffer::From(c->lhs)});
                pass.inputs.emplace_back(new mz::app::TShaderBinding{.var = "rhs", .val = mz::Buffer::From(video)});
                GServices.MakeAPICall(pass, true);
                std::swap(c->rt, c->lhs);
                auto id = c->GetPinId("Video");
            }
            else
            {
                const u32 idx = c->CleanPlates.size();
                std::string path = "R:/Reality/Assets/CleanPlates/" + std::to_string(idx) + UUID2STR(c->NodeId) + ".png";
                fb::TTexture tex = c->lhs;
                fb::TTexture tmp = c->lhs;
                fb::TTexture srgb = c->lhs;
                tex.format  = fb::Format::R8G8B8A8_UNORM;
                srgb.format = fb::Format::R8G8B8A8_SRGB;
                GServices.Create(tex);
                GServices.Create(srgb);
                GServices.Create(c->lhs);
                c->CleanPlates.push_back({  .Path = path,
                                            .Texture = tex,
                                            .Track = c->Track,
                                            .Pos = c->Position,
                                            .Rot = c->Rotation,});
                std::thread([services=GServices, tmp, tex, srgb, path=std::move(path)] 
                {
                    services.Blit(tmp, tex);
                    services.Blit(tmp, srgb);
                    auto buf = services.Download(srgb);
                    auto re = stbi_write_png(path.c_str(), tex.width, tex.height, 4, services.Map(buf), tex.width * 4);
                    services.Destroy(buf);
                    services.Destroy(tmp);
                    services.Destroy(srgb);
                }).detach();
                c->Capturing = false;
                c->CapturedFrameCount = 0;
                c->UpdateCleanPlatesValue();
				flatbuffers::FlatBufferBuilder fbb;
				GServices.HandleEvent(CreateAppEvent(fbb, mz::app::CreateScheduleRequest(fbb, mz::app::ScheduleRequestKind::PIN, &id, true)));
            }
        }
    };

    actions.EntryPoint = [](mz::Args &pins, void *ctx) {
        auto c = (Cyclorama *)ctx;
        auto renderPinData = pins.Get<mz::fb::Texture>("Render");

        app::TRunPass2 pass;
        std::vector<std::unique_ptr<app::TShaderBinding>> inputs;
        pass.wireframe = *pins.Get<bool>("Wireframe");
        pass.pass = "Cyclorama_" + UUID2STR(c->NodeId);
        pass.output = std::make_unique<mz::fb::TTexture>();
        
        auto trackBuffer = pins.GetBuffer("Track");
        auto trackTbl = trackBuffer->As<mz::fb::Track>();
        mz::fb::TTrack track;
        trackTbl->UnPackTo(&track);
        glm::dvec3 pos = (glm::dvec3 &)track.location;
        glm::dvec3 rot = (glm::dvec3 &)track.rotation;
        auto view = MakeTransform(pos, rot);
        auto prj = Perspective((f32)track.fov, (f32)track.pixel_aspect_ratio, (glm::dvec2&)track.sensor_size, (glm::dvec2&)track.center_shift);
        // glm::vec3 msize = *pins.Get<glm::dvec3>("Size");
        glm::vec3 mpos = *pins.Get<glm::dvec3>("Position");
        glm::vec3 mrot = glm::radians(*pins.Get<glm::dvec3>("Rotation"));

        // c->Size = msize;
        c->Position = mpos;
        c->Rotation = mrot;
        c->Track = std::move(track);

        glm::mat4 model = glm::eulerAngleZYX(mrot.z, mrot.y, mrot.x) * glm::mat4(100.f);
        model[3] = glm::vec4(mpos, 1.f);

        std::vector<u8> MVP_val  = mz::Buffer::From(prj* view * model);
        
        auto Offset =  c->GetOrigin(*pins.Get<u32>("OriginPreset"));
        inputs.emplace_back(new app::TShaderBinding{.var = "VOffset", .val = mz::Buffer::From(Offset) });
        inputs.emplace_back(new app::TShaderBinding{.var = "MVP", .val = MVP_val});

        AddParam<f32>(inputs, pins, "SmoothnessCurve");

    
        renderPinData->UnPackTo(pass.output.get());

        {
            std::unique_ptr<mz::app::TDrawCall> call(new app::TDrawCall());
            call->verts = std::make_unique<mz::app::VertexData>(c->Verts);
            call->inputs = std::move(inputs);
            call->inputs.emplace_back(new app::TShaderBinding{ .var = "UVSmoothness", .val = mz::Buffer::From(0.f) });
            call->inputs.emplace_back(new app::TShaderBinding{ .var = "CP_MVP", .val = MVP_val });
            call->inputs.emplace_back(new app::TShaderBinding{ .var = "Source", .val = mz::Buffer::From(GServices.Color(*pins.Get<mz::fb::vec4>("CycloramaColor"))) });
            pass.draws.emplace_back(std::move(call));
        }
        
        for(auto& last : c->CleanPlates)
        {
            std::unique_ptr<mz::app::TDrawCall> call(new app::TDrawCall());
            call->verts = std::make_unique<mz::app::VertexData>(c->Verts);
            call->verts->mutate_depth_write(false);
            call->verts->mutate_depth_test(false);
            call->verts->mutate_depth_func(app::DepthFunction::NEVER);

            glm::vec3 pos = (glm::dvec3&)last.Track.location;
            glm::vec3 rot = (glm::dvec3&)last.Track.rotation;
            glm::mat4 model = glm::eulerAngleZYX(last.Rot.z, last.Rot.y, last.Rot.x) * glm::mat4(100.f);
            model[3] = glm::vec4(last.Pos, 1.f);
            
            call->inputs.emplace_back(new app::TShaderBinding{ .var = "UVSmoothness", .val = mz::Buffer::From(.05f) });
            std::vector<u8> CP_MVP_val = mz::Buffer::From(Perspective((f32)last.Track.fov, (f32) last.Track.pixel_aspect_ratio, (glm::dvec2&)last.Track.sensor_size, (glm::dvec2&)last.Track.center_shift) * MakeTransform(pos, rot) * model);
            call->inputs.emplace_back(new app::TShaderBinding{ .var = "CP_MVP", .val = CP_MVP_val });
            call->inputs.emplace_back(new app::TShaderBinding{ .var = "Source", .val = mz::Buffer::From(last.Texture) });
            pass.draws.emplace_back(std::move(call));
        }
        
        GServices.MakeAPICall(pass, true);

        {
            glm::vec4 smoothness(*pins.Get<f32>("BottomSmoothness"),
                                 *pins.Get<f32>("LeftSmoothness"), 
                                 *pins.Get<f32>("TopSmoothness"),
                                 *pins.Get<f32>("RightSmoothness"));
            glm::vec4 crop      (*pins.Get<f32>("BottomCrop"),
                                 *pins.Get<f32>("LeftCrop"), 
                                 *pins.Get<f32>("TopCrop"),
                                 *pins.Get<f32>("RightCrop"));
            glm::vec2 diag      (*pins.Get<f32>("DiagonalCrop"),
                                 *pins.Get<f32>("DiagonalSmoothness"));
            auto scale = glm::vec4(c->Width, c->LeftWingLength, c->Height, c->RightWingLength) / 100.f;
            // smoothness = glm::clamp(smoothness / scale, glm::vec4(0), glm::vec4(1));
            // crop = glm::clamp(crop / scale, glm::vec4(0), glm::vec4(1));

            u32 flags = MakeFlags(pins, 
                "SharpEdges", SHARP_EDGES_BIT, 
                "HasLeftWing", HAS_LEFT_WING_BIT, 
                "HasRightWing", HAS_RIGHT_WING_BIT);
            
            app::TRunPass maskPass;
            maskPass.pass = "Cyclorama_Mask_" + UUID2STR(c->NodeId);
            auto maskPinData = pins.Get<mz::fb::Texture>("Mask");
            maskPass.output = std::make_unique<mz::fb::TTexture>();
            maskPinData->UnPackTo(maskPass.output.get());
            maskPass.verts = std::make_unique<mz::app::VertexData>(c->Verts);
            maskPass.inputs.emplace_back(new app::TShaderBinding { .var = "MVP", .val = MVP_val });

            AddParam(maskPass.inputs, "VOffset", Offset);
            AddParam(maskPass.inputs, "Flags", flags);;
            AddParam<f32>(maskPass.inputs, pins, "SmoothnessCurve");
            AddParam<glm::vec4>(maskPass.inputs, pins, "MaskColor");
            AddParam(maskPass.inputs, "Smoothness", smoothness / 100.f);
            AddParam(maskPass.inputs, "SmoothnessCrop", crop / 100.f);
            AddParam(maskPass.inputs, "Scale", scale);
            AddParam(maskPass.inputs, "Roundness", (f32)c->EdgeRoundness / 100.F);
            AddParam(maskPass.inputs, "Angle", glm::radians(glm::vec2(c->LeftWingAngle, c->RightWingAngle)));
            AddParam(maskPass.inputs, "Diag", diag / 100.f);
            
            
            GServices.MakeAPICall(maskPass, true);
        }
        return true;
    };

    actions.NodeRemoved = [](void *ctx, mz::fb::UUID const &id) { delete (Cyclorama *)ctx; };

    actions.NodeFunctions["AddProjection"] = [](mz::Args &pins, mz::Args &functionParams, void *ctx) {
        auto c = (Cyclorama *)ctx;
        if(c->Capturing) return;
        c->Capturing = true;
        c->CapturedFrameCount = 0;
        flatbuffers::FlatBufferBuilder fbb;
        auto id = c->GetPinId("Video");
        GServices.HandleEvent(CreateAppEvent(fbb, mz::app::CreateScheduleRequest(fbb, mz::app::ScheduleRequestKind::PIN, &id)));
    };

    actions.NodeFunctions["ClearProjection"] = [](mz::Args &pins, mz::Args &functionParams, void *ctx) {
        auto c = (Cyclorama *)ctx;
        if(c->Capturing) return;
        c->Capturing = false;
        c->CapturedFrameCount = 0;
        c->CleanCleanPlates();
        c->UpdateCleanPlatesValue();
        c->Clear();
    };

    actions.NodeFunctions["ReloadShaders"] = [&actions](mz::Args &pins, mz::Args &functionParams, void *ctx) {
        auto c = (Cyclorama *)ctx;
        system("glslc " MZ_REPO_ROOT "/Plugins/mzBasic/Source/VirtualStudio/Cyclorama/Cyclorama.frag -c -o " MZ_REPO_ROOT "/cyclo.frag");
        system("glslc " MZ_REPO_ROOT "/Plugins/mzBasic/Source/VirtualStudio/Cyclorama/Cyclorama.vert -c -o " MZ_REPO_ROOT "/cyclo.vert");
        system("glslc " MZ_REPO_ROOT "/Plugins/mzBasic/Source/VirtualStudio/Cyclorama/CycloramaMask.frag -c -o " MZ_REPO_ROOT "/cyclo_mask.frag");
        system("glslc " MZ_REPO_ROOT "/Plugins/mzBasic/Source/VirtualStudio/Cyclorama/CycloramaMask.vert -c -o " MZ_REPO_ROOT "/cyclo_mask.vert");
        system("glslc " MZ_REPO_ROOT "/Plugins/mzBasic/Source/VirtualStudio/Cyclorama/CleanPlateAcc.frag -c -o " MZ_REPO_ROOT "/cp.frag");
        auto frag = ReadSpirv(MZ_REPO_ROOT "/cyclo.frag");
        auto vert = ReadSpirv(MZ_REPO_ROOT "/cyclo.vert");
        auto mask_frag = ReadSpirv(MZ_REPO_ROOT "/cyclo_mask.frag");
        auto mask_vert = ReadSpirv(MZ_REPO_ROOT "/cyclo_mask.vert");
        auto cp_frag = ReadSpirv(MZ_REPO_ROOT "/cp.frag");

        GServices.MakeAPICalls(
            true,
            app::TRegisterShader{.key = "Cyclorama_Frag", .spirv = frag },
            app::TRegisterShader{.key = "Cyclorama_Vert", .spirv = vert},
            app::TRegisterShader{.key = "Cyclorama_Mask_Frag", .spirv = mask_frag},
            app::TRegisterShader{.key = "Cyclorama_Mask_Vert", .spirv = mask_vert}, 
            app::TRegisterShader{.key = "Cyclorama_CleanPlateAccumulator", .spirv = cp_frag}, 
            app::TRegisterPass{
                                  .key = "Cyclorama_CleanPlateAccumulator" + UUID2STR(c->NodeId),
                                  .shader = "Cyclorama_CleanPlateAccumulator",
                                  .blend = false,
            },
            app::TRegisterPass{
                .key = "Cyclorama_" + UUID2STR(c->NodeId),
                .shader = "Cyclorama_Frag",
                .vertex_shader = "Cyclorama_Vert",
                .blend = true,
            },
            app::TRegisterPass{
                .key = "Cyclorama_Mask_" + UUID2STR(c->NodeId),
                .shader = "Cyclorama_Mask_Frag",
                .vertex_shader = "Cyclorama_Mask_Vert",
                .blend = false, 
            });
    };
}

struct XF: glm::mat4
{
    using glm::mat4::mat4;
    XF(glm::mat4 m = glm::mat4(1.f)) : glm::mat4(m) {}


    XF ROT(f32 Angle, u32 i, u32 j) const
    {
        XF const& A = *this;
        const f32 s = sin(Angle);
        const f32 c = cos(Angle);
        XF M = A;
		M[i] = A[i] * c + A[j] * s;
		M[j] = A[j] * c - A[i] * s;
        return M;
    }

    XF RX(f32 Angle) const
    {
        return ROT(Angle, 1, 2);
    }

    XF RY(f32 Angle) const
    {
        return ROT(Angle, 2, 0);
    }

    XF RZ(f32 Angle) const
    {
        return ROT(Angle, 0, 1);
    }

    // XF Rotate(glm::vec3 Axis, f32 Angle) const
    // {
    //     return glm::mat4(*this) * (glm::mat4)glm::angleAxis(glm::radians(Angle), Axis);
    // }

    XF Translate(f32 x, f32 y, f32 z) const
    {
        return Translate(glm::vec3(x, y, z));
    }
    
    XF Translate(glm::vec2 Pos) const
    {
        return Translate(glm::vec3(Pos, 0));
    }

    XF Translate(glm::vec3 Pos) const
    {
        XF re = *this;
        re[0][3]+=Pos.x;
        re[1][3]+=Pos.y;
        re[2][3]+=Pos.z;
        return re;
    }

    XF Scale(f32 u) const
    {
        return Scale(glm::vec3(u));
    }

    XF Scale(f32 x, f32 y, f32 z) const
    {
        return Scale(glm::vec3(x, y, z));
    }

    XF Scale(glm::vec3 Scale) const
    {
        glm::mat4 m(1);
        m[0][0] = Scale.x;
        m[1][1] = Scale.y;
        m[2][2] = Scale.z;
        return glm::mat4(*this) * m;
    }
};

void MakeSymmetry(
    std::vector<Vertex> &vertices, 
    std::vector<glm::uvec3> &indices,
    f64 R, f64 A, f64 W, f64 H, f64 L, bool flip)
{
    const f64 COS = -cos(A);
    const f64 SIN = +sin(A);
    const f64 COT = abs(1. /  tan(A * .5));
    
    const auto M = glm::outerProduct(glm::dvec2(SIN, COS), glm::dvec2(R * COT, L));

    const glm::vec3 P0 = glm::vec3(0, .5 * W, 0) + glm::vec3(M[0], H);
    const glm::vec3 P1 = glm::vec3(0, .5 * W, 0) + glm::vec3(M[1], H);
    const glm::vec3 P2 = glm::vec3(0, .5 * W, 0) + glm::vec3(M[0], R);
    
    const f32 ref =  flip ? -1 : 1;

    // Right Plane
    Plane(P0, P1, P2, XF()
            .Scale(1, ref, 1))
            .Add(vertices, indices, flip);

    // Right Horizontal
    Cylinder(90.f).Add(vertices, indices,
        XF()
        .Scale (R, L - R * COT, R)
        .RZ(glm::pi<f32>() - A)
        .Translate(glm::vec2(P0 + P1) * .5f)
        .Scale(1, ref, 1), flip
    );
    
    // Right Vertical
    Cylinder(180 - glm::degrees(A))
        .Add(vertices, indices,
            XF().Scale(R, H - R, R)
            .RX(glm::pi<f32>() * -.5)
            .Translate(glm::vec3(0, .5 * W + R * (1-COT),  .5 * (H + R)))
            .Scale(1, ref, 1), flip
    );

    // Right Sphere
    QSphere(180 - glm::degrees(A))
        .Add(vertices, indices,
            XF().Scale(R)
            .RX(glm::pi<f32>())
            .RZ(glm::pi<f32>())
            .Translate(R, .5 * W - R * COT, +R)
            .Scale(1, ref, 1), flip
    );
}


void Cyclorama::GenerateVertices2(std::vector<Vertex> &vertices, std::vector<glm::uvec3> &indices)
{
    const f64 R = EdgeRoundness / 100.;
    const f64 W = Width / 100.;
    const f64 H = Height / 100.;
    const auto L = glm::dvec2(LeftWingLength, RightWingLength) / 100.;

    const auto ANG = glm::radians(glm::dvec2(LeftWingAngle, RightWingAngle));
    const auto SIN = glm::sin(ANG);
    const auto COS = glm::cos(ANG);

    const auto WIN = R * glm::dvec2(HasLeftWing, HasRightWing);
    const auto COT = WIN * abs(1. / glm::tan(.5*ANG));
    
    const auto LL = L - R + WIN - COT;
    const auto LX = LL * SIN + R;
    const auto LY = LL * COS + COT - .5 * W;
    this->BL = glm::vec3(+LX.x, +LY.x, 0); 
    this->BR = glm::vec3(+LX.y, -LY.y, 0);
    // Bottom Plane
    Plane(glm::vec3(R, +COT.x -.5 * W, 0), glm::vec3(R, -COT.y +.5 * W, 0), BL,  BR).Add(vertices, indices);

    // Back Plane
    Plane(
        glm::vec3(0, COT.x - .5 * W, H),
        glm::vec3(0, .5 * W - COT.y, H),
        glm::vec3(0, COT.x - .5 * W, R))
        .Add(vertices, indices);

    // Back
     Cylinder(90.f).Add(vertices, indices, XF()
         .Scale(glm::vec3(R, W - (COT.x + COT.y), R))
         .Translate(glm::vec3(0, (COT.x - COT.y) * .5, 0 )
         ));

    if(HasLeftWing)
    {
        MakeSymmetry(vertices, indices, R, ANG.x, W, H, L.x, true);
    }

    if(HasRightWing)
    {
        MakeSymmetry(vertices, indices, R, ANG.y, W, H, L.y, false);
    }
}

} // namespace mz
