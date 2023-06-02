// Copyright MediaZ AS. All Rights Reserved.


#include "glm/common.hpp"
#include <type_traits>

#include <MediaZ/Helpers.hpp>

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


MZ_INIT();



namespace mz
{
#undef near
#undef far

MZ_REGISTER_NAME2(CleanPlates);
MZ_REGISTER_NAME2(CaptureFolder);
MZ_REGISTER_NAME2(Video);
MZ_REGISTER_NAME2(EdgeRoundness);
MZ_REGISTER_NAME2(HasLeftWing);
MZ_REGISTER_NAME2(HasRightWing);
MZ_REGISTER_NAME2(RightWingAngle);
MZ_REGISTER_NAME2(LeftWingAngle);
MZ_REGISTER_NAME2(LeftWingLength);
MZ_REGISTER_NAME2(RightWingLength);
MZ_REGISTER_NAME2(Width);
MZ_REGISTER_NAME2(Height);
MZ_REGISTER_NAME2(BottomCrop);
MZ_REGISTER_NAME2(LeftCrop);
MZ_REGISTER_NAME2(RightCrop);
MZ_REGISTER_NAME2(TopCrop);
MZ_REGISTER_NAME2(BottomSmoothness);
MZ_REGISTER_NAME2(LeftSmoothness);
MZ_REGISTER_NAME2(RightSmoothness);
MZ_REGISTER_NAME2(TopSmoothness);
MZ_REGISTER_NAME2(DiagonalCrop);
MZ_REGISTER_NAME2(DiagonalSmoothness);
MZ_REGISTER_NAME2(SmoothnessCurve);
MZ_REGISTER_NAME2(Wireframe);
MZ_REGISTER_NAME2(Render);
MZ_REGISTER_NAME2(Track);
MZ_REGISTER_NAME2(Position);
MZ_REGISTER_NAME2(Rotation);
MZ_REGISTER_NAME2(OriginPreset);
MZ_REGISTER_NAME2(CycloramaColor);
MZ_REGISTER_NAME2(SharpEdges);
MZ_REGISTER_NAME2(MVP);

MZ_REGISTER_NAME2(Mask);
MZ_REGISTER_NAME2(MaskColor);
MZ_REGISTER_NAME2(VOffset);
MZ_REGISTER_NAME2(Flags);
MZ_REGISTER_NAME2(Smoothness);
MZ_REGISTER_NAME2(SmoothnessCrop);
MZ_REGISTER_NAME2(Scale);
MZ_REGISTER_NAME2(Roundness);
MZ_REGISTER_NAME2(Angle);
MZ_REGISTER_NAME2(Diag);
MZ_REGISTER_NAME2(UVSmoothness);
MZ_REGISTER_NAME2(CP_MVP);
MZ_REGISTER_NAME2(Source);

MZ_REGISTER_NAME2(Cyclorama_CleanPlateAccumulator_Pass);
MZ_REGISTER_NAME2(Cyclorama_Main_Pass);
MZ_REGISTER_NAME2(Cyclorama_Mask_Pass);

MZ_REGISTER_NAME2(Cyclorama_CleanPlateAccumulator_Shader);
MZ_REGISTER_NAME2(Cyclorama_Main_Frag);
MZ_REGISTER_NAME2(Cyclorama_Main_Vert);
MZ_REGISTER_NAME2(Cyclorama_Mask_Frag);
MZ_REGISTER_NAME2(Cyclorama_Mask_Vert);
MZ_REGISTER_NAME2(lhs);
MZ_REGISTER_NAME2(rhs);


struct Vertex : glm::vec3
{
};

static_assert(sizeof(Vertex) == 12);

template<class T>
bool NotSame(T a, T b) {
    if constexpr (std::is_floating_point_v<T>)
    {
        return glm::abs(a - b) > glm::epsilon<T>();
    }
    else
    {
        return a != b;
    }
}

template<class...T>
u32 MakeFlags(std::unordered_map<Name, void*> const& pins, Name var, u32 idx, T&&... tail)
{
    return MakeFlags(pins, var, idx) | MakeFlags(pins, tail...);
}

template<>
u32 MakeFlags<>(std::unordered_map<Name, void*> const& pins, Name var, u32 idx)
{
    auto val = GetPinValue<bool>(pins, var);
    return val ? (*val << idx) : 0;
}

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

template<class T>
void AddParam(std::vector<mzShaderBinding>& inputs, std::unordered_map<Name, void*> const& pins, Name name)
{
    if(auto val = GetPinValue<T>(pins, name))
    {
        inputs.push_back({.Name = name, .FixedSize = val});
    }
}

template<class T>
bool GetValue(std::unordered_map<Name, const mz::fb::Pin*>& pins, Name name, T& dst)
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
bool GetValue(std::unordered_map<Name, const mz::fb::Pin*>& pins, Name name, std::function<void(T*)>&& cb)
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
    mzVertexData Verts = {};

    enum 
    {
        CAPTURING = 1,
        SAVING    = 2,
    };

    std::atomic_uint Status = 0;

    std::atomic_uint CapturedFrameCount = 0;

    mzResourceShareInfo lhs = {};
    mzResourceShareInfo rt = {};
    
    struct FrameData
    {
        std::string Path;
        mzResourceShareInfo Texture;
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
                mzResourceShareInfo tex = {};
                tex.Info.Texture.Width = w;
                tex.Info.Texture.Height = h;
                tex.Info.Texture.Format = MZ_FORMAT_R8G8B8A8_UNORM;
				mzEngine.ImageLoad(raw, mzVec2u(w, h), MZ_FORMAT_R8G8B8A8_SRGB, &tex);
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


    std::string CaptureFolder = "R:/Reality/Assets/CleanPlates";
    std::vector<FrameData> CleanPlates;
    std::mutex CPLock;

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
        mzCmd cmd;
        mzEngine.Begin(&cmd);
        mzEngine.Clear(cmd, &rt, mzVec4(0,0,0,1));
        mzEngine.Clear(cmd, &lhs, mzVec4(0,0,0,1));
        mzEngine.End(cmd);
    }

    void CleanCleanPlates()
    {
        for (auto &cp : CleanPlates) mzEngine.Destroy(&cp.Texture);
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
        
        std::vector<u8> buf = mz::Buffer::From(arr);
        auto id = GetPinId(CleanPlates_Name);
        flatbuffers::FlatBufferBuilder fbb;
        mzEngine.HandleEvent(CreateAppEvent(fbb, mz::CreatePinValueChangedDirect(fbb, &id, &buf)));
    }

    void UpdateStatus()
    {
        std::vector<flatbuffers::Offset<mz::fb::NodeStatusMessage>> msg;
        flatbuffers::FlatBufferBuilder fbb;
        if(Status.load() & CAPTURING) msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, "Capturing Clean Plate", fb::NodeStatusMessageType::INFO));
        if(Status.load() & SAVING)    msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, "Saving Clean Plate", fb::NodeStatusMessageType::INFO));
        mzEngine.HandleEvent(CreateAppEvent(
            fbb, mz::CreatePartialNodeUpdateDirect(fbb, &NodeId, ClearFlags::NONE, 0, 0, 0, 0, 0, 0, &msg)));
    }

    void GenerateVertices2(std::vector<Vertex>& vertices, std::vector<glm::uvec3>& indices);
    void LoadVertices()
    {
        if(Verts.Buffer.Memory.PID)
        {
            mzEngine.Destroy(&Verts.Buffer);
        }

        std::vector<Vertex> vertices;
        std::vector<glm::uvec3> indices;

        GenerateVertices2(vertices, indices);

        u32 vsz = vertices.size() * sizeof(vertices[0]);
        u32 isz = indices.size() * sizeof(indices[0]);
        Verts.Buffer.Info.Type = MZ_RESOURCE_TYPE_BUFFER;
        Verts.Buffer.Info.Buffer.Size = vsz + isz;
        Verts.Buffer.Info.Buffer.Usage = mzBufferUsage(MZ_BUFFER_USAGE_VERTEX_BUFFER | MZ_BUFFER_USAGE_INDEX_BUFFER);
        Verts.VertexOffset= 0;
        Verts.IndexOffset = vsz;
        Verts.IndexCount = indices.size() * 3;
        // Verts.mutate_depth_func(mz::app::DepthFunction::LESS);
        // Verts.mutate_depth_test(true);
        // Verts.mutate_depth_write(true);

        mzEngine.Create(&Verts.Buffer);
        u8 *mapping = mzEngine.Map(&Verts.Buffer);
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

        GetValue<void>(name2pin, CleanPlates_Name, [this](auto captures) 
        { 
            LoadCleanPlates(*flatbuffers::GetRoot<mz::fb::CaptureDataArray>(captures));
        });

        GetValue<const char>(name2pin, CaptureFolder_Name, [this](const char* folder)
        { 
            if (std::filesystem::is_directory(folder))
            {
                CaptureFolder = std::filesystem::path(folder).string();
            }
        });

        fb::TTexture src;
        if (GetValue<void>(name2pin, Video_Name, 
            [&src](auto* bufStart) { flatbuffers::GetRoot<mz::fb::Texture>(bufStart)->UnPackTo(&src); }))
        {
            lhs.Info.Texture.Height = src.height;
            lhs.Info.Texture.Width  = src.width;
			lhs.Info.Texture.Usage  = mzImageUsage(MZ_IMAGE_USAGE_SAMPLED | MZ_IMAGE_USAGE_RENDER_TARGET | MZ_IMAGE_USAGE_TRANSFER_SRC | MZ_IMAGE_USAGE_TRANSFER_DST);
            lhs.Info.Texture.Format = MZ_FORMAT_R16G16B16A16_UNORM;
            lhs.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
            rt = lhs;
            mzEngine.Create(&lhs);
            mzEngine.Create(&rt);
        }
        GetValue(name2pin, EdgeRoundness_Name,  EdgeRoundness);
        GetValue(name2pin, HasLeftWing_Name,    HasLeftWing);
        GetValue(name2pin, HasRightWing_Name,   HasRightWing);
        GetValue(name2pin, RightWingAngle_Name, RightWingAngle);
        GetValue(name2pin, LeftWingAngle_Name,  LeftWingAngle);
        GetValue(name2pin, LeftWingLength_Name,   LeftWingLength);
        GetValue(name2pin, RightWingLength_Name,  RightWingLength);
        GetValue(name2pin, Width_Name,  Width);
        GetValue(name2pin, Height_Name,  Height);

        LoadVertices();
    }

    void DestroyTransientResources()
    {
        if (lhs.Memory.PID)
        {
            mzEngine.Destroy(&lhs);
            lhs = {};
        }
        if (rt.Memory.PID)
        {
            mzEngine.Destroy(&rt);
            rt = {};
        }
    }

    Cyclorama() 
    {
        mzBuffer val;
        switch (mzEngine.GetDefaultValueOfType("mz.fb.Track", &val))
        {
        case MZ_RESULT_SUCCESS:
            flatbuffers::GetRoot<fb::Track>(val.Data)->UnPackTo(&Track);
            break;
        default: break;
        }
    }

    ~Cyclorama()
    {
        DestroyTransientResources();
        mzEngine.Destroy(&Verts.Buffer);
        for (auto& c : CleanPlates)
        {
            mzEngine.Destroy(&c.Texture);
        }
    }


    void Validate(std::unordered_map<Name, void*>& pins)
    {
        auto clampAndBroadcast = [&](f32 lo, f32 hi, mzName name) -> f32
        {
            f32& tmp = *GetPinValue<f32>(pins, name);
            if(tmp<lo||tmp>hi) 
            {
                tmp = glm::clamp(tmp, lo, hi);
                mzEngine.SetPinValueByName(NodeId, name, {&tmp, sizeof(tmp)});
            }
            return tmp;
        };

        const f32 Cb = clampAndBroadcast(0, Width - EdgeRoundness, BottomCrop_Name);
        const f32 Cl = clampAndBroadcast(0, LeftWingLength - EdgeRoundness, LeftCrop_Name);
        const f32 Cr = clampAndBroadcast(0, RightWingLength - EdgeRoundness, RightCrop_Name);
        const f32 Ct = clampAndBroadcast(0, Height - EdgeRoundness, TopCrop_Name);
        clampAndBroadcast(0, Width - EdgeRoundness - Cb, BottomSmoothness_Name);
        clampAndBroadcast(0, LeftWingLength - EdgeRoundness - Cl, LeftSmoothness_Name);
        clampAndBroadcast(0, RightWingLength - EdgeRoundness - Cr, RightSmoothness_Name);
        clampAndBroadcast(0, Height - EdgeRoundness - Ct, TopSmoothness_Name);

        const f32 DMax = glm::length(glm::vec2(LeftWingLength - EdgeRoundness - Cl, RightWingLength - EdgeRoundness - Cr)) * 0.5;
        const f32 D = clampAndBroadcast(0, DMax, DiagonalCrop_Name);
        clampAndBroadcast(0, DMax - D, DiagonalSmoothness_Name);
    }

    inline static std::pair<const char*, std::vector<u8>> spirvs[] = 
    {
        {"Cyclorama_Main_Frag",{std::begin(Cyclorama_frag_spv), std::end(Cyclorama_frag_spv)}},
        {"Cyclorama_Main_Vert",{std::begin(Cyclorama_vert_spv), std::end(Cyclorama_vert_spv)}},
        {"Cyclorama_Mask_Frag",{std::begin(CycloramaMask_frag_spv), std::end(CycloramaMask_frag_spv)}},
        {"Cyclorama_Mask_Vert",{std::begin(CycloramaMask_vert_spv), std::end(CycloramaMask_vert_spv)}},
        {"Cyclorama_CleanPlateAccumulator_Shader",{std::begin(CleanPlateAcc_frag_spv), std::end(CleanPlateAcc_frag_spv)}},
    };

    static mzResult GetShaders(size_t* outCount, mzName* names, mzBuffer* outSpirvBufs)
    {
        *outCount = sizeof(spirvs) / sizeof(spirvs[0]);
        if (!outSpirvBufs)
            return MZ_RESULT_SUCCESS;

        for (auto& [name, spirv] : spirvs)
        {
            *names++ = mzEngine.GetName(name);
            *outSpirvBufs++ = { spirv.data(), spirv.size() };
        }

        return MZ_RESULT_SUCCESS;
    };

    static mzResult GetPasses(size_t* outCount, mzPassInfo* outMzPassInfos)
    {
        mzPassInfo passes[] =
        {
            {.Key = Cyclorama_CleanPlateAccumulator_Pass_Name, .Shader = Cyclorama_CleanPlateAccumulator_Shader_Name, .MultiSample = 1},
            {.Key = Cyclorama_Main_Pass_Name, .Shader = Cyclorama_Main_Frag_Name, .VertexShader = Cyclorama_Main_Vert_Name, .Blend = true, .MultiSample = 8, },
            {.Key = Cyclorama_Mask_Pass_Name, .Shader = Cyclorama_Mask_Frag_Name, .VertexShader = Cyclorama_Mask_Vert_Name, .MultiSample = 1},
        };

        *outCount = sizeof(passes) / sizeof(passes[0]);

        if (!outMzPassInfos)
            return MZ_RESULT_SUCCESS;

        memcpy(outMzPassInfos, passes, sizeof(passes));
        
        return MZ_RESULT_SUCCESS;
    }

    // Node graph event callbacks
    static mzResult CanCreateNode(const mzFbNode* node) { return MZ_RESULT_SUCCESS; }
    static void OnNodeCreated(const mzFbNode* node, void** ctx) 
    {
        Cyclorama* c = new Cyclorama();
        c->Load(*node);
        *ctx = c;
    }
    static void OnNodeUpdated(void* ctx, const mzFbNode* updatedNode) { }
    static void OnNodeDeleted(void* ctx, mzUUID nodeId) { delete (Cyclorama*)ctx; }
    static void OnPinValueChanged(void* ctx, mzUUID id, mzBuffer* value) 
    { 
        auto c = static_cast<Cyclorama*>(ctx);
        auto PinName = c->GetPinName(id);

#define CHECK_DIFF_AND_LOAD_VERTS(name) \
        if(#name == PinName) { \
            auto& val = *(std::remove_reference_t<decltype(c->name)>*)(value->Data);\
            if(NotSame(c->name, val)) { \
                c->name = val; \
                c->LoadVertices(); \
            } \
            return; \
        }

#define CHECK_AND_SET(name) \
        if(#name == PinName) { \
            c->name = *(std::remove_reference_t<decltype(c->name)>*)(value->Data);\
            return;\
        }

#define CHECK_AND_SET_TABLE(name, type) \
        if(#name == PinName) { \
            flatbuffers::GetRoot<type>(value->Data)->UnPackTo(&c->name);\
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

        if (CaptureFolder_Name == PinName)
        {
            c->CaptureFolder = (char*)value->Data;
            return;
        }

        if (CleanPlates_Name == PinName)
        {
            // c->LoadCleanPlates(*value->As<mz::fb::CaptureDataArray>());
            return;
        }

        if (!(c->Status & Cyclorama::CAPTURING))
        {
            return;
        }

        if (Video_Name == c->GetPinName(id))
        {
            auto video = DeserializeTextureInfo(value->Data);
            
            if (c->CapturedFrameCount++ < 50)
            {
                mzShaderBinding bindings[] =
                {
                    ShaderBinding(lhs_Name, c->lhs),
                    ShaderBinding(rhs_Name, video),
                };

                mzRunPassParams pass = {};
                pass.Key = Cyclorama_CleanPlateAccumulator_Pass_Name;
                pass.Output = c->rt;
                pass.BindingCount = sizeof(bindings) / sizeof(bindings[0]);
                pass.Bindings = bindings;
                mzEngine.RunPass(0, &pass);
                std::swap(c->rt, c->lhs);
            }
            else
            {
                std::unique_lock lock(c->CPLock);
                const u32 idx = c->CleanPlates.size();
                std::filesystem::path path = c->CaptureFolder;
                if (!std::filesystem::is_directory(path))
                    path = std::filesystem::current_path();
                path /= (std::to_string(idx) + UUID2STR(c->NodeId) + ".png");

                mzResourceShareInfo tmp = c->lhs;
                mzResourceShareInfo tex = c->lhs;
                tex.Info.Texture.Format = MZ_FORMAT_R8G8B8A8_UNORM;
                tex.Info.Texture.Width = 960;
                tex.Info.Texture.Height = 540;
                mzEngine.Create(&tex);
                mzEngine.Create(&c->lhs);
                mzEngine.Clear(0, &c->lhs, {});
                c->CleanPlates.push_back({  .Path = path.string(),
                                            .Texture = tex,
                                            .Track = c->Track,
                                            .Pos = c->Position,
                                            .Rot = c->Rotation});

                std::thread([c, tmp, tex, path = std::move(path)]
                {
                    c->Status |= Cyclorama::SAVING;
                    c->UpdateStatus();
                    mzResourceShareInfo srgb = tmp;
                    srgb.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
                    srgb.Info.Texture.Format = MZ_FORMAT_R8G8B8A8_SRGB;
                    srgb.Info.Texture.Width = 960;
                    srgb.Info.Texture.Height = 540;
                    srgb.Info.Texture.Usage = mzImageUsage(MZ_IMAGE_USAGE_TRANSFER_DST | MZ_IMAGE_USAGE_TRANSFER_SRC);
                    mzEngine.Create(&srgb);
                    mzCmd cmd;
                    mzEngine.Begin(&cmd);
                    mzEngine.Blit(cmd, &tmp, &tex);
                    mzEngine.Blit(cmd, &tmp, &srgb);
                    mzResourceShareInfo buf;
                    mzEngine.Download(cmd, &srgb, &buf);
                    mzEngine.End(cmd);

                    auto dst = new u8[srgb.Info.Texture.Width * srgb.Info.Texture.Height * 3];
                    auto src = mzEngine.Map(&buf);
                    for(u32 i = 0; i < srgb.Info.Texture.Width; ++i)
                        for(u32 j = 0; j < srgb.Info.Texture.Height; ++j)
                            memcpy(dst + 3 * (j + i * srgb.Info.Texture.Height), src + 4 * (j + i * srgb.Info.Texture.Height), 3);
                    auto re = stbi_write_png(path.string().c_str(), srgb.Info.Texture.Width, srgb.Info.Texture.Height, 3, dst, srgb.Info.Texture.Width * 3);
                    delete dst;
                    mzEngine.Destroy(&buf);
                    mzEngine.Destroy(&tmp);
                    mzEngine.Destroy(&srgb);
                    c->Status &= ~Cyclorama::SAVING;
                    c->UpdateStatus();
                }).detach();
                c->Status &= ~Cyclorama::CAPTURING;
                c->CapturedFrameCount = 0;
                c->UpdateStatus();
                c->UpdateCleanPlatesValue();
				flatbuffers::FlatBufferBuilder fbb;
                mzEngine.HandleEvent(CreateAppEvent(fbb, mz::app::CreateScheduleRequest(fbb, mz::app::ScheduleRequestKind::PIN, &id, true)));
            }
        }
    }
    static void OnPinConnected(void* ctx, mzUUID pinId) { }
    static void OnPinDisconnected(void* ctx, mzUUID pinId) { }
    static void OnPinShowAsChanged(void* ctx, mzUUID pinId, mzFbShowAs showAs) { }
    static void OnNodeSelected(mzUUID graphId, mzUUID selectedNodeId) { }
    static void OnPathCommand(void* ctx, const mzPathCommand* command) { }

    
    // Execution
    static mzResult ExecuteNode(void* ctx, const mzNodeExecuteArgs* inArgs) \
    { 
        auto c = (Cyclorama*)ctx;
        
        auto args = GetPinValues(inArgs);

        mzResourceShareInfo color;
        mzRunPass2Params pass;
        std::vector<mzDrawCall> calls(1 + c->CleanPlates.size());
        std::vector<std::vector<mzShaderBinding>> inputs(calls.size());

        pass.Wireframe = *GetPinValue<bool>(args, Wireframe_Name);
        pass.Key = Cyclorama_Main_Pass_Name;
        pass.Output = DeserializeTextureInfo(GetPinValue<void>(args, Render_Name));;
        auto track = GetPinValue<fb::TTrack>(args, Track_Name);
      
        glm::dvec3 pos = (glm::dvec3&)track.location;
        glm::dvec3 rot = (glm::dvec3&)track.rotation;
        auto view = MakeTransform(pos, rot);
        auto prj = Perspective((f32)track.fov, (f32)track.pixel_aspect_ratio, (glm::dvec2&)track.sensor_size, (glm::dvec2&)track.center_shift);
        // glm::vec3 msize = *pins.Get<glm::dvec3>("Size");
        glm::vec3 mpos = *GetPinValue<glm::dvec3>(args, Position_Name);
        glm::vec3 mrot = glm::radians(*GetPinValue<glm::dvec3>(args, Rotation_Name));

        // c->Size = msize;
        c->Position = mpos;
        c->Rotation = mrot;
        c->Track = std::move(track);

        glm::mat4 model = glm::eulerAngleZYX(mrot.z, mrot.y, mrot.x) * glm::mat4(100.f);
        model[3] = glm::vec4(mpos, 1.f);

        glm::mat4 MVP = prj * view * model;

        auto Offset = c->GetOrigin(*GetPinValue<u32>(args, OriginPreset_Name));
        inputs[0].push_back(ShaderBinding(VOffset_Name, Offset));
        inputs[0].push_back(ShaderBinding(MVP_Name, MVP));

        AddParam<f32>(inputs[0], args, SmoothnessCurve_Name);
        f32 zero = 0.f;
        f32 half = 0.5f;

        {
            mzEngine.GetColorTexture(*GetPinValue<mzVec4>(args, CycloramaColor_Name), &color);
            inputs[0].push_back(ShaderBinding(UVSmoothness_Name, zero));
            inputs[0].push_back(ShaderBinding(CP_MVP_Name, MVP));
            inputs[0].push_back(ShaderBinding(Source_Name, color));
        }

        std::vector<glm::mat4> matrices(calls.size());

        auto* mat = &matrices[0];
        auto* input = &inputs[0];
        auto* call = &calls[0];

        for (auto& last : c->CleanPlates)
        {
            ++input, ++call, ++mat;
            call->Vertices.DepthWrite = 0;
            call->Vertices.DepthTest = 0;
            call->Vertices.DepthFunc = MZ_DEPTH_FUNCTION_NEVER;

            input->push_back(ShaderBinding(UVSmoothness_Name, half));
            input->push_back(ShaderBinding(CP_MVP_Name, *mat));
            input->push_back(ShaderBinding(Source_Name, last.Texture));

            glm::vec3 pos = (glm::dvec3&)last.Track.location;
            glm::vec3 rot = (glm::dvec3&)last.Track.rotation;
            glm::mat4 model = glm::eulerAngleZYX(last.Rot.z, last.Rot.y, last.Rot.x) * glm::mat4(100.f);
            model[3] = glm::vec4(last.Pos, 1.f);
            *mat = Perspective((f32)last.Track.fov, (f32)last.Track.pixel_aspect_ratio, (glm::dvec2&)last.Track.sensor_size, (glm::dvec2&)last.Track.center_shift) * MakeTransform(pos, rot) * model;
        }

        pass.DrawCalls = calls.data();
        pass.DrawCallCount = calls.size();

        for (u32 i = 0; i < calls.size(); ++i)
        {
            calls[i].Vertices = c->Verts;
            calls[i].Bindings = inputs[i].data();
            calls[i].BindingCount = inputs[i].size();
        }

        mzEngine.RunPass2(0, &pass);

        {
            c->Validate(args);

            glm::vec4 smoothness = glm::vec4(
                *GetPinValue<f32>(args, BottomSmoothness_Name),
                *GetPinValue<f32>(args, LeftSmoothness_Name),
                *GetPinValue<f32>(args, TopSmoothness_Name),
                *GetPinValue<f32>(args, RightSmoothness_Name)
            ) / 100.f;
            glm::vec4 crop = glm::vec4(
                *GetPinValue<f32>(args, BottomCrop_Name),
                *GetPinValue<f32>(args, LeftCrop_Name),
                *GetPinValue<f32>(args, TopCrop_Name),
                *GetPinValue<f32>(args, RightCrop_Name)
            ) / 100.f;
            glm::vec2 diag = glm::vec2(
                *GetPinValue<f32>(args, DiagonalCrop_Name),
                *GetPinValue<f32>(args, DiagonalSmoothness_Name)
            ) / 100.f;

            f32 roundness = c->EdgeRoundness / 100.f;
            auto scale = glm::vec4(c->Width, c->LeftWingLength, c->Height, c->RightWingLength) / 100.f;
            
            glm::vec2 angle = glm::radians(glm::vec2(c->LeftWingAngle, c->RightWingAngle));

            u32 flags = MakeFlags(args,
                SharpEdges_Name, SHARP_EDGES_BIT,
                HasLeftWing_Name, HAS_LEFT_WING_BIT,
                HasRightWing_Name, HAS_RIGHT_WING_BIT);

            std::vector<mzShaderBinding> maskInputs;
            mzRunPassParams maskPass;
            maskPass.Key = Cyclorama_Mask_Pass_Name;
            maskPass.Output = DeserializeTextureInfo(GetPinValue<void>(args, Mask_Name));;
            maskPass.Vertices = c->Verts;
            maskInputs.push_back(ShaderBinding(MVP_Name, MVP));

       
            AddParam<f32>(maskInputs, args, SmoothnessCurve_Name);
            AddParam<glm::vec4>(maskInputs, args, MaskColor_Name);
            maskInputs.push_back(ShaderBinding(VOffset_Name, Offset));
            maskInputs.push_back(ShaderBinding(Flags_Name, flags));
            maskInputs.push_back(ShaderBinding(Smoothness_Name, smoothness));
            maskInputs.push_back(ShaderBinding(SmoothnessCrop_Name, crop));
            maskInputs.push_back(ShaderBinding(Scale_Name, scale));
            maskInputs.push_back(ShaderBinding(Roundness_Name, roundness));
            maskInputs.push_back(ShaderBinding(Angle_Name, angle));
            maskInputs.push_back(ShaderBinding(Diag_Name, diag));

            maskPass.Bindings = maskInputs.data();
            maskPass.BindingCount = maskInputs.size();
            mzEngine.RunPass(0, &maskPass);
        }

        return MZ_RESULT_SUCCESS;
    }

    static mzResult CanCopy(void* ctx, mzCopyInfo* copyInfo) { return MZ_RESULT_SUCCESS; }
    static mzResult BeginCopyFrom(void* ctx, mzCopyInfo* cospyInfo) { return MZ_RESULT_SUCCESS; }
    static mzResult BeginCopyTo(void* ctx, mzCopyInfo* copyInfo) { return MZ_RESULT_SUCCESS; }
    static void EndCopyFrom(void* ctx, mzCopyInfo* copyInfo) {  }
    static void EndCopyTo(void* ctx, mzCopyInfo* copyInfo) { }
    // Menu & key events
    static void OnMenuRequested(void* ctx, const mzContextMenuRequest* request) { }
    static void OnMenuCommand(void* ctx, uint32_t cmd) { }
    static void OnKeyEvent(void* ctx, const mzKeyEvent* keyEvent) { }


    static void AddProjection(void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs)
    {
        auto c = (Cyclorama*)ctx;
        if (c->Status & Cyclorama::CAPTURING) return;
        c->Status |= Cyclorama::CAPTURING;
        c->CapturedFrameCount = 0;
        c->UpdateStatus();
        flatbuffers::FlatBufferBuilder fbb;
        auto id = c->GetPinId(Video_Name);
        mzEngine.HandleEvent(CreateAppEvent(fbb, mz::app::CreateScheduleRequest(fbb, mz::app::ScheduleRequestKind::PIN, &id)));
    };

    static void ClearProjection(void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs) {
        auto c = (Cyclorama*)ctx;
        if (c->Status & Cyclorama::CAPTURING) return;
        c->Status &= ~Cyclorama::CAPTURING;
        c->CapturedFrameCount = 0;
        c->UpdateStatus();
        c->CleanCleanPlates();
        c->UpdateCleanPlatesValue();
        c->Clear();
    }

    static void ReloadShaders(void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs) {
        auto c = (Cyclorama*)ctx;
        system(("glslc " + std::string(mzEngine.WorkFolder()) +  "/../Plugins/mzCyclorama/Source/Cyclorama.frag -c -o " + std::string(mzEngine.WorkFolder()) +  "/cyclo.frag").c_str());
        system(("glslc " + std::string(mzEngine.WorkFolder()) +  "/../Plugins/mzCyclorama/Source/Cyclorama.vert -c -o " + std::string(mzEngine.WorkFolder()) +  "/cyclo.vert").c_str());
        system(("glslc " + std::string(mzEngine.WorkFolder()) +  "/../Plugins/mzCyclorama/Source/CycloramaMask.frag -c -o " + std::string(mzEngine.WorkFolder()) +  "/cyclo_mask.frag").c_str());
        system(("glslc " + std::string(mzEngine.WorkFolder()) +  "/../Plugins/mzCyclorama/Source/CycloramaMask.vert -c -o " + std::string(mzEngine.WorkFolder()) +  "/cyclo_mask.vert").c_str());
        system(("glslc " + std::string(mzEngine.WorkFolder()) +  "/../Plugins/mzCyclorama/Source/CleanPlateAcc.frag -c -o " + std::string(mzEngine.WorkFolder()) +  "/cp.frag").c_str());
        spirvs[0].second = ReadSpirv(std::string(mzEngine.WorkFolder()) + "/cyclo.frag");
        spirvs[1].second = ReadSpirv(std::string(mzEngine.WorkFolder()) + "/cyclo.vert");
        spirvs[2].second = ReadSpirv(std::string(mzEngine.WorkFolder()) + "/cyclo_mask.frag");
        spirvs[3].second = ReadSpirv(std::string(mzEngine.WorkFolder()) + "/cyclo_mask.vert");
        spirvs[4].second = ReadSpirv(std::string(mzEngine.WorkFolder()) + "/cp.frag");

        mzEngine.ReloadShaders("Cyclorama.Cyclorama");

    }

    inline static std::pair<const char*, mzPfnNodeFunctionExecute> Functions[] = 
    {
        {"AddProjection", AddProjection},
        {"ClearProjection", ClearProjection},
        {"ReloadShaders", ReloadShaders}
    };

    // Function Nodes
    static mzResult GetFunctions(size_t* outCount, const char** pName, mzPfnNodeFunctionExecute* outFunction)
    {
        *outCount = sizeof(Functions)/sizeof(Functions[0]);

        if(!pName || !outFunction) 
            return MZ_RESULT_SUCCESS;

        for (auto& [name, fn] : Functions)
        {
            *pName++ = name;
            *outFunction++ = fn;
        }

        return MZ_RESULT_SUCCESS;
    }
};

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



extern "C"
{

    MZAPI_ATTR mzResult MZAPI_CALL mzExportNodeFunctions(size_t* outSize, mzNodeFunctions* outFunctions)
    {
        *outSize = 1;
        if (!outFunctions)
            return MZ_RESULT_SUCCESS;

        using namespace mz;
        *outFunctions = {
            .TypeName = "Cyclorama.Cyclorama",
            .CanCreateNode = Cyclorama::CanCreateNode,
            .OnNodeCreated = Cyclorama::OnNodeCreated,
            .OnNodeUpdated = Cyclorama::OnNodeUpdated,
            .OnNodeDeleted = Cyclorama::OnNodeDeleted,
            .OnPinValueChanged = Cyclorama::OnPinValueChanged,
            .OnPinConnected = Cyclorama::OnPinConnected,
            .OnPinDisconnected = Cyclorama::OnPinDisconnected,
            .OnPinShowAsChanged = Cyclorama::OnPinShowAsChanged,
            .OnNodeSelected = Cyclorama::OnNodeSelected,
            .OnPathCommand = Cyclorama::OnPathCommand,
            .GetFunctions = Cyclorama::GetFunctions,
            .ExecuteNode = Cyclorama::ExecuteNode,
            .CanCopy = Cyclorama::CanCopy,
            .BeginCopyFrom = Cyclorama::BeginCopyFrom,
            .BeginCopyTo = Cyclorama::BeginCopyTo,
            .EndCopyFrom = Cyclorama::EndCopyFrom,
            .EndCopyTo = Cyclorama::EndCopyTo,
            .GetShaders = Cyclorama::GetShaders,
            .GetPasses = Cyclorama::GetPasses,
            .OnMenuRequested = Cyclorama::OnMenuRequested,
            .OnMenuCommand = Cyclorama::OnMenuCommand,
            .OnKeyEvent = Cyclorama::OnKeyEvent,
        };

        return MZ_RESULT_SUCCESS;
    }

}
