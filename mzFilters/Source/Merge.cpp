// Copyright MediaZ AS. All Rights Reserved.


#include "AppService_generated.h"
#include "BasicMain.h"

#include <Args.h>
#include "Merge.frag.spv.dat"
#include <uuid.h>

namespace mz
{

std::seed_seq Seed()
{
    std::random_device rd;
    auto seed_data = std::array<int, std::mt19937::state_size>{};
    std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
    return std::seed_seq(std::begin(seed_data), std::end(seed_data));
}

std::seed_seq seed = Seed();
std::mt19937 mtengine(seed);
uuids::uuid_random_generator generator(mtengine);

struct MergeContext
{
    mz::fb::TTexture DummyTex;

    std::atomic_uint TextureCount = 2;

    ~MergeContext() 
    {
        GServices.Destroy(DummyTex);
    }

    void Run(mz::Args& pins)
    {
        app::TRunPass mergePass;
        mergePass.pass = "Merge_Pass";

        for (auto& [name, pin] : pins.PinData)
        {
            if (name.compare("Out") == 0) continue;
            CopyUniformFromPin(mergePass, pins, name);
        }

        for (auto& [name, _] : pins.PinData)
            if (name.starts_with("Texture_"))
                TextureCount = std::max(TextureCount.load(), (u32)std::atoi(name.c_str() + sizeof("Texture_")));
        
        // If there is no pin for uniform, use dummy texture
        for (u32 i = TextureCount; i < 16; i++)
        {
            u32 idx = AddUniform(mergePass, "Texture_" + std::to_string(i));
            std::vector<u8> data = mz::Buffer::From(DummyTex);
            mergePass.inputs[idx]->val = std::move(data);
        }

        u32 tex = TextureCount;
        AddUniform(mergePass, "Texture_Count", &tex, sizeof(u32));
        mergePass.output.reset(pins.Get<mz::fb::Texture>("Out")->UnPack());
        GServices.MakeAPICall(mergePass, true);
    }

    fb::UUID id;
    void OnMenuFired(mz::fb::Node const& node, mz::TContextMenuRequest const& request)
    {
        id = *node.id();
        flatbuffers::FlatBufferBuilder fbb;

        std::vector<flatbuffers::Offset<mz::ContextMenuItem>> items = {
           mz::CreateContextMenuItemDirect(fbb, "Add Input Texture", 1)
        };

        GServices.HandleEvent(CreateAppEvent(fbb, mz::CreateContextMenuUpdateDirect(fbb, node.id(), &request.pos, &request.instigator, &items)));
    }

    void OnCommandFired(u32 cmd)
    {
        if (!cmd)
        {
            return;
        }
        TextureCount++;

        auto defaultTexBuf = GServices.GetDefaultDataOfType("mz.fb.Texture");
        auto tex = defaultTexBuf->As<mz::fb::TTexture>();

        std::string         texturePinName = "Texture_" + std::to_string(TextureCount.load());
        mz::fb::UUID        textureId = *(mz::fb::UUID*)generator().as_bytes().data();
        std::vector<u8>     textureData = mz::Buffer::From(tex);
        
        std::string         opacityPinName = "Opacity_" + std::to_string(TextureCount.load());
        mz::fb::UUID        opacityId = *(mz::fb::UUID*)generator().as_bytes().data();
        std::vector<u8>     opacityData(sizeof(float), 1.0f);

        std::string         blendModePinName = "Blend_Mode_" + std::to_string(TextureCount.load());
        mz::fb::UUID        blendModeId = *(mz::fb::UUID*)generator().as_bytes().data();
        std::vector<u8>     blendModeData(sizeof(unsigned int), 0);

        std::string pinCategory = "Layer (" + std::to_string(TextureCount.load()) + ")";
        
        flatbuffers::FlatBufferBuilder fbb;
        std::vector<flatbuffers::Offset<mz::fb::Pin>> pins = {
            mz::fb::CreatePinDirect(fbb, &textureId, texturePinName.c_str(), "mz.fb.Texture", mz::fb::ShowAs::INPUT_PIN, mz::fb::CanShowAs::INPUT_PIN_ONLY, pinCategory.c_str(), 0, &textureData),
            mz::fb::CreatePinDirect(fbb, &opacityId, opacityPinName.c_str(), "float", mz::fb::ShowAs::PROPERTY, mz::fb::CanShowAs::INPUT_PIN_OR_PROPERTY, pinCategory.c_str(), 0 , &opacityData),
            mz::fb::CreatePinDirect(fbb, &blendModeId, blendModePinName.c_str(), "mz.fb.BlendMode", mz::fb::ShowAs::PROPERTY, mz::fb::CanShowAs::INPUT_PIN_OR_PROPERTY, pinCategory.c_str(), 0 , &blendModeData),
        };

        GServices.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &id, ClearFlags::NONE, 0, &pins)));
    }
};

void RegisterMerge(NodeActionsMap& functions)
{
    auto& actions = functions["mz.Merge"];
    actions.NodeCreated = [](fb::Node const& node, mz::Args& args, void** ctx) {
        *ctx = new MergeContext();
        auto* mergeCtx = (MergeContext*)*ctx;

        // Create dummy texture
        mergeCtx->DummyTex.width = 1;
        mergeCtx->DummyTex.height = 1;
        mergeCtx->DummyTex.format = mz::fb::Format::R8_UNORM;
        GServices.Create(mergeCtx->DummyTex);
    };

    actions.Shaders = []() { return ShaderLibrary{ { "Merge_Shader", ShaderSrc<sizeof(Merge_frag_spv)>(Merge_frag_spv) }}; };
    actions.Passes = []() { return PassLibrary{ app::TRegisterPass { .key =  "Merge_Pass", .shader = "Merge_Shader" } }; };

    actions.NodeRemoved = [](void* ctx, mz::fb::UUID const& id) {
        delete (MergeContext*)ctx;
    };

    actions.EntryPoint = [](mz::Args& pins, void* ctx) {
        auto* mergeCtx = (MergeContext*)ctx;
        mergeCtx->Run(pins);
        return true;
    };
    
    actions.MenuFired = [](auto& node, auto ctx, auto& request) { 
        auto* mergeCtx = (MergeContext*)ctx;
        if (mergeCtx->TextureCount >= 16)
            return false;
        mergeCtx->OnMenuFired(node, request);
        return true;
    };
    
    actions.CommandFired = [](auto& node, auto ctx, u32 cmd) { 
        auto* mergeCtx = (MergeContext*)ctx;
        mergeCtx->OnCommandFired(cmd);
        return true;
    };
    
}

} // namespace mz
