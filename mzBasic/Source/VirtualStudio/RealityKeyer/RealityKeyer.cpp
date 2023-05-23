// Copyright MediaZ AS. All Rights Reserved.

#include "AppService_generated.h"
#include "BasicMain.h"

#include <Args.h>

#include "IBKPass1.frag.spv.dat"
#include "IBKPass2.frag.spv.dat"
#include "IBKHorizontalBlur.frag.spv.dat"

namespace mz
{

struct RealityKeyerContext : public NodeContext
{

    ~RealityKeyerContext()
    {
        DestroyResources();
    }

    void RegisterShaders()
    {
        static bool shadersRegistered = false;
        if (shadersRegistered)
        {
            return;
        }
        mzEngine.MakeAPICalls(true,
                              app::TRegisterShader{
                                  .key = "IBK_Pass_1_Shader",
                                  .spirv = ShaderSrc<sizeof(IBKPass1_frag_spv)>(IBKPass1_frag_spv)},
                              app::TRegisterShader{
                                  .key = "IBK_Pass_2_Shader",
                                  .spirv = ShaderSrc<sizeof(IBKPass2_frag_spv)>(IBKPass2_frag_spv)},
                              app::TRegisterShader{
                                  .key = "IBK_Horz_Blur_Pass_Shader",
                                  .spirv = ShaderSrc<sizeof(IBKHorizontalBlur_frag_spv)>(IBKHorizontalBlur_frag_spv)});

        shadersRegistered = true;
    }

    void RegisterPasses()
    {
        mzEngine.MakeAPICalls(true,
                              app::TRegisterPass{
                                  .key = "IBK_Pass_1_" + UUID2STR(NodeId),
                                  .shader = "IBK_Pass_1_Shader",
                              },
                              app::TRegisterPass{
                                  .key = "IBK_Pass_2_" + UUID2STR(NodeId),
                                  .shader = "IBK_Pass_2_Shader"},
                              app::TRegisterPass{
                                  .key = "IBK_Horz_Blur_Pass_" + UUID2STR(NodeId),
                                  .shader = "IBK_Horz_Blur_Pass_Shader"});
    }

    void Setup()
    {
        RegisterShaders();
        RegisterPasses();
    }

    void DestroyResources()
    {
        mzEngine.MakeAPICalls(true,
                              app::TUnregisterPass{
                                  .key = "IBK_Pass_1_" + UUID2STR(NodeId),
                              },
                              app::TUnregisterPass{
                                  .key = "IBK_Pass_2_" + UUID2STR(NodeId),
                              },
                              app::TUnregisterPass{
                                  .key = "IBK_Horz_Blur_Pass_" + UUID2STR(NodeId),
                              });
    }

    void Run(mz::Args& pins)
    {
        auto* outputPinData = pins.Get<mz::fb::Texture>("Output");
        auto* hardMaskPinData = pins.Get<mz::fb::Texture>("Hard_Mask");
        auto* hardMaskHorzBlurPinData = pins.Get<mz::fb::Texture>("Hard_Mask_Horz_Blur");

        // Pass 1 begin
        app::TRunPass ibkPass1;
        ibkPass1.pass = "IBK_Pass_1_" + UUID2STR(NodeId);
        CopyUniformFromPin(ibkPass1, pins, "Input");
        CopyUniformFromPin(ibkPass1, pins, "Clean_Plate");
        CopyUniformFromPin(ibkPass1, pins, "Key_High_Brightness");
        CopyUniformFromPin(ibkPass1, pins, "Core_Matte_Clean_Plate_Gain");
        CopyUniformFromPin(ibkPass1, pins, "Core_Matte_Gamma_1");
        CopyUniformFromPin(ibkPass1, pins, "Core_Matte_Gamma_2");
        CopyUniformFromPin(ibkPass1, pins, "Core_Matte_Red_Weight");
        CopyUniformFromPin(ibkPass1, pins, "Core_Matte_Blue_Weight");
        CopyUniformFromPin(ibkPass1, pins, "Core_Matte_Black_Point");
        CopyUniformFromPin(ibkPass1, pins, "Core_Matte_White_Point");
        ibkPass1.output = std::make_unique<mz::fb::TTexture>();
        hardMaskPinData->UnPackTo(ibkPass1.output.get());
        mzEngine.MakeAPICall(ibkPass1, true);
        // Pass 1 end
        // Horz blur begin
        app::TRunPass ibkHorzBlurPass;
        ibkHorzBlurPass.pass = "IBK_Horz_Blur_Pass_" + UUID2STR(NodeId);
        AddUniform(ibkHorzBlurPass, "Input");
        Chain(ibkPass1, ibkHorzBlurPass, "Input");
        float blurRadius = *pins.Get<float>("Erode") + *pins.Get<float>("Softness");
        AddUniform(ibkHorzBlurPass, "Blur_Radius", &blurRadius, sizeof(float));
        mz::fb::vec2 blurInputSize(hardMaskPinData->width(), hardMaskPinData->height());
        AddUniform(ibkHorzBlurPass, "Input_Texture_Size", &blurInputSize, sizeof(mz::fb::vec2));
        ibkHorzBlurPass.output = std::make_unique<mz::fb::TTexture>();
		hardMaskHorzBlurPinData->UnPackTo(ibkHorzBlurPass.output.get());
        mzEngine.MakeAPICall(ibkHorzBlurPass, true);
        // Horz blur end
        // Pass 2 begin
        app::TRunPass ibkPass2;
        ibkPass2.pass = "IBK_Pass_2_" + UUID2STR(NodeId);
        CopyUniformFromPin(ibkPass2, pins, "Input");
        CopyUniformFromPin(ibkPass2, pins, "Clean_Plate");
        CopyUniformFromPin(ibkPass2, pins, "Clean_Plate_Mask");
        AddUniform(ibkPass2, "Core_Matte");
        AddUniform(ibkPass2, "Unblurred_Core_Matte");
        Chain(ibkHorzBlurPass, ibkPass2, "Core_Matte");
        Chain(ibkPass1, ibkPass2, "Unblurred_Core_Matte");
        mz::fb::vec2 coreMatteTextureSize(ibkPass1.output->width, ibkPass1.output->height);
        AddUniform(ibkPass2, "Core_Matte_Texture_Size", &coreMatteTextureSize, sizeof(coreMatteTextureSize));
        CopyUniformFromPin(ibkPass2, pins, "Erode");
        CopyUniformFromPin(ibkPass2, pins, "Softness");
        CopyUniformFromPin(ibkPass2, pins, "Soft_Matte_Red_Weight");
        CopyUniformFromPin(ibkPass2, pins, "Soft_Matte_Blue_Weight");
        CopyUniformFromPin(ibkPass2, pins, "Soft_Matte_Gamma_1");
        CopyUniformFromPin(ibkPass2, pins, "Soft_Matte_Gamma_2");
        CopyUniformFromPin(ibkPass2, pins, "Soft_Matte_Clean_Plate_Gain");
        CopyUniformFromPin(ibkPass2, pins, "Soft_Matte_422_Filtering", [](mz::Buffer* buf) {
            float val = *(float*)buf->data();
            mz::fb::vec2 vec;
            vec.mutate_x(1.0 - val);
            vec.mutate_y(val * .5);
            std::vector<u8> data(sizeof(vec));
            memcpy(data.data(), &vec, sizeof(vec));
            return data;
        });
        CopyUniformFromPin(ibkPass2, pins, "Key_High_Brightness");
        CopyUniformFromPin(ibkPass2, pins, "Core_Matte_Blend");
        CopyUniformFromPin(ibkPass2, pins, "Edge_Spill_Replace_Color", [](mz::Buffer* buf) {
            mz::fb::vec3 col = *(mz::fb::vec3*)buf->data();
            col.mutate_x(pow(2.0, col.x()));
            col.mutate_y(pow(2.0, col.y()));
            col.mutate_z(pow(2.0, col.z()));
            std::vector<u8> data(sizeof(mz::fb::vec3));
            memcpy(data.data(), &col, sizeof(mz::fb::vec3));
            return data;
        });
        CopyUniformFromPin(ibkPass2, pins, "Core_Spill_Replace_Color", [](mz::Buffer* buf) {
            mz::fb::vec3 col = *(mz::fb::vec3*)buf->data();
            col.mutate_x(pow(2.0, col.x()));
            col.mutate_y(pow(2.0, col.y()));
            col.mutate_z(pow(2.0, col.z()));
            std::vector<u8> data(sizeof(mz::fb::vec3));
            memcpy(data.data(), &col, sizeof(mz::fb::vec3));
            return data;
        });
        CopyUniformFromPin(ibkPass2, pins, "Spill_Matte_Gamma");
        CopyUniformFromPin(ibkPass2, pins, "Spill_Matte_Red_Weight");
        CopyUniformFromPin(ibkPass2, pins, "Spill_Matte_Blue_Weight");
        CopyUniformFromPin(ibkPass2, pins, "Spill_Matte_Gain");
        CopyUniformFromPin(ibkPass2, pins, "Spill_RB_Weight");
        CopyUniformFromPin(ibkPass2, pins, "Spill_Suppress_Weight");
        CopyUniformFromPin(ibkPass2, pins, "Spill_422_Filtering", [](mz::Buffer* buf) {
            float val = *(float*)buf->data();
            mz::fb::vec2 vec;
            vec.mutate_x(1.0 - val);
            vec.mutate_y(val * .5);
            std::vector<u8> data(sizeof(vec));
            memcpy(data.data(), &vec, sizeof(vec));
            return data;
        });
        CopyUniformFromPin(ibkPass2, pins, "Screen_Subtract_Edge");
        CopyUniformFromPin(ibkPass2, pins, "Screen_Subtract_Core");
        CopyUniformFromPin(ibkPass2, pins, "Keep_Edge_Luma");
        CopyUniformFromPin(ibkPass2, pins, "Keep_Core_Luma");
        CopyUniformFromPin(ibkPass2, pins, "Final_Matte_Black_Point");
        CopyUniformFromPin(ibkPass2, pins, "Final_Matte_White_Point");
        CopyUniformFromPin(ibkPass2, pins, "Final_Matte_Gamma");
        CopyUniformFromPin(ibkPass2, pins, "Gamma", [&pins](mz::Buffer* buf) {
            mz::fb::vec3 val = *(mz::fb::vec3*)buf->data();
            val.mutate_x(val.x() * *pins.Get<float>("Master_Gamma"));
            val.mutate_y(val.y() * *pins.Get<float>("Master_Gamma"));
            val.mutate_z(val.z() * *pins.Get<float>("Master_Gamma"));
            std::vector<u8> data(sizeof(mz::fb::vec3));
            memcpy(data.data(), &val, sizeof(mz::fb::vec3));
            return data;
        });
        CopyUniformFromPin(ibkPass2, pins, "Exposure", [&pins](mz::Buffer* buf) {
            mz::fb::vec3 val = *(mz::fb::vec3*)buf->data();
            val.mutate_x(val.x() + *pins.Get<float>("Master_Exposure"));
            val.mutate_y(val.y() + *pins.Get<float>("Master_Exposure"));
            val.mutate_z(val.z() + *pins.Get<float>("Master_Exposure"));
            val.mutate_x(pow(2.0, val.x()));
            val.mutate_y(pow(2.0, val.y()));
            val.mutate_z(pow(2.0, val.z()));
            std::vector<u8> data(sizeof(mz::fb::vec3));
            memcpy(data.data(), &val, sizeof(mz::fb::vec3));
            return data;
        });
        CopyUniformFromPin(ibkPass2, pins, "Offset", [&pins](mz::Buffer* buf) {
            mz::fb::vec3 val = *(mz::fb::vec3*)buf->data();
            val.mutate_x(val.x() + *pins.Get<float>("Master_Offset"));
            val.mutate_y(val.y() + *pins.Get<float>("Master_Offset"));
            val.mutate_z(val.z() + *pins.Get<float>("Master_Offset"));
            std::vector<u8> data(sizeof(mz::fb::vec3));
            memcpy(data.data(), &val, sizeof(mz::fb::vec3));
            return data;
        });
        CopyUniformFromPin(ibkPass2, pins, "Saturation", [&pins](mz::Buffer* buf) {
            mz::fb::vec3 val = *(mz::fb::vec3*)buf->data();
            val.mutate_x(val.x() * *pins.Get<float>("Master_Saturation"));
            val.mutate_y(val.y() * *pins.Get<float>("Master_Saturation"));
            val.mutate_z(val.z() * *pins.Get<float>("Master_Saturation"));
            std::vector<u8> data(sizeof(mz::fb::vec3));
            memcpy(data.data(), &val, sizeof(mz::fb::vec3));
            return data;
        });
        CopyUniformFromPin(ibkPass2, pins, "Contrast", [&pins](mz::Buffer* buf) {
            mz::fb::vec3 val = *(mz::fb::vec3*)buf->data();
            val.mutate_x(val.x() * *pins.Get<float>("Master_Contrast"));
            val.mutate_y(val.y() * *pins.Get<float>("Master_Contrast"));
            val.mutate_z(val.z() * *pins.Get<float>("Master_Contrast"));
            std::vector<u8> data(sizeof(mz::fb::vec3));
            memcpy(data.data(), &val, sizeof(mz::fb::vec3));
            return data;
        });
        CopyUniformFromPin(ibkPass2, pins, "Contrast_Center", [&pins](mz::Buffer* buf) {
            mz::fb::vec3 val = *(mz::fb::vec3*)buf->data();
            val.mutate_x(val.x() + *pins.Get<float>("Master_Contrast_Center"));
            val.mutate_y(val.y() + *pins.Get<float>("Master_Contrast_Center"));
            val.mutate_z(val.z() + *pins.Get<float>("Master_Contrast_Center"));
            std::vector<u8> data(sizeof(mz::fb::vec3));
            memcpy(data.data(), &val, sizeof(mz::fb::vec3));
            return data;
        });
        CopyUniformFromPin(ibkPass2, pins, "Output_Type");
        ibkPass2.output = std::make_unique<mz::fb::TTexture>();
        outputPinData->UnPackTo(ibkPass2.output.get());
        mzEngine.MakeAPICall(ibkPass2, true);
        // Pass 2 end
    }
};

void RegisterRealityKeyer(NodeActionsMap& functions)
{
    auto& actions = functions["mz.RealityKeyer"];
    actions.NodeCreated = [](fb::Node const& node, Args& args, void** ctx) {
        *ctx = new RealityKeyerContext();
        auto* RealityKeyerCtx = static_cast<RealityKeyerContext*>(*ctx);
        RealityKeyerCtx->Load(node);
        RealityKeyerCtx->Setup();
    };
    actions.NodeRemoved = [](void* ctx, mz::fb::UUID const& id) {
        delete static_cast<RealityKeyerContext*>(ctx);
    };
    actions.PinValueChanged = [](void* ctx, mz::fb::UUID const& id, mz::Buffer* value) {};
    actions.EntryPoint = [](mz::Args& pins, void* ctx) {
        auto* RealityKeyerCtx = static_cast<RealityKeyerContext*>(ctx);
        RealityKeyerCtx->Run(pins);
        return true;
    };
}

} // namespace mz