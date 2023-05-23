#include "GaussianBlur.hpp"

#include "MediaZ/Helpers.hpp"

#include "GaussianBlur.frag.spv.dat"

namespace mz::filters 
{

struct GaussBlurContext
{
    mz::fb::TTexture IntermediateTexture;
    mz::fb::UUID NodeId;

    GaussBlurContext(fb::Node const& node)
    {
        NodeId = *node.id();
        RegisterShaders();
        RegisterPasses();

		IntermediateTexture.filtering = mz::fb::Filtering::LINEAR;
		IntermediateTexture.usage = mz::fb::ImageUsage::RENDER_TARGET | mz::fb::ImageUsage::SAMPLED;
    }

    ~GaussBlurContext()
    {
        // DestroyResources();
    }

    void RegisterShaders()
    {
        static bool registered = false;
        if (registered)
            return;
        mzEngine.RegisterShader("Gaussian_Blur", MzBuffer {
            .Data = (void*)GaussianBlur_frag_spv, 
            .Size = sizeof(GaussianBlur_frag_spv)
        }); // TODO Check result
        registered = true;
    }

    void RegisterPasses()
    {
        auto key = "Gaussian_Blur_Pass_" + mz::Uuid2String(NodeId);
        MzPassInfo info = {};
        info.Key = key.c_str();
        info.Shader = "Gaussian_Blur";
        mzEngine.RegisterPass(&info); // TODO Check result
    }

    // void DestroyResources()
    // {
    //     if (IntermediateTexture.handle)
    //         GServices.Destroy(IntermediateTexture);
    //     GServices.MakeAPICalls(true, app::TUnregisterPass{
    //                                     .key = "Gaussian_Blur_Pass_" + UUID2STR(NodeId)});
    // }

    // void SetupIntermediateTexture(mz::fb::TTexture* outputTexture)
    // {
    //     if (IntermediateTexture.width == outputTexture->width &&
    //         IntermediateTexture.height == outputTexture->height &&
    //         IntermediateTexture.format == outputTexture->format)
    //         return;

    //     GServices.Destroy(IntermediateTexture);
    //     IntermediateTexture.width = outputTexture->width;
    //     IntermediateTexture.height = outputTexture->height;
    //     IntermediateTexture.format = outputTexture->format;
    //     GServices.Create(IntermediateTexture);
    // }

    // void Run(mz::Args& pins)
    // {
    //     float softness = *pins.Get<float>("Softness");
    //     softness += 1.0f; // [0, 1] -> [1, 2]
    //     mz::fb::vec2 kernelSize = *pins.Get<mz::fb::vec2>("Kernel_Size");
    //     float horzKernel = kernelSize.x();
    //     float vertKernel = kernelSize.y();

	// 	auto outputTexture = pins.GetBuffer("Output")->As<mz::fb::TTexture>();
	// 	SetupIntermediateTexture(&outputTexture);

    //     // Pass 1 begin
    //     app::TRunPass horzPass;
    //     horzPass.pass = "Gaussian_Blur_Pass_" + UUID2STR(NodeId);
    //     CopyUniformFromPin(horzPass, pins, "Input");
    //     AddUniform(horzPass, "Softness", &softness, sizeof(softness));
    //     AddUniform(horzPass, "Kernel_Size", &horzKernel, sizeof(horzKernel));
    //     u32 passType = 0; // Horizontal pass
    //     AddUniform(horzPass, "Pass_Type", &passType, sizeof(passType));
    //     horzPass.output.reset(&IntermediateTexture);
    //     // Pass 1 end
    //     // Pass 2 begin
    //     app::TRunPass vertPass;
    //     vertPass.pass = "Gaussian_Blur_Pass_" + UUID2STR(NodeId);
    //     AddUniform(vertPass, "Input", mz::Buffer::From(IntermediateTexture));
    //     AddUniform(vertPass, "Softness", &softness, sizeof(softness));
    //     AddUniform(vertPass, "Kernel_Size", &vertKernel, sizeof(vertKernel));
    //     passType = 1; // Vertical pass
    //     AddUniform(vertPass, "Pass_Type", &passType, sizeof(passType));
    //     vertPass.output.reset(&outputTexture);
    //     // Pass 2 end
    //     // Run passes
    //     GServices.MakeAPICalls(false, horzPass, vertPass);
    //     vertPass.output.release();
    //     horzPass.output.release();
    // }
};

}

void MZAPI_CALL GaussianBlur_OnNodeCreated(const MzFbNode* node, void** outCtxPtr)
{
    *outCtxPtr = new mz::filters::GaussBlurContext(*node);
}

bool MZAPI_CALL GaussianBlur_ExecuteNode(void* ctx, const MzNodeExecuteArgs* args)
{   
    (mz::filters::GaussBlurContext*)ctx->Run(args);
    return true;
}