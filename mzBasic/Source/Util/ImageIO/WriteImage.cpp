// Copyright MediaZ AS. All Rights Reserved.

#include "BasicMain.h"
#include "Args.h"
#include "Builtins_generated.h"
#include <AppService_generated.h>
#include <string.h>
#include <stdio.h>

#include <stb_image.h>
#include <stb_image_write.h>

#include "Shaders/Linear2SRGB.frag.spv.dat"
#include "Filters/Sampler.frag.spv.dat"

namespace mz
{

void RegisterWriteImageNode(NodeActionsMap& functions)
{
    auto& actions = functions["mz.WriteImage"];

    struct WriteImageContext : NodeContext
    {
        std::filesystem::path Path;
        mz::fb::TTexture InputTexture;

        WriteImageContext(fb::Node const& node) : NodeContext(node)
        {
            SetupShaders();
            RegisterPasses();
        }

        ~WriteImageContext()
        {
            DestroyResources();
        }

        mz::fb::TTexture ConvertLinearToSrgb(mz::fb::TTexture texture)
        {
			mz::fb::TTexture srgbTexture = {
                .width = texture.width,
                .height = texture.height,
                .format = mz::fb::Format::R8G8B8A8_UNORM,
                .usage = mz::fb::ImageUsage::RENDER_TARGET | mz::fb::ImageUsage::TRANSFER_DST | mz::fb::ImageUsage::TRANSFER_SRC,
            };

			mzEngine.Create(srgbTexture);
            app::TRunPass srgbPass;
            srgbPass.pass = "WriteImage_Linear2SRGB_Pass_" + UUID2STR(NodeId);
			AddUniform(srgbPass, "Input", mz::Buffer::From(texture));
            srgbPass.output.reset(&srgbTexture);
            mzEngine.MakeAPICall(srgbPass, true);
            srgbPass.output.release();
            return srgbTexture;
        }

        bool SaveTextureToFile(mz::fb::TTexture& texture, bool ownsTexture)
        {
            mz::fb::Buffer buf;
            buf.mutate_usage((mz::fb::BufferUsage)(mz::fb::BufferUsage::TRANSFER_DST | mz::fb::BufferUsage::TRANSFER_SRC));
            buf.mutate_size(mzEngine.GetAllocatedSize(EngineNodeServices::ResBorrow(texture)));
            bool success = false;
            mzEngine.Create<mz::fb::Buffer>(buf);
            if (auto buf2write = mzEngine.Map(buf))
            {
                mzEngine.Copy(texture, buf);
                if (stbi_write_png(Path.string().c_str(), texture.width, texture.height, 4, buf2write, texture.width * 4))
                {
                    success = true;
                }
                else
                {
                    mzEngine.LogE("WriteImage: Unable to write frame to file", "");
                }
            }
            mzEngine.Destroy(buf);
            if (ownsTexture)
            {
                mzEngine.Destroy(texture);
            }
            return success;
        }

        void OnPinValueChanged(mz::fb::UUID const& id, void* value)
        {
            if (GetPinName(id) == "Path")
            {
                auto* str = (char*)value;
                std::filesystem::path newPath(str);
                if (newPath != Path)
                {
                    Path = newPath;
                    mzEngine.Log("WriteImage: New path: " + Path.generic_string());
                }
            }
            else if (GetPinName(id) == "In")
            {
                mzEngine.Log("WriteImage: New input texture", "");
                this->InputTexture = TableBufferToNativeTable<mz::fb::Texture>(value);
            }
        }

        bool Write()
        {
            auto srgbTexture = ConvertLinearToSrgb(InputTexture);
            if (SaveTextureToFile(srgbTexture, true))
            {
                mzEngine.Log("WriteImage: Texture saved to " + Path.generic_string());
                return true;
            }
            else
            {
                mzEngine.LogE("WriteImage: Failed to save texture");
                return false;
            }
        }

        void SetupShaders()
        {
            static bool shadersRegistered = false;
            if (shadersRegistered)
            {
                return;
            }
            mzEngine.MakeAPICalls(true,
                                  app::TRegisterShader{
                                      .key = "WriteImage_Linear2SRGB",
                                      .spirv = ShaderSrc<sizeof(Linear2SRGB_frag_spv)>(Linear2SRGB_frag_spv)});
            shadersRegistered = true;
        }

        void RegisterPasses()
        {
            mzEngine.MakeAPICalls(true,
                                  app::TRegisterPass{
                                      .key = "WriteImage_Linear2SRGB_Pass_" + UUID2STR(NodeId),
                                      .shader = "WriteImage_Linear2SRGB",
                                  });
        }

        void DestroyResources()
        {
            mzEngine.MakeAPICalls(true,
                                  app::TUnregisterPass{
                                      .key = "WriteImage_Linear2SRGB_Pass_" + UUID2STR(NodeId)});
        }
    };

    actions.NodeFunctions["WriteImage_Save"] = [](Args& nodePins, Args& funcParams, void* context) {
        auto* ctx = static_cast<WriteImageContext*>(context);
        return ctx->Write();
    };
    
    actions.NodeCreated = [](fb::Node const& node, Args& args, void** context) {
        *context = new WriteImageContext(node);
        auto* ctx = static_cast<WriteImageContext*>(*context);
        ctx->InputTexture = args.GetBuffer("In")->As<mz::fb::TTexture>();
    };

    actions.PinValueChanged = [](void* context, mz::fb::UUID const& id, mz::Buffer* value) {
        auto* ctx = static_cast<WriteImageContext*>(context);
        ctx->OnPinValueChanged(id, value->data());
    };

    actions.EntryPoint = [](mz::Args& args, void* context) {
        auto* ctx = static_cast<WriteImageContext*>(context);
        return true;
    };

    actions.NodeRemoved = [](void* ctx, mz::fb::UUID const& id) {
        delete static_cast<WriteImageContext*>(ctx);
    };
}

} // namespace mz