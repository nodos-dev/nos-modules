// Copyright MediaZ AS. All Rights Reserved.

#include "BasicMain.h"

// std
#include <string.h>
#include <stdio.h>

// External
#include <stb_image.h>
#include <stb_image_write.h>

// Framework
#include <Args.h>
#include <Builtins_generated.h>
#include <AppService_generated.h>

// mzNodes
#include "Shaders/SRGB2Linear.frag.spv.dat"

namespace mz
{

void RegisterReadImageNode(NodeActionsMap& functions)
{
    auto& actions = functions["mz.ReadImage"];

    struct ReadImageContext : NodeContext
    {
        ReadImageContext(fb::Node const& node) : NodeContext(node)
        {
            SetupShaders();
            RegisterPasses();
        }
        ~ReadImageContext()
        {
            DestroyResources();
        }

        bool Upload(u8* img, i32 width, i32 height, mz::Buffer* OutPin)
        {
            u64 size = width * height * 4;
            mz::fb::Buffer buf;
            buf.mutate_usage((mz::fb::BufferUsage)(mz::fb::BufferUsage::TRANSFER_DST | mz::fb::BufferUsage::TRANSFER_SRC));
            buf.mutate_size(size);
            GServices.Log("ReadImage: Requesting to create a buffer");
            GServices.Create(buf);
            auto buf2write = GServices.Map(buf);
            if (!buf2write)
            {
                GServices.Log("ReadImage: Failed to map buffer");
                return false;
            }
            GServices.Log("ReadImage: Writing to buffer");
            memcpy(buf2write, img, size);
            mz::fb::TTexture srgbTexture = {
                .width  = (u32)width,
                .height = (u32)height,
                .format = mz::fb::Format::R8G8B8A8_UNORM,
                .usage = mz::fb::ImageUsage::SAMPLED | mz::fb::ImageUsage::TRANSFER_DST | mz::fb::ImageUsage::TRANSFER_SRC,
            };
            
            GServices.Create(srgbTexture);
            GServices.Copy(buf, srgbTexture);

            // renew out pin if size does not match
            auto outTex = OutPin->As<mz::fb::TTexture>();
            if (outTex.width != width || 
                outTex.height != height)
            {
                GServices.Destroy(outTex);
                outTex.width = width;
                outTex.height = height;
                GServices.Create(outTex);
                *OutPin = mz::Buffer::From(outTex);
            }

			app::TRunPass linearPass;
			linearPass.pass = "ReadImage_SRGB2Linear_Pass_" + UUID2STR(NodeId);
            linearPass.output.reset(&outTex); // set pass output to out texture pin
			
			AddUniform(linearPass, "Input", Buffer::From(srgbTexture));
			GServices.MakeAPICall(linearPass, true);
            linearPass.output.release(); // do not try to delete our stack object

            GServices.Destroy(buf);
            GServices.Destroy(srgbTexture);
            return true;
        }

        void SetupShaders()
        {
            static bool shadersRegistered = false;
            if (shadersRegistered)
            {
                return;
            }
            GServices.MakeAPICalls(true,
                                  app::TRegisterShader{
                                      .key = "ReadImage_SRGB2Linear",
                                      .spirv = ShaderSrc<sizeof(SRGB2Linear_frag_spv)>(SRGB2Linear_frag_spv)});
            shadersRegistered = true;
        }

        void RegisterPasses()
        {
            GServices.MakeAPICalls(true,
                                  app::TRegisterPass{
                                      .key = "ReadImage_SRGB2Linear_Pass_" + UUID2STR(NodeId),
                                      .shader = "ReadImage_SRGB2Linear",
                                  });
        }

        void DestroyResources()
        {
            GServices.MakeAPICalls(
                true,
                app::TUnregisterPass{
                    .key = "ReadImage_SRGB2Linear_Pass_" + UUID2STR(NodeId)});
        }
    };

    actions.NodeCreated = [](fb::Node const& node, Args& args, void** context) 
    {
        *context = new ReadImageContext(node);
    };

    actions.EntryPoint = [](mz::Args& args, void* context) mutable
    {
        auto path = args.Get<char>("Path");
        if (!path || strlen(path) == 0)
            return false;

        i32 width, height, channels;
        auto* ctx = static_cast<ReadImageContext*>(context);
        u8* img = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
        bool ret = !!img && ctx->Upload(img, width, height, args.GetBuffer("Out"));
		if (!ret)
		{
			GServices.LogE("ReadImage: Failed to load image");
			flatbuffers::FlatBufferBuilder fbb;
			std::vector<flatbuffers::Offset<mz::fb::NodeStatusMessage>> messages 
                { mz::fb::CreateNodeStatusMessageDirect(fbb, "Failed to load image", mz::fb::NodeStatusMessageType::FAILURE)};
            GServices.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &ctx->NodeId, ClearFlags::NONE, 0, 0, 0, 0, 0, 0, &messages)));
		}
        if (img)
		    stbi_image_free(img);

        return ret;
    };

    actions.NodeRemoved = [](void* ctx, mz::fb::UUID const& id) {
        delete static_cast<ReadImageContext*>(ctx);
    };
}

} // namespace mz