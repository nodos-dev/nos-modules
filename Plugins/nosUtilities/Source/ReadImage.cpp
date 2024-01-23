// Copyright Nodos AS. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>

// External
#include <stb_image.h>
#include <stb_image_write.h>

// Framework
#include <Builtins_generated.h>
#include <AppService_generated.h>

// nosNodes
#include <nosVulkanSubsystem/Helpers.hpp>
#include "../Shaders/SRGB2Linear.frag.spv.dat"

#include "Names.h"

namespace nos::utilities
{
extern nosVulkanSubsystem* nosVulkan;

NOS_REGISTER_NAME(SRGB2Linear_Pass);
NOS_REGISTER_NAME(SRGB2Linear_Shader);
NOS_REGISTER_NAME_SPACED(Nos_Utilities_ReadImage, "nos.utilities.ReadImage")

static nosResult GetFunctions(size_t* count, nosName* names, nosPfnNodeFunctionExecute* fns)
{
    *count = 1;
    if(!names || !fns)
        return NOS_RESULT_SUCCESS;
    
    *names = NOS_NAME_STATIC("ReadImage_Load");
	*fns = [](void* ctx, const nosNodeExecuteArgs* nodeArgs, const nosNodeExecuteArgs* functionArgs)
    {
		nos::NodeExecuteArgs args(nodeArgs);
		std::filesystem::path path = InterpretPinValue<const char>(args[NSN_Path].Data->Data);
		try
		{
			if (!std::filesystem::exists(path))
			{
				nosEngine.LogE("Read Image cannot load file %s", path.string().c_str());
				return;
			}
			
			sys::vulkan::Texture* out = InterpretPinValue<sys::vulkan::Texture>(args[NSN_Out].Data->Data);

			int w, h, n;
			u8* img = stbi_load(path.string().c_str(), &w, &h, &n, 4);
			if (!img)
			{
				nosEngine.LogE("Couldn't load image from %s.", path.string().c_str());
				return;
			}
			sys::vulkan::TTexture tex;
			tex.width = w;
			tex.height = h;

			if (*InterpretPinValue<bool>(args[NSN_sRGB].Data->Data))
				tex.format = sys::vulkan::Format::R8G8B8A8_SRGB;
			else
				tex.format = sys::vulkan::Format::R8G8B8A8_UNORM;

			if (tex.width != out->width() || tex.height != out->height() || tex.format != out->format())
			{
				nosEngine.SetPinValue(args[NSN_Out].Id, nos::Buffer::From(tex));
			}
			nosResourceShareInfo outRes = vkss::DeserializeTextureInfo(args[NSN_Out].Data->Data);

			nosCmd cmd{};
			nosVulkan->Begin("ReadImage: Load", &cmd);
			nosVulkan->ImageLoad(cmd, img, nosVec2u(w, h), NOS_FORMAT_R8G8B8A8_SRGB, &outRes);
			nosCmdEndParams endParams {.ForceSubmit = true};
			nosVulkan->End(cmd, &endParams);
			
			free(img);
			
			flatbuffers::FlatBufferBuilder fbb;
			HandleEvent(CreateAppEvent(fbb, app::CreatePinDirtied(fbb, &args[NSN_Out].Id)));
		}
		catch(const std::exception& e)
		{
			nosEngine.LogE("Error while loading image: %s", e.what());
		}
    };
    
    return NOS_RESULT_SUCCESS;
}


nosResult RegisterReadImage(nosNodeFunctions* fn)
{
	fn->ClassName = NSN_Nos_Utilities_ReadImage;
	fn->GetFunctions = GetFunctions;

	nosShaderInfo shader = {.Key=NSN_SRGB2Linear_Shader, .Source = { .SpirvBlob = {(void*)SRGB2Linear_frag_spv, sizeof(SRGB2Linear_frag_spv)}}};
	auto ret = nosVulkan->RegisterShaders(1, &shader);
	if (NOS_RESULT_SUCCESS != ret)
		return ret;

	nosPassInfo pass = {
		.Key = NSN_SRGB2Linear_Pass,
		.Shader = NSN_SRGB2Linear_Shader,
		.MultiSample = 1,
	};
	return nosVulkan->RegisterPasses(1, &pass);
}

// void RegisterReadImage(nosNodeFunctions* fn)
// {
	// auto& actions = functions["nos.ReadImage"];

	// actions.NodeCreated = [](fb::Node const& node, Args& args, void** context) {
	// 	*context = new ReadImageContext(node);
	// };

	// actions.EntryPoint = [](nos::Args& args, void* context) mutable {
	// 	auto path = args.Get<char>("Path");
	// 	if (!path || strlen(path) == 0)
	// 		return false;

	// 	i32 width, height, channels;
	// 	auto* ctx = static_cast<ReadImageContext*>(context);
	// 	u8* img = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
	// 	bool ret = !!img && ctx->Upload(img, width, height, args.GetBuffer("Out"));
	// 	if (!ret)
	// 	{
	// 		nosEngine.LogE("ReadImage: Failed to load image");
	// 		flatbuffers::FlatBufferBuilder fbb;
	// 		std::vector<flatbuffers::Offset<nos::fb::NodeStatusMessage>> messages{nos::fb::CreateNodeStatusMessageDirect(
	// 			fbb, "Failed to load image", nos::fb::NodeStatusMessageType::FAILURE)};
	// 		HandleEvent(CreateAppEvent(
	// 			fbb,
	// 			nos::app::CreatePartialNodeUpdateDirect(fbb, &ctx->NodeId, ClearFlags::NONE, 0, 0, 0, 0, 0, 0, &messages)));
	// 	}
	// 	if (img)
	// 		stbi_image_free(img);

	// 	return ret;
	// };

	// actions.NodeRemoved = [](void* ctx, nos::fb::UUID const& id) { delete static_cast<ReadImageContext*>(ctx); };
// }

} // namespace nos::utilities