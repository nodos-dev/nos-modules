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

#include "Names.h"

#include <atomic>
#include <chrono>
#include <sstream>

using Clock = std::chrono::high_resolution_clock;

namespace nos::vkss
{
#include <nosVulkanSubsystem/CAPIStructHelpers.inl>
}

namespace nos::utilities
{
extern nosVulkanSubsystem* nosVulkan;

NOS_REGISTER_NAME(SRGB2Linear_Pass);
NOS_REGISTER_NAME(SRGB2Linear_Shader);
NOS_REGISTER_NAME_SPACED(Nos_Utilities_ReadImage, "nos.utilities.ReadImage")

enum State
{
    Idle = 0,
    Loading = 1,
};

struct ReadImageContext
{
    std::atomic<State> state;
    decltype(Clock::now()) time_started;
	fb::UUID nodeid;
	std::string load_path;


	void UpdateStatus(State newState)
	{
        if(newState == state.exchange(newState))
        {
            return;
        }

        flatbuffers::FlatBufferBuilder fbb;
        std::vector<flatbuffers::Offset<nos::fb::NodeStatusMessage>> msg;
        if(newState == State::Loading)
        {
			time_started = Clock::now();
			msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, "Loading image", fb::NodeStatusMessageType::INFO));
        }
        else
        {
			std::stringstream ss;
			auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - time_started);
            ss << "Read Image: Loaded in " << dt << "\nFile: " << load_path;

            nosEngine.LogDI(load_path.c_str(), ss.str().c_str());
        }
        HandleEvent(CreateAppEvent(fbb, nos::CreatePartialNodeUpdateDirect(fbb, &nodeid, ClearFlags::NONE, 0, 0, 0, 0, 0, 0, &msg)));
	}
};

static nosResult GetFunctions(size_t* count, nosName* names, nosPfnNodeFunctionExecute* fns)
{
    *count = 1;
    if(!names || !fns)
        return NOS_RESULT_SUCCESS;
    
    *names = NOS_NAME_STATIC("ReadImage_Load");
    
	*fns = [](void* ctx, const nosNodeExecuteArgs* nodeArgs, const nosNodeExecuteArgs* functionArgs)
    {
        auto c = (ReadImageContext*)ctx;
        if(c->state != State::Idle)
        {
            nosEngine.LogE("Read Image is already loading an image.");
            return;
        }
        c->UpdateStatus(State::Loading);

		nos::NodeExecuteArgs args(nodeArgs);
		std::filesystem::path path = InterpretPinValue<const char>(args[NSN_Path].Data->Data);
		c->load_path = path.string();
		try
		{
			if (!std::filesystem::exists(path))
			{
				nosEngine.LogE("Read Image cannot load file %s", path.string().c_str());
				return;
			}

			int w, h, n;
			stbi_info(path.string().c_str(), &w, &h, &n);
			
			nosResourceShareInfo outRes = {
				.Info = {.Type = NOS_RESOURCE_TYPE_TEXTURE,
						 .Texture = {.Width = (u32)w, .Height = (u32)h, .Format = NOS_FORMAT_R8G8B8A8_UNORM, .FieldType = NOS_TEXTURE_FIELD_TYPE_PROGRESSIVE}}};

			// unless reading raw bytes, this is useless since samplers convert to linear space automatically
			if (*InterpretPinValue<bool>(args[NSN_sRGB].Data->Data))
				outRes.Info.Texture.Format = NOS_FORMAT_R8G8B8A8_SRGB;

			nosEngine.SetPinValue(args[NSN_Out].Id, nos::Buffer::From(nos::vkss::ConvertTextureInfo(outRes)));

			std::thread([c, nodeArgs = vkss::OwnedNodeExecuteArgs(*nodeArgs), &outRes] {
				nos::NodeExecuteArgs args(&nodeArgs);
				std::filesystem::path path = InterpretPinValue<const char>(args[NSN_Path].Data->Data);
				try
				{
					if (!std::filesystem::exists(path))
					{
						nosEngine.LogE("Read Image cannot load file %s", path.string().c_str());
						return;
					}

					int w, h, n;
					u8* img = stbi_load(path.string().c_str(), &w, &h, &n, 4);
					if (!img)
					{
						nosEngine.LogE("Couldn't load image from %s.", path.string().c_str());
						return;
					}

					auto res = vkss::DeserializeTextureInfo(args[NSN_Out].Data->Data);

					nosCmd cmd{};
					nosVulkan->Begin("ReadImage: Load", &cmd);
					nosVulkan->ImageLoad(cmd, img, nosVec2u(w, h), NOS_FORMAT_R8G8B8A8_SRGB, &res);
					nosCmdEndParams endParams{.ForceSubmit = true};
					nosVulkan->End(cmd, &endParams);

					free(img);
					flatbuffers::FlatBufferBuilder fbb;
					HandleEvent(CreateAppEvent(fbb, app::CreatePinDirtied(fbb, &args[NSN_Out].Id)));
				}
				catch (const std::exception& e)
				{
					nosEngine.LogE("Error while loading image: %s", e.what());
				}

                c->UpdateStatus(State::Idle);
			}).detach();
		}
		catch (const std::exception& e)
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

	fn->OnNodeCreated = [](const nosFbNode* node, void** outCtxPtr)
	{
		*outCtxPtr = new ReadImageContext{.nodeid = *node->id()};
	};
	fn->OnNodeDeleted = [](void* ctx, nosUUID nodeId)
	{
		delete (ReadImageContext*)ctx;
	};

	return NOS_RESULT_SUCCESS;
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