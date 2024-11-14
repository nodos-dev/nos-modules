// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

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
NOS_REGISTER_NAME_SPACED(Nos_Utilities_ReadImage, "nos.utilities.ReadImage")

enum State
{
    Idle = 0,
    Loading = 1,
    Failed = 2,
};

struct ReadImageContext : NodeContext
{
    std::atomic<State> CurrentState;
    decltype(Clock::now()) TimeStarted;
	std::filesystem::path FilePath;

	std::mutex OutImageDecRefCallbacksMutex;
	std::queue<std::function<void()>> OutImageDecRefCallbacks;

	ReadImageContext(nosFbNode const* node) : 
		NodeContext(node), 
		CurrentState(State::Idle), 
		TimeStarted(Clock::now())
	{
	}

	~ReadImageContext()
	{
		FlushImageDecRefCallbacks();
	}

	void UpdateStatus(State newState)
	{
        if(newState == CurrentState.exchange(newState))
        {
            return;
        }

        flatbuffers::FlatBufferBuilder fbb;
        std::vector<flatbuffers::Offset<nos::fb::NodeStatusMessage>> msg;
        switch(newState)
        {
        case State::Loading:
		    TimeStarted = Clock::now();
			msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, "Loading image", fb::NodeStatusMessageType::INFO));
            break;
        case State::Idle:
        {
			std::stringstream ss;
			auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - TimeStarted).count();
			auto statusText = std::string("Image ") + FilePath.filename().string() + " loaded in " + std::to_string(dt) + "ms";
			msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, statusText.c_str(), fb::NodeStatusMessageType::INFO));
            break;
        }
        case State::Failed:
            msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, "Failed to load image", fb::NodeStatusMessageType::FAILURE));
            break;
        }

        HandleEvent(CreateAppEvent(fbb, nos::CreatePartialNodeUpdateDirect(fbb, &NodeId, ClearFlags::NONE, 0, 0, 0, 0, 0, 0, &msg)));
	}

	void FlushImageDecRefCallbacks()
	{
		std::lock_guard<std::mutex> lock(OutImageDecRefCallbacksMutex);
		while (!OutImageDecRefCallbacks.empty())
		{
			OutImageDecRefCallbacks.front()();
			OutImageDecRefCallbacks.pop();
		}
	}

	static nosResult OnImageLoaded(void* ctx, nosFunctionExecuteParams* params)
	{
		auto c = (ReadImageContext*)ctx;
		c->FlushImageDecRefCallbacks();
		return NOS_RESULT_SUCCESS;
	}
	
	static nosResult Load(void* ctx, nosFunctionExecuteParams* params)
	{
		auto c = (ReadImageContext*)ctx;
		if (c->CurrentState == State::Loading)
		{
			nosEngine.LogE("Read Image is already loading an image.");
			return NOS_RESULT_FAILED;
		}

		c->UpdateStatus(State::Loading);

		nos::NodeExecuteParams nodeParams(params->ParentNodeExecuteParams);
		std::filesystem::path path = InterpretPinValue<const char>(nodeParams[NSN_Path].Data->Data);
		c->FilePath = path.string();
		try
		{
			if (!std::filesystem::exists(path))
			{
				nosEngine.LogE("Read Image cannot load file %s", path.string().c_str());
				c->UpdateStatus(State::Failed);
				return NOS_RESULT_FAILED;
			}

			int w, h, n;
			stbi_info(path.string().c_str(), &w, &h, &n);

			if (w < 0 || h < 0 || n < 0) {
				nosEngine.LogE("STBI couldn't load image from %s.", path.string().c_str());
				c->UpdateStatus(State::Failed);
				return NOS_RESULT_FAILED;
			}

			nosResourceShareInfo outRes = {
				.Info = {.Type = NOS_RESOURCE_TYPE_TEXTURE,
							.Texture = {.Width = (u32)w, .Height = (u32)h, .Format = NOS_FORMAT_R8G8B8A8_UNORM, .FieldType = NOS_TEXTURE_FIELD_TYPE_PROGRESSIVE}} };

			// unless reading raw bytes, this is useless since samplers convert to linear space automatically
			if (*InterpretPinValue<bool>(nodeParams[NSN_sRGB].Data->Data))
				outRes.Info.Texture.Format = NOS_FORMAT_R8G8B8A8_SRGB;

			std::filesystem::path path = InterpretPinValue<const char>(nodeParams[NSN_Path].Data->Data);
			auto outPinId = nodeParams[NSN_Out].Id;

			std::thread([c, outPinId, path, outRes]() mutable {
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

					nosVulkan->CreateResource(&outRes);
					nosVulkan->SetResourceTag(&outRes, "ReadImage Texture");

					nosCmd cmd{};
					nosCmdBeginParams beginParams {
						.Name = NOS_NAME("ReadImage Load"),
						.AssociatedNodeId = c->NodeId,
						.OutCmdHandle = &cmd
					};
					nosVulkan->Begin2(&beginParams);
					nosVulkan->ImageLoad(cmd, img, nosVec2u(w, h), NOS_FORMAT_R8G8B8A8_SRGB, &outRes);
					nosCmdEndParams endParams{ .ForceSubmit = true };
					nosVulkan->End(cmd, &endParams);

					nosEngine.SetPinValue(outPinId, nos::Buffer::From(vkss::ConvertTextureInfo(outRes)));

					{
						std::lock_guard<std::mutex> lock(c->OutImageDecRefCallbacksMutex);
						c->OutImageDecRefCallbacks.push([c, outRes]() {
							nosVulkan->DestroyResource(&outRes);
						});
					}
					
					nosEngine.CallNodeFunction(c->NodeId, NOS_NAME_STATIC("OnImageLoaded"));

					free(img);
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
			c->UpdateStatus(State::Failed);
		}
		return NOS_RESULT_SUCCESS;
	}

	static nosResult GetFunctions(size_t* count, nosName* names, nosPfnNodeFunctionExecute* fns)
	{
		*count = 2;
		if (!names || !fns)
			return NOS_RESULT_SUCCESS;

		names[0] = NOS_NAME_STATIC("ReadImage_Load");
		names[1] = NOS_NAME_STATIC("OnImageLoaded");

		fns[0] = &Load;
		fns[1] = &OnImageLoaded;

		return NOS_RESULT_SUCCESS;
	}
};

nosResult RegisterReadImage(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NSN_Nos_Utilities_ReadImage, ReadImageContext, fn);
	return NOS_RESULT_SUCCESS;
}

} // namespace nos::utilities