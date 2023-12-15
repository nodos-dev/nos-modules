#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/PluginHelpers.hpp>
#include <AppService_generated.h>
#include <Windows.h>
#include <nosUtil/Thread.h>
#include <nosVulkanSubsystem/Helpers.hpp>

// nosNodes

NOS_INIT();
NOS_REGISTER_NAME(NVVFX)
NOS_REGISTER_NAME(In)
NOS_REGISTER_NAME(UpscaleFactor)
NOS_REGISTER_NAME(UpscaleStrength)
NOS_REGISTER_NAME(Out)

inline void SearchPath(const char* path)
{
#if _WIN32
	::SetDllDirectory(path);
#else
#pragma error "Not implemented"
#endif
}

extern nosVulkanSubsystem* nosVulkan = nullptr;

class nosWebRTCStatsLogger
{
public:
	nosWebRTCStatsLogger(std::string name, int refreshRate = 100) : RefreshRate(refreshRate)
	{
		Name_FPS = name + " FPS: ";
		Name_MAX_FPS = name + " MAX FPS: ";
		Name_MIN_FPS = name + " MIN FPS: ";
		startTime = std::chrono::high_resolution_clock::now();
	};

	void LogStats() {
		if (++FrameCount > RefreshRate)
		{
			// Clear stats for each 100 frame
			FrameCount = 0;
			MaxFPS = -INFINITY;
			MinFPS = INFINITY;
		}
		auto now = std::chrono::high_resolution_clock::now();
		auto FPS = 1.0 / (std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count()) * 1000.0;
		MaxFPS = (FPS > MaxFPS) ? (FPS) : (MaxFPS);
		MinFPS = (MinFPS > FPS) ? (FPS) : (MinFPS);
		nosEngine.WatchLog(Name_FPS.c_str(), std::to_string(FPS).c_str());
		nosEngine.WatchLog(Name_MAX_FPS.c_str(), std::to_string(MaxFPS).c_str());
		nosEngine.WatchLog(Name_MIN_FPS.c_str(), std::to_string(MinFPS).c_str());
		startTime = now;
	}

	float GetMaxFPS() const {
		return MaxFPS;
	}

	float GetMinFPS() const {
		return MinFPS;
	}

private:
	std::string Name_FPS, Name_MAX_FPS, Name_MIN_FPS;
	int FrameCount = 0;
	int RefreshRate = 100;
	std::chrono::steady_clock::time_point startTime;
	float MinFPS = 9999, MaxFPS = -9999, FPS = 0;

};

extern "C" __declspec(dllimport) void ProcessImage(uint8_t * inData, int width, int height, float strength, int targetRes, uint8_t * &outData, const char* modelsDir);

struct NVVFXNodeContext : nos::NodeContext {
	nosWebRTCStatsLogger logger;
	nos::fb::UUID NodeID, InputID, OutputID;
	nosResourceShareInfo InputFormatted{}, InputBuffer{};
	int UpscaleFactor;
	float UpscaleStrength;
	uint8_t* OutData = nullptr;

	NVVFXNodeContext(nos::fb::Node const* node) :NodeContext(node), logger("NVVFX") {
		nosEngine.RequestSubsystem(NOS_NAME_STATIC(NOS_VULKAN_SUBSYSTEM_NAME), 1, 0, (void**)&nosVulkan);
		
		for (const auto& pin : *node->pins()) {
			if (NSN_UpscaleFactor.Compare(pin->name()->c_str()) == 0) {
				UpscaleFactor = *(int*)pin->data()->data();
			}
			if (NSN_UpscaleStrength.Compare(pin->name()->c_str()) == 0) {
				UpscaleStrength = *(float*)pin->data()->data();
			}
			if (NSN_Out.Compare(pin->name()->c_str()) == 0) {
				OutputID = *pin->id();
			}
		}

	}


	void  OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer value) override {
		if (pinName == NSN_In) {
			nosResourceShareInfo input = nos::vkss::DeserializeTextureInfo(value.Data);
			if(InputFormatted.Memory.Handle == NULL || 
				(InputFormatted.Info.Texture.Width != input.Info.Texture.Width || InputFormatted.Info.Texture.Height != input.Info.Texture.Height))
			{
				if (InputFormatted.Memory.Handle != NULL) {
					nosVulkan->DestroyResource(&InputFormatted);
				}

				InputFormatted.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
				InputFormatted.Info.Texture.Width = input.Info.Texture.Width;
				InputFormatted.Info.Texture.Height = input.Info.Texture.Height;
				//TODO: Make use of tensor type before deciding to this
				InputFormatted.Info.Texture.Format = NOS_FORMAT_R8G8B8A8_SRGB;
				InputFormatted.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_SRC | NOS_IMAGE_USAGE_TRANSFER_DST);
				nosVulkan->CreateResource(&InputFormatted);

				if (InputBuffer.Memory.Handle != NULL) {
					nosVulkan->DestroyResource(&InputBuffer);
				}

				InputBuffer.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
				InputBuffer.Info.Buffer.Size = InputFormatted.Info.Texture.Width * InputFormatted.Info.Texture.Height * sizeof(uint8_t) * 4;
				InputBuffer.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_TRANSFER_DST);
				nosVulkan->CreateResource(&InputBuffer);
			}
		}
		if (pinName == NSN_UpscaleFactor) {
			UpscaleFactor = *static_cast<int*>(value.Data);
		}
		if (pinName == NSN_UpscaleStrength) {
			UpscaleStrength = *static_cast<float*>(value.Data);
		}
	}

	bool NeedsToUpdateOutput(nos::NodeExecuteArgs& args)
	{
		int targetHeight = InputFormatted.Info.Texture.Height * UpscaleFactor;
		int targetWidth = InputFormatted.Info.Texture.Width * UpscaleFactor;
		auto outInfo = nos::InterpretPinValue<nos::sys::vulkan::Texture>(args[NSN_Out].Data->Data);
		return outInfo->width() != targetWidth || outInfo->height() != targetHeight ||
			   outInfo->format() != (nos::sys::vulkan::Format)NOS_FORMAT_R8G8B8A8_SRGB || !outInfo->unscaled();
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* execArgs) {
		nos::NodeExecuteArgs args(execArgs);
		if (NeedsToUpdateOutput(args)) {
			nos::sys::vulkan::TTexture tex{};
			tex.unscaled = true;
			tex.width = InputFormatted.Info.Texture.Width * UpscaleFactor;
			tex.height = InputFormatted.Info.Texture.Height * UpscaleFactor;
			tex.format = (nos::sys::vulkan::Format)InputFormatted.Info.Texture.Format;
			nosEngine.SetPinValue(OutputID, nos::Buffer::From(tex));
		}
		
		nosCmd downloadCmd;
		nosResourceShareInfo in = nos::vkss::DeserializeTextureInfo(args[NSN_In].Data->Data);
		nosVulkan->Begin("NVVFX Download", &downloadCmd);
		nosVulkan->Copy(downloadCmd, &in, &InputFormatted, 0);
		nosVulkan->Copy(downloadCmd, &InputFormatted, &InputBuffer, 0);
		nosGPUEvent waitEvent;
		nosVulkan->End2(downloadCmd, NOS_TRUE, &waitEvent);
		nosVulkan->WaitGpuEvent(&waitEvent);


		int targetHeight = InputFormatted.Info.Texture.Height * UpscaleFactor;
		int targetWidth = InputFormatted.Info.Texture.Width * UpscaleFactor;
		auto data = nosVulkan->Map(&InputBuffer);
		//stbi_write_png("C:/TRASH/INPUT_FROM_PLUGIN.png", InputFormatted.Info.Texture.Width, InputFormatted.Info.Texture.Height, 4, data, InputFormatted.Info.Texture.Width * sizeof(uint8_t));

		ProcessImage(data, InputFormatted.Info.Texture.Width, InputFormatted.Info.Texture.Height, UpscaleStrength, targetHeight, OutData, "C:/WorkInParallel/MAXINE-VFX-SDK/models");
		//stbi_write_png("C:/TRASH/RESULT_FROM_PLUGIN.png", OutputTexture.Info.Texture.Width, OutputTexture.Info.Texture.Height, 4, OutData, OutputTexture.Info.Texture.Width * sizeof(uint8_t));

		
		nosResourceShareInfo out = nos::vkss::DeserializeTextureInfo(args[NSN_Out].Data->Data);
		nosCmd cmd;
		nosVulkan->Begin("NVVFX Upload", &cmd);
		nosVulkan->ImageLoad(cmd, OutData,
			nosVec2u(targetWidth, targetHeight),
			InputFormatted.Info.Texture.Format, &out);
		nosVulkan->End(cmd, NOS_FALSE);

		return NOS_RESULT_SUCCESS;

	}

	nosResult BeginCopyTo(nosCopyInfo* cpy) override {
		return NOS_RESULT_SUCCESS;
	}

	void EndCopyTo(nosCopyInfo* cpy) override {
	}
};

extern "C"
{

	NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outCount, nosNodeFunctions** outFunctions) {
		*outCount = (size_t)(1);
		if (!outFunctions)
			return NOS_RESULT_SUCCESS;

		NOS_BIND_NODE_CLASS(NSN_NVVFX, NVVFXNodeContext, outFunctions[0]);


		return NOS_RESULT_SUCCESS;
	}
}