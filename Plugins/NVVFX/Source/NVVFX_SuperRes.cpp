#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/PluginHelpers.hpp>
#include <AppService_generated.h>
#include <Windows.h>
#include <nosUtil/Thread.h>
#include <nosVulkanSubsystem/Helpers.hpp>
#include "CUDAVulkanInterop.h"
#include "NVVFXAppRunner.h"
#include "NVVFX_Names.h"

// nosNodes

struct NVVFX_SuperRes_NodeContext : nos::NodeContext {

	NVVFXAppRunner VFX;

	nos::fb::UUID NodeID, InputID, OutputID;
	nosResourceShareInfo InputFormatted = {}, InputBuffer = {}, output = {};
	nosResourceShareInfo OutputPre = {};

	std::atomic_bool OutReady = false;
	CUDAVulkanInterop interop;
	int lastWidth, lastHeight;

	NvCVImage nvcv_NV_Input;
	NvCVImage nvcv_NV_Output;
	NvCVImage nvcv_NOS_Input;
	NvCVImage nvcv_NOS_Output;

	int UpscaleFactor;
	float UpscaleStrength;
	std::filesystem::path ModelsPath;
	bool OutputSizeSet = false;

	NVVFX_SuperRes_NodeContext(nos::fb::Node const* node) :NodeContext(node) {
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



		ModelsPath = std::filesystem::path("C:/WorkInParallel/MAXINE-VFX-SDK/models");
		VFX.CreateSuperResolutionEffect("C:/WorkInParallel/MAXINE-VFX-SDK/models");
	}


	void  OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer value) override {
		if (pinName == NSN_In) {
			nosResourceShareInfo input = nos::vkss::DeserializeTextureInfo(value.Data);
			lastWidth = input.Info.Texture.Width;
			lastHeight = input.Info.Texture.Height;

		}
		if (pinName == NSN_UpscaleFactor) {
			UpscaleFactor = *static_cast<int*>(value.Data);
		}
		if (pinName == NSN_UpscaleStrength) {
			UpscaleStrength = *static_cast<float*>(value.Data);
		}
		if (pinName == NSN_ModelsPath) {
			ModelsPath = static_cast<const char*>(value.Data);
		}
	}

	bool RefreshInputFormatted(nosResourceShareInfo* in)
	{
		if (InputFormatted.Memory.Handle == NULL)
			return true;
		bool sizeChanged = (in->Info.Texture.Width != InputFormatted.Info.Texture.Width)
			|| (in->Info.Texture.Height != InputFormatted.Info.Texture.Height);


		return sizeChanged;
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* execArgs) {
		nos::NodeExecuteArgs args(execArgs);
		nosResult res = NOS_RESULT_SUCCESS;
		nosResourceShareInfo in = nos::vkss::DeserializeTextureInfo(args[NSN_In].Data->Data);

		if (!OutputSizeSet) {

			float upscaleFactor = 2;

			InputFormatted.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
			InputFormatted.Info.Texture.Width = in.Info.Texture.Width;
			InputFormatted.Info.Texture.Height = in.Info.Texture.Height;
			InputFormatted.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_DST | NOS_IMAGE_USAGE_TRANSFER_SRC);
			InputFormatted.Info.Texture.Format = NOS_FORMAT_R32G32B32A32_SFLOAT;
			nosVulkan->CreateResource(&InputFormatted);

			res = interop.nosTextureToNVCVImage(InputFormatted, nvcv_NOS_Input);

			nosResourceShareInfo dummy = {};
			dummy.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
			dummy.Info.Texture.Width = InputFormatted.Info.Texture.Width * upscaleFactor;
			dummy.Info.Texture.Height = InputFormatted.Info.Texture.Height * upscaleFactor;
			dummy.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_DST | NOS_IMAGE_USAGE_TRANSFER_SRC);
			dummy.Info.Texture.Format = InputFormatted.Info.Texture.Format;
			dummy.Memory.ExternalMemory.HandleType = NOS_EXTERNAL_MEMORY_HANDLE_TYPE_WIN32;

			auto texFb = nos::vkss::ConvertTextureInfo(dummy);
			texFb.unscaled = true;
			auto texFbBuf = nos::Buffer::From(texFb);
			nosEngine.SetPinValue(OutputID, { .Data = texFbBuf.Data(), .Size = texFbBuf.Size() });


			interop.AllocateNVCVImage("Trial", nvcv_NOS_Input.width * upscaleFactor, nvcv_NOS_Input.height * upscaleFactor,
				nvcv_NOS_Input.pixelFormat, nvcv_NOS_Input.componentType,
				nvcv_NOS_Input.bufferBytes * upscaleFactor * upscaleFactor, 
				NVCV_INTERLEAVED, &nvcv_NOS_Output);


			nosResult res2 = interop.NVCVImageToNosTexture(nvcv_NOS_Output, output);


			NvCVImage srcTemp = {}, dstTemp = {};

			//NvCVImage_Alloc(&srcTemp, nvcv_NOS_Input.width, nvcv_NOS_Input.height, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_CUDA, 0);
			//NvCVImage_Alloc(&dstTemp, nvcv_NOS_Input.width * upscaleFactor, nvcv_NOS_Input.height * upscaleFactor, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_CUDA, 0);

			interop.AllocateNVCVImage("SrcTemp", nvcv_NOS_Input.width, nvcv_NOS_Input.height, NVCV_BGR, NVCV_F32, nvcv_NOS_Input.bufferBytes, NVCV_PLANAR, &srcTemp);
			interop.AllocateNVCVImage("DstTemp", nvcv_NOS_Input.width * upscaleFactor, nvcv_NOS_Input.height * upscaleFactor, NVCV_BGR, NVCV_F32, nvcv_NOS_Input.bufferBytes * upscaleFactor * upscaleFactor, NVCV_PLANAR, &dstTemp);

			VFX.InitTransferBuffers(&srcTemp, &dstTemp);

			OutputSizeSet = true;
			return NOS_RESULT_FAILED;
		}


		nosCmd cmd = {};
		nosGPUEvent gpuevent = {};
		nosVulkan->Begin("Input DownloadD", &cmd);
		nosVulkan->Copy(cmd, &in, &InputFormatted, 0);
		nosVulkan->End2(cmd, NOS_TRUE, &gpuevent);
		nosVulkan->WaitGpuEvent(&gpuevent, 0);


		if (res == NOS_RESULT_SUCCESS) {
			cudaError cudaRes;
			res = interop.nosTextureToNVCVImage(InputFormatted, nvcv_NOS_Input);
			VFX.RunSuperResolution(&nvcv_NOS_Input, &nvcv_NOS_Output);

			OutReady = true;

			nosResourceShareInfo out = nos::vkss::DeserializeTextureInfo(args[NSN_Out].Data->Data);
			nosCmd cmd2;
			nosGPUEvent gpuevent2 = {};
			nosVulkan->Begin("NVVFX Upload", &cmd2);
			//nosVulkan->ImageLoad(cmd, hostBuffer.data(), nosVec2u{.x=1920, .y=1080}, NOS_FORMAT_R8G8B8A8_UNORM, &out);
			nosVulkan->Copy(cmd2, &output, &out, 0);
			nosVulkan->End(cmd2, NOS_FALSE);
		}

		return res;
	}

	void ResourceBinder() {

	}

};


void RegisterNVVFX_SuperRes(nosNodeFunctions* outFunctions) {
	NOS_BIND_NODE_CLASS(NSN_NVVFX_SuperRes, NVVFX_SuperRes_NodeContext, outFunctions);
}