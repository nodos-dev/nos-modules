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

struct UUIDGenerator {
	UUIDGenerator() = default;

	uuids::uuid_random_generator Generate() {
		std::random_device rd;
		auto seed_data = std::array<int, std::mt19937::state_size>{};
		std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));

		std::seed_seq seed = std::seed_seq(std::begin(seed_data), std::end(seed_data));
		std::mt19937 mtengine(seed);
		uuids::uuid_random_generator generator(mtengine);
		return generator;
	}
};

// nosNodes

struct NVVFX_SuperRes_NodeContext : nos::NodeContext {

	NVVFXAppRunner VFX;

	nos::fb::UUID NodeID, InputID, OutputID, UpscaleFactorID;
	nosResourceShareInfo InputFormatted = {}, InputBuffer = {}, output = {};
	nosResourceShareInfo OutputPre = {};

	std::atomic_bool OutReady = false;
	CUDAVulkanInterop interop;
	int lastWidth, lastHeight;

	NvCVImage nvcv_NV_Input;
	NvCVImage nvcv_NV_Output;
	NvCVImage nvcv_NOS_Input;
	NvCVImage nvcv_NOS_Output;

	float UpscaleFactor;
	float UpscaleStrength;
	std::filesystem::path ModelsPath;
	bool OutputSizeSet = false;

	NVVFX_SuperRes_NodeContext(nos::fb::Node const* node) :NodeContext(node) {
		nosEngine.RequestSubsystem(NOS_NAME_STATIC(NOS_VULKAN_SUBSYSTEM_NAME), 1, 0, (void**)&nosVulkan);
		NodeID = *node->id();
		for (const auto& pin : *node->pins()) {
			if (NSN_UpscaleFactor.Compare(pin->name()->c_str()) == 0) {
				//UpscaleFactor = std::stof((const char*)(pin->data()->data()));
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
		CreateUpscaleFactorList({ "4/3x", "1.5x", "2x", "3x", "4x" });
	}


	void  OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer value) override {
		if (pinName == NSN_In) {
			nosResourceShareInfo input = nos::vkss::DeserializeTextureInfo(value.Data);
			lastWidth = input.Info.Texture.Width;
			lastHeight = input.Info.Texture.Height;
		}
		if (pinName == NSN_UpscaleStrength) {
			UpscaleStrength = *static_cast<float*>(value.Data);
		}
		if (pinName == NSN_UpscaleFactor) {
			UpscaleFactor = std::stof(static_cast<const char*>(value.Data));
		}
		if (pinName == NSN_ModelsPath) {
			ModelsPath = static_cast<const char*>(value.Data);
		}
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* execArgs) {
		nos::NodeExecuteArgs args(execArgs);
		nosResult res = NOS_RESULT_SUCCESS;
		nosResourceShareInfo in = nos::vkss::DeserializeTextureInfo(args[NSN_In].Data->Data);

		res = PrepareResources(in);

		if (res == NOS_RESULT_SUCCESS) {
			nosCmd cmd = {};
			nosGPUEvent gpuevent = {};
			nosVulkan->Begin("Input DownloadD", &cmd);
			nosVulkan->Copy(cmd, &in, &InputFormatted, 0);
			nosVulkan->End2(cmd, NOS_TRUE, &gpuevent);
			nosVulkan->WaitGpuEvent(&gpuevent, 0);

			res = interop.nosTextureToNVCVImage(InputFormatted, nvcv_NOS_Input);
			if (res != NOS_RESULT_SUCCESS)
				return res;

			res = VFX.RunSuperResolution(&nvcv_NOS_Input, &nvcv_NOS_Output);
			if (res != NOS_RESULT_SUCCESS)
				return res;

			nosResourceShareInfo out = nos::vkss::DeserializeTextureInfo(args[NSN_Out].Data->Data);
			nosCmd cmd2;
			nosGPUEvent gpuevent2 = {};
			nosVulkan->Begin("NVVFX Upload", &cmd2);
			nosVulkan->Copy(cmd2, &output, &out, 0);
			nosVulkan->End(cmd2, NOS_FALSE);
		}

		return res;
	}

	nosResult PrepareResources(nosResourceShareInfo& in) {
		if (in.Info.Texture.Width == InputFormatted.Info.Texture.Width &&
			in.Info.Texture.Height == InputFormatted.Info.Texture.Height &&
			nvcv_NOS_Output.width == in.Info.Texture.Width * UpscaleFactor) {
			//No change
			return NOS_RESULT_SUCCESS;
		}
		
		nosResult res = NOS_RESULT_SUCCESS;

		if (InputFormatted.Memory.Handle != NULL) {
			nosVulkan->DestroyResource(&InputFormatted);
		}
		//if (output.Memory.Handle != NULL) {
		//	nosVulkan->DestroyResource(&output);
		//}

		InputFormatted.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
		InputFormatted.Info.Texture.Width = in.Info.Texture.Width;
		InputFormatted.Info.Texture.Height = in.Info.Texture.Height;
		InputFormatted.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_DST | NOS_IMAGE_USAGE_TRANSFER_SRC);
		InputFormatted.Info.Texture.Format = NOS_FORMAT_R32G32B32A32_SFLOAT;
		nosVulkan->CreateResource(&InputFormatted);

		res = interop.nosTextureToNVCVImage(InputFormatted, nvcv_NOS_Input);

		nosResourceShareInfo dummy = {};
		dummy.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
		dummy.Info.Texture.Width = InputFormatted.Info.Texture.Width * UpscaleFactor;
		dummy.Info.Texture.Height = InputFormatted.Info.Texture.Height * UpscaleFactor;
		dummy.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_DST | NOS_IMAGE_USAGE_TRANSFER_SRC);
		dummy.Info.Texture.Format = InputFormatted.Info.Texture.Format;
		dummy.Memory.ExternalMemory.HandleType = NOS_EXTERNAL_MEMORY_HANDLE_TYPE_WIN32;

		auto texFb = nos::vkss::ConvertTextureInfo(dummy);
		texFb.unscaled = true;
		auto texFbBuf = nos::Buffer::From(texFb);
		nosEngine.SetPinValue(OutputID, { .Data = texFbBuf.Data(), .Size = texFbBuf.Size() });

		res = interop.AllocateNVCVImage("Trial", nvcv_NOS_Input.width * UpscaleFactor, nvcv_NOS_Input.height * UpscaleFactor,
			nvcv_NOS_Input.pixelFormat, nvcv_NOS_Input.componentType,
			nvcv_NOS_Input.bufferBytes * UpscaleFactor * UpscaleFactor,
			NVCV_INTERLEAVED, &nvcv_NOS_Output);

		res = interop.NVCVImageToNosTexture(nvcv_NOS_Output, output);

		NvCVImage srcTemp = {}, dstTemp = {};
		res = interop.AllocateNVCVImage("SrcTemp", nvcv_NOS_Input.width, nvcv_NOS_Input.height, NVCV_BGR, NVCV_F32, nvcv_NOS_Input.bufferBytes, NVCV_PLANAR, &srcTemp);
		res = interop.AllocateNVCVImage("DstTemp", nvcv_NOS_Input.width * UpscaleFactor, nvcv_NOS_Input.height * UpscaleFactor, NVCV_BGR, NVCV_F32, 
			nvcv_NOS_Input.bufferBytes * UpscaleFactor * UpscaleFactor, NVCV_PLANAR, &dstTemp);

		res = VFX.InitTransferBuffers(&srcTemp, &dstTemp);

		OutputSizeSet = true;
		return res;
	}

	void CreateUpscaleFactorList(std::vector<std::string> list) {
		flatbuffers::FlatBufferBuilder fbb;
		flatbuffers::FlatBufferBuilder fbb2;
		std::vector<flatbuffers::Offset<nos::fb::Pin>> UpscaleFactorPin;
		nos::fb::TVisualizer vis = { .type = nos::fb::VisualizerType::COMBO_BOX, .name = NSN_UpscaleFactor.AsString() };
		UUIDGenerator generator;
		auto buf = std::vector<u8>((u8*)list.front().data(), (u8*)list.front().data() + list.front().size() + 1);

		UpscaleFactorID = *(nosUUID*)generator.Generate()().as_bytes().data();
		UpscaleFactorPin.push_back(nos::fb::CreatePinDirect(fbb,
			&UpscaleFactorID,
			NSN_UpscaleFactor.AsCStr(),
			"string",
			nos::fb::ShowAs::PROPERTY,
			nos::fb::CanShowAs::PROPERTY_ONLY,
			0,
			nos::fb::Visualizer::Pack(fbb, &vis),
			&buf));

		HandleEvent(nos::CreateAppEvent(fbb,
			nos::CreatePartialNodeUpdateDirect(fbb, &NodeID, nos::ClearFlags::NONE, 0, &UpscaleFactorPin)));

		HandleEvent(nos::CreateAppEvent(
			fbb2, nos::app::CreateUpdateStringList(fbb2, nos::fb::CreateStringList(fbb2, fbb2.CreateString(NSN_UpscaleFactor.AsString()), fbb2.CreateVectorOfStrings(list)))));
		UpscaleFactor = 4.0f / 3.0f;
	}

};


void RegisterNVVFX_SuperRes(nosNodeFunctions* outFunctions) {
	NOS_BIND_NODE_CLASS(NSN_NVVFX_SuperRes, NVVFX_SuperRes_NodeContext, outFunctions);
}