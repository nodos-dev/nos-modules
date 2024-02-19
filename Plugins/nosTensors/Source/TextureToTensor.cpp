#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/PluginHelpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include "nosTensorSubsystem/nosTensorSubsystem.h"
#include "nosTensorSubsystem/TensorTypes_generated.h"
#include "flatbuffers/flatbuffers.h"
#include <nosVulkanSubsystem/Helpers.hpp>
#include "CUDAKernels/RGBAtoRGBAPlanar.cu.ptx_generated.h"

#define CHECK_NOS_RESULT(nosRes) \
	do { \
		nosResult __MACRO__RESULT__= nosRes; \
		if (__MACRO__RESULT__ != NOS_RESULT_SUCCESS) { \
			nosEngine.LogE("Failed from %s %d with error %d.",__FILE__, __LINE__,__MACRO__RESULT__); \
			return NOS_RESULT_FAILED; \
		} \
	} while (0); \


NOS_REGISTER_NAME(TextureToTensor);
NOS_REGISTER_NAME(Layot)
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME(Out);

struct TextureToTensor : nos::NodeContext
{
	nosUUID LayoutUUID, OutFormatUUID;
	nosUUID InputUUID, OutputUUID;
	nosUUID NodeID;
	nosTensorInfo OutTensor = {};
	nosCUDABufferInfo    InputLinearCUDA = {}, InputLinearPlanarCUDA = {};
	nosResourceShareInfo InputTexture = {}, InputTextureLinear = {};
	std::string Layout_NHWC = "NHWC", Layout_NCHW = "NCHW";
	std::string OutFormat_RGBA = "RGBA", OutFormat_RGB = "RGB", OutFormat_RG = "RG", OutFormat_R = "R";
	std::string CurrentLayout;
	int InputFormatChannelCount, OutputFormatChannelCount;
	nosCUDAModule CUDAModule = {};
	nosCUDAKernelFunction CUDAFunction = {};
	TextureToTensor(nos::fb::Node const* node) :NodeContext(node) {
		NodeID = *node->id();
		
		CreateStringList(LayoutUUID, NodeID, "Layout", { Layout_NHWC,Layout_NCHW });
		CreateStringList(OutFormatUUID, NodeID, "Output Format", { OutFormat_RGBA, OutFormat_RGB, OutFormat_RG, OutFormat_R });  
		OutputFormatChannelCount = 4;

		char ErrorLog[8192];
		nosCUDAModule cudaModule = {};
		nosCUDAKernelFunction function = {};
		nosCUDA->LoadKernelModuleFromCString(RGBAtoRGBAPlanar, ErrorLog, 8192, &CUDAModule);
		nosCUDA->GetModuleKernelFunction(RGBAtoRGBAPlanar_NAME, CUDAModule, &CUDAFunction);
		CurrentLayout = Layout_NHWC;
		
	}

	~TextureToTensor() {
	}

	void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer value) override{
		if (NSN_In.Compare(pinName.AsCStr()) == 0) {
			auto inputTex = nos::vkss::DeserializeTextureInfo(value.Data);
			UpdateResources(inputTex);
		}
		else if (LayoutUUID == pinId) {
			auto layoutStr = std::string(static_cast<char*>(value.Data));
			CurrentLayout = layoutStr;

			if (layoutStr == Layout_NHWC) {
				UpdateTensors();
			}
			else if (layoutStr == Layout_NCHW) {
				UpdateTensors();
			}
		}
		else if (OutFormatUUID == pinId) {
			auto outFormatStr = std::string(static_cast<char*>(value.Data));
			if (outFormatStr == OutFormat_RGBA) {
				OutputFormatChannelCount = 4;
			}
			else if (outFormatStr == OutFormat_RGB) {
				OutputFormatChannelCount = 3;
			}
			else if (outFormatStr == OutFormat_RG) {
				OutputFormatChannelCount = 2;
			}
			else if (outFormatStr == OutFormat_R) {
				OutputFormatChannelCount = 1;
			}
			UpdateTensors();
		}
	}

	void LaunchConverterKernel() {
		
		nosCUDABufferInfo CPUData = {};
		nosCUDA->CreateBuffer(&CPUData, InputLinearCUDA.CreateInfo.AllocationSize);
		nosCUDA->CopyBuffers(&InputLinearCUDA, &CPUData);
		uint16_t* CPUDataPointer = reinterpret_cast<uint16_t*>(CPUData.Address);
		uint8_t* vulkanCPUPointer = nosVulkan->Map(&InputTextureLinear);

		void* InData = reinterpret_cast<void*>(InputLinearCUDA.Address);
		int TotalSizeInBytes = InputLinearCUDA.CreateInfo.AllocationSize;
		int BytesPerElement = GetComponentBytesFromVulkanFormat(InputTexture.Info.Texture.Format);
		int Width = InputTexture.Info.Texture.Width;
		int Height = InputTexture.Info.Texture.Height;
		void* OutData = reinterpret_cast<void*>(OutTensor.MemoryInfo.Address);
		void* args[] = { &InData, &TotalSizeInBytes, &BytesPerElement, &Width, &Height, &InputFormatChannelCount ,&OutData };
		nosCUDAStream stream = {};
		nosCUDA->CreateStream(&stream);

		nosCUDAKernelLaunchConfig config = { .GridDimensions = {.x = (InputTexture.Info.Texture.Width * InputTexture.Info.Texture.Height / MAX_THREAD_PER_BLOCK ) + 1, .y = 1, .z = 1 }, 
											 .BlockDimensions = {.x = MAX_THREAD_PER_BLOCK, .y = 1, .z = 1},
											 .DynamicMemorySize = 0 };
		
		nosCUDA->LaunchModuleKernelFunction(stream, CUDAFunction, config, args, nullptr, nullptr);
		nosCUDA->WaitStream(stream);

		nosCUDA->CopyBuffers(&InputLinearPlanarCUDA, &CPUData);

		nosCUDAError lastError = nosCUDA->GetLastError();
	}

	nosResult UpdateResources(nosResourceShareInfo& in) {
		if (in.Info.Texture.Width == InputTexture.Info.Texture.Width &&
			in.Info.Texture.Height == InputTexture.Info.Texture.Height &&
			in.Info.Texture.Format == InputTexture.Info.Texture.Format) {
			//No change, update the buffer
			nosCmd texToBuf = {};
			nosGPUEvent waitTexToBuf = {};
			nosCmdEndParams endParams = { .ForceSubmit = true, .OutGPUEventHandle = &waitTexToBuf };
			nosVulkan->Begin("TexToBuf", &texToBuf);
			nosVulkan->Copy(texToBuf, &in, &InputTextureLinear, 0);
			nosVulkan->End(texToBuf, &endParams);
			nosVulkan->WaitGpuEvent(&waitTexToBuf, UINT64_MAX);
			if (CurrentLayout == Layout_NCHW) {
				LaunchConverterKernel();
			}
			return NOS_RESULT_SUCCESS;
		}

		nosResult res = NOS_RESULT_SUCCESS;

		if (InputTexture.Memory.Handle != NULL) {
			nosVulkan->DestroyResource(&InputTexture);
		}
		if (InputTextureLinear.Memory.Handle != NULL) {
			if (InputLinearCUDA.Address != NULL) {
				nosCUDA->DestroyBuffer(&InputLinearCUDA);
			}
			nosVulkan->DestroyResource(&InputTextureLinear);
		}

		InputFormatChannelCount = GetComponentNumFromVulkanFormat(in.Info.Texture.Format);
		int componentByte = GetComponentBytesFromVulkanFormat(in.Info.Texture.Format);

		InputTextureLinear.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
		InputTextureLinear.Info.Buffer.Size = in.Info.Texture.Width * in.Info.Texture.Height * componentByte * InputFormatChannelCount;
		InputTextureLinear.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_TRANSFER_DST);
		nosVulkan->CreateResource(&InputTextureLinear);

		nosCmd texToBuf = {};
		nosGPUEvent waitTexToBuf = {};
		nosCmdEndParams endParams = { .ForceSubmit = true, .OutGPUEventHandle = &waitTexToBuf };
		nosVulkan->Begin("TexToBuf", &texToBuf);
		nosVulkan->Copy(texToBuf, &in, &InputTextureLinear, 0);
		nosVulkan->End(texToBuf, &endParams);
		nosVulkan->WaitGpuEvent(&waitTexToBuf, UINT64_MAX);
		
		nosTensorCreateInfo tensorCreateInfo = {};
		tensorCreateInfo.Location = MEMORY_LOCATION_VULKAN;
		tensorCreateInfo.ShapeInfo.DimensionCount = 4;
		int64_t tensorDimensions[] = { 1, in.Info.Texture.Height, in.Info.Texture.Width, OutputFormatChannelCount }; //NHWC
		tensorCreateInfo.ShapeInfo.Dimensions = tensorDimensions;

		res = nosCUDA->ImportExternalMemoryAsCUDABuffer(InputTextureLinear.Memory.ExternalMemory.Handle, InputTextureLinear.Memory.ExternalMemory.AllocationSize,
			InputTextureLinear.Memory.Size, InputTextureLinear.Memory.ExternalMemory.Offset, EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUEWIN32, &InputLinearCUDA);
		CHECK_NOS_RESULT(res);
		TensorElementType elementType = {};
		nosTensor->GetTensorElementTypeFromVulkanResource(&elementType, &in);

		if (InputLinearPlanarCUDA.Address != NULL) {
			nosCUDA->DestroyBuffer(&InputLinearPlanarCUDA);
		}

		res = nosCUDA->CreateBufferOnCUDA(&InputLinearPlanarCUDA, InputLinearCUDA.CreateInfo.AllocationSize);
		CHECK_NOS_RESULT(res);
		
		InputTexture = in;

		UpdateTensors();
		return res;
	}

	nosResult UpdateTensors() {
		
		TensorElementType elementType = {};
		nosTensor->GetTensorElementTypeFromVulkanResource(&elementType, &InputTexture);

		nosTensorCreateInfo planarCreateInfo = {};
		planarCreateInfo.ShapeInfo.DimensionCount = 4; //this is fixed since either NHWC or NCHW
		nosResult res;
		if (CurrentLayout == Layout_NHWC) {
			int64_t planarTensorDimensions[] = { 1, InputTexture.Info.Texture.Height, InputTexture.Info.Texture.Width, OutputFormatChannelCount };
			planarCreateInfo.ShapeInfo.Dimensions = planarTensorDimensions;
			res = nosTensor->ImportTensorFromCUDABuffer(&OutTensor, &InputLinearCUDA, planarCreateInfo.ShapeInfo, elementType);
		}
		else { //NCHW
			int64_t planarTensorDimensions[] = { 1, OutputFormatChannelCount, InputTexture.Info.Texture.Height, InputTexture.Info.Texture.Width };
			planarCreateInfo.ShapeInfo.Dimensions = planarTensorDimensions;
			res = nosTensor->ImportTensorFromCUDABuffer(&OutTensor, &InputLinearPlanarCUDA, planarCreateInfo.ShapeInfo, elementType);
			LaunchConverterKernel();
		}

		CHECK_NOS_RESULT(res);
		TensorPinConfig pinConfig = {};
		pinConfig.CanShowAs = TENSOR_CAN_SHOW_AS_OUTPUT_PIN_OR_PROPERTY;
		pinConfig.ShowAs = TENSOR_SHOW_AS_OUTPUT_PIN;
		pinConfig.Name = "Output Tensor";
		nosTensor->UpdateTensorPin(&OutTensor, &NodeID, &OutputUUID, pinConfig);
	}

	static nosResult GetFunctions(size_t* count, nosName* names, nosPfnNodeFunctionExecute* fns)
	{
		*count = 0;
		if (!names || !fns)
			return NOS_RESULT_SUCCESS;
		return NOS_RESULT_SUCCESS;
	}

	void CreateStringList(nosUUID& GenUUID, nosUUID& NodeUUID, std::string name,std::vector<std::string> list) {
		flatbuffers::FlatBufferBuilder fbb;
		flatbuffers::FlatBufferBuilder fbb2;
		std::vector<flatbuffers::Offset<nos::fb::Pin>> StrListPin;
		nos::fb::TVisualizer vis = { .type = nos::fb::VisualizerType::COMBO_BOX, .name = name };
		auto buf = std::vector<u8>((u8*)list.front().data(), (u8*)list.front().data() + list.front().size() + 1);

		nosEngine.GenerateID(&GenUUID);

		StrListPin.push_back(nos::fb::CreatePinDirect(fbb,
			&GenUUID,
			name.c_str(),
			"string",
			nos::fb::ShowAs::PROPERTY,
			nos::fb::CanShowAs::PROPERTY_ONLY,
			0,
			nos::fb::Visualizer::Pack(fbb, &vis),
			&buf));

		HandleEvent(nos::CreateAppEvent(fbb,
			nos::CreatePartialNodeUpdateDirect(fbb, &NodeUUID, nos::ClearFlags::NONE, 0, &StrListPin)));

		HandleEvent(nos::CreateAppEvent(
			fbb2, nos::app::CreateUpdateStringList(fbb2, nos::fb::CreateStringList(fbb2, fbb2.CreateString(name), fbb2.CreateVectorOfStrings(list)))));
	}

	int GetComponentBytesFromVulkanFormat(nosFormat format)
	{
		switch (format) {
		case NOS_FORMAT_R8_UNORM:
		case NOS_FORMAT_R8G8_UNORM:
		case NOS_FORMAT_R8G8B8_UNORM:
		case NOS_FORMAT_B8G8R8_UNORM:
		case NOS_FORMAT_R8G8B8A8_UNORM:
		case NOS_FORMAT_B8G8R8A8_UNORM:
		case NOS_FORMAT_G8B8G8R8_422_UNORM:
		case NOS_FORMAT_B8G8R8G8_422_UNORM:
		case NOS_FORMAT_R8_UINT:
		case NOS_FORMAT_R8G8_UINT:
		case NOS_FORMAT_B8G8R8_UINT:
		case NOS_FORMAT_R8G8B8A8_UINT:
		case NOS_FORMAT_R8_SRGB:
		case NOS_FORMAT_R8G8_SRGB:
		case NOS_FORMAT_R8G8B8_SRGB:
		case NOS_FORMAT_B8G8R8_SRGB:
		case NOS_FORMAT_R8G8B8A8_SRGB:
		case NOS_FORMAT_B8G8R8A8_SRGB:
			return 1;

		case NOS_FORMAT_R16_UNORM:
		case NOS_FORMAT_R16G16_UNORM:
		case NOS_FORMAT_R16G16B16_UNORM:
		case NOS_FORMAT_R16G16B16A16_UNORM:
		case NOS_FORMAT_D16_UNORM:
		case NOS_FORMAT_R16_UINT:
		case NOS_FORMAT_R16G16B16_UINT:
		case NOS_FORMAT_R16G16_UINT:
		case NOS_FORMAT_R16G16B16A16_UINT:
		case NOS_FORMAT_R32_UINT:
		case NOS_FORMAT_R16_USCALED:
		case NOS_FORMAT_R16G16_USCALED:
		case NOS_FORMAT_R16G16B16_USCALED:
		case NOS_FORMAT_R16G16B16A16_USCALED:
			return 2;

		case NOS_FORMAT_R32G32_UINT:
		case NOS_FORMAT_R32G32B32_UINT:
		case NOS_FORMAT_R32G32B32A32_UINT:
		case NOS_FORMAT_A2R10G10B10_UINT_PACK32:
		case NOS_FORMAT_A2R10G10B10_UNORM_PACK32:
		case NOS_FORMAT_A2R10G10B10_USCALED_PACK32:
		case NOS_FORMAT_X8_D24_UNORM_PACK32:
			return 4;

		case NOS_FORMAT_R16_SINT:
		case NOS_FORMAT_R16G16_SINT:
		case NOS_FORMAT_R16G16B16_SINT:
		case NOS_FORMAT_R16G16B16A16_SINT:
		case NOS_FORMAT_R16_SNORM:
		case NOS_FORMAT_R16G16_SNORM:
		case NOS_FORMAT_R16G16B16_SNORM:
		case NOS_FORMAT_R16G16B16A16_SNORM:
		case NOS_FORMAT_R16_SSCALED:
		case NOS_FORMAT_R16G16_SSCALED:
		case NOS_FORMAT_R16G16B16_SSCALED:
		case NOS_FORMAT_R16G16B16A16_SSCALED:
			return 4;

		case NOS_FORMAT_R16_SFLOAT:
		case NOS_FORMAT_R16G16_SFLOAT:
		case NOS_FORMAT_R16G16B16_SFLOAT:
		case NOS_FORMAT_R16G16B16A16_SFLOAT:
			return 2;

		case NOS_FORMAT_A2R10G10B10_SNORM_PACK32:
		case NOS_FORMAT_A2R10G10B10_SINT_PACK32:
		case NOS_FORMAT_A2R10G10B10_SSCALED_PACK32:
		case NOS_FORMAT_R32_SINT:
		case NOS_FORMAT_R32G32_SINT:
		case NOS_FORMAT_R32G32B32_SINT:
		case NOS_FORMAT_R32G32B32A32_SINT:
			return 4;

		case NOS_FORMAT_R32_SFLOAT:
		case NOS_FORMAT_R32G32_SFLOAT:
		case NOS_FORMAT_R32G32B32_SFLOAT:
		case NOS_FORMAT_R32G32B32A32_SFLOAT:
		case NOS_FORMAT_B10G11R11_UFLOAT_PACK32:
		case NOS_FORMAT_D32_SFLOAT:
			return 4;

			//Cant be mapped for now
			//return NvCVImage_ComponentType::NVCV_U64;
			//return NvCVImage_ComponentType::NVCV_S64;
			//return NvCVImage_ComponentType::NVCV_F64;

		}
		return 0;
	}
	int GetComponentNumFromVulkanFormat(nosFormat format)
	{
		switch (format) {
		case NOS_FORMAT_NONE:
		case NOS_FORMAT_R8_UNORM:
		case NOS_FORMAT_R8_UINT:
		case NOS_FORMAT_R8_SRGB:
		case NOS_FORMAT_R16_UNORM:
		case NOS_FORMAT_R16_SNORM:
		case NOS_FORMAT_R16_USCALED:
		case NOS_FORMAT_R16_SSCALED:
		case NOS_FORMAT_R16_UINT:
		case NOS_FORMAT_R16_SINT:
		case NOS_FORMAT_R16_SFLOAT:
		case NOS_FORMAT_R32_UINT:
		case NOS_FORMAT_R32_SINT:
		case NOS_FORMAT_R32_SFLOAT:
		case NOS_FORMAT_D16_UNORM:
		case NOS_FORMAT_D32_SFLOAT:
			return 1;
		case NOS_FORMAT_R8G8_UNORM:
		case NOS_FORMAT_R8G8_UINT:
		case NOS_FORMAT_R8G8_SRGB:
		case NOS_FORMAT_R16G16_UNORM:
		case NOS_FORMAT_R16G16_SNORM:
		case NOS_FORMAT_R16G16_USCALED:
		case NOS_FORMAT_R16G16_SSCALED:
		case NOS_FORMAT_R16G16_UINT:
		case NOS_FORMAT_R16G16_SINT:
		case NOS_FORMAT_R16G16_SFLOAT:
		case NOS_FORMAT_R32G32_UINT:
		case NOS_FORMAT_R32G32_SINT:
		case NOS_FORMAT_R32G32_SFLOAT:
			return 2;
		case NOS_FORMAT_R8G8B8_UNORM:
		case NOS_FORMAT_R8G8B8_SRGB:
		case NOS_FORMAT_B8G8R8_UNORM:
		case NOS_FORMAT_B8G8R8_UINT:
		case NOS_FORMAT_B8G8R8_SRGB:
		case NOS_FORMAT_R16G16B16_UNORM:
		case NOS_FORMAT_R16G16B16_SNORM:
		case NOS_FORMAT_R16G16B16_USCALED:
		case NOS_FORMAT_R16G16B16_SSCALED:
		case NOS_FORMAT_R16G16B16_UINT:
		case NOS_FORMAT_R16G16B16_SINT:
		case NOS_FORMAT_R16G16B16_SFLOAT:
		case NOS_FORMAT_R32G32B32_UINT:
		case NOS_FORMAT_R32G32B32_SINT:
		case NOS_FORMAT_R32G32B32_SFLOAT:
		case NOS_FORMAT_B10G11R11_UFLOAT_PACK32:
		case NOS_FORMAT_G8B8G8R8_422_UNORM:
		case NOS_FORMAT_B8G8R8G8_422_UNORM:
			return 3;
		case NOS_FORMAT_R8G8B8A8_UNORM:
		case NOS_FORMAT_R8G8B8A8_UINT:
		case NOS_FORMAT_R8G8B8A8_SRGB:
		case NOS_FORMAT_B8G8R8A8_UNORM:
		case NOS_FORMAT_B8G8R8A8_SRGB:
		case NOS_FORMAT_R16G16B16A16_UNORM:
		case NOS_FORMAT_R16G16B16A16_SNORM:
		case NOS_FORMAT_R16G16B16A16_USCALED:
		case NOS_FORMAT_R16G16B16A16_SSCALED:
		case NOS_FORMAT_R16G16B16A16_UINT:
		case NOS_FORMAT_R16G16B16A16_SINT:
		case NOS_FORMAT_R16G16B16A16_SFLOAT:
		case NOS_FORMAT_R32G32B32A32_UINT:
		case NOS_FORMAT_R32G32B32A32_SINT:
		case NOS_FORMAT_R32G32B32A32_SFLOAT:
		case NOS_FORMAT_A2R10G10B10_UNORM_PACK32:
		case NOS_FORMAT_A2R10G10B10_SNORM_PACK32:
		case NOS_FORMAT_A2R10G10B10_USCALED_PACK32:
		case NOS_FORMAT_A2R10G10B10_SSCALED_PACK32:
		case NOS_FORMAT_A2R10G10B10_UINT_PACK32:
		case NOS_FORMAT_A2R10G10B10_SINT_PACK32:
			return 4;
		default:
			return 0;
		}
	}
};

void RegisterTextureToTensor(nosNodeFunctions* outFunctions) {
	NOS_BIND_NODE_CLASS(NSN_TextureToTensor, TextureToTensor, outFunctions);
}