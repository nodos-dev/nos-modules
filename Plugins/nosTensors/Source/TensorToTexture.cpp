#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/PluginHelpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include "nosTensorSubsystem/nosTensorSubsystem.h"
#include "nosTensorSubsystem/TensorTypes_generated.h"
#include "flatbuffers/flatbuffers.h"
#include <nosVulkanSubsystem/Helpers.hpp>
#include "TensorsNames.h"

#define CHECK_NOS_RESULT(nosRes) \
	do { \
		nosResult __MACRO__RESULT__= nosRes; \
		if (__MACRO__RESULT__ != NOS_RESULT_SUCCESS) { \
			nosEngine.LogE("Failed from %s %d with error %d.",__FILE__, __LINE__,__MACRO__RESULT__); \
			return NOS_RESULT_FAILED; \
		} \
	} while (0); \

struct TensorToTexture : nos::NodeContext
{
	nosUUID InputUUID, OutputUUID, LayoutUUID;
	nosUUID NodeID;
	nosResourceShareInfo OutputTexture = {}, OutputTextureBuffer = {};
	nosTensorInfo InputTensor = {};
	std::string Layout_NHWC = "NHWC", Layout_NCHW = "NCHW", CurrentLayout;
	int ChannelCountIndex = 3;
	nosCUDABufferInfo TensorCUDABuffer = {};

	TensorToTexture(nos::fb::Node const* node) :NodeContext(node) {
		NodeID = *node->id();
		CreateStringList(LayoutUUID, NodeID, "Layout", { Layout_NHWC,Layout_NCHW });
		CurrentLayout = Layout_NHWC;

		for (const auto& pin : *node->pins()) {
			const char* currentPinName = pin->name()->c_str();
			if (NSN_In.Compare(pin->name()->c_str()) == 0) {
				InputUUID = *pin->id();
			}
			else if (NSN_Out.Compare(pin->name()->c_str()) == 0) {
				OutputUUID = *pin->id();
			}
		}

	}

	~TensorToTexture() {
	}

	void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer value) override{
		if (InputUUID == pinId) {
			auto tensorPin = flatbuffers::GetMutableRoot<nos::sys::tensor::Tensor>(value.Data);
			nosTensorInfo newTensor = {};
			nosTensor->DeserializeTensorPin(&newTensor, tensorPin);
			bool needRefresh = false;
			if (newTensor.CreateInfo.ShapeInfo.DimensionCount == InputTensor.CreateInfo.ShapeInfo.DimensionCount) {
				for (int i = 0; i < newTensor.CreateInfo.ShapeInfo.DimensionCount; i++) {
					if (newTensor.CreateInfo.ShapeInfo.Dimensions[i] != InputTensor.CreateInfo.ShapeInfo.Dimensions[i]) {
						needRefresh = true;
					}
				}
			}
			else {
				needRefresh = true;
			}
			nosTensor->DeserializeTensorPin(&InputTensor, tensorPin);
			if (needRefresh) {
				PrepareResources();
			}
		}
		if (LayoutUUID == pinId) {
			auto layoutStr = std::string(static_cast<char*>(value.Data));
			if (layoutStr.compare(Layout_NCHW) == 0) {
				ChannelCountIndex = 1;
			}
			else if (layoutStr.compare(Layout_NHWC) == 0) {
				ChannelCountIndex = 3;
			}
			if (CurrentLayout != layoutStr) {
				CurrentLayout = layoutStr; 
				PrepareResources();
			}
		}
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{ 
		auto pinIds = nos::GetPinIds(args);
		auto pinValues = nos::GetPinValues(args);

		nosCmd cmd1;
		nosGPUEvent gpuevent1 = {};
		nosCmdEndParams endParams1 = { .ForceSubmit = true, .OutGPUEventHandle = &gpuevent1 };
		nosVulkan->Begin("NVVFX Upload", &cmd1);
		nosVulkan->Copy(cmd1, &OutputTextureBuffer, &OutputTexture, 0);
		nosVulkan->End(cmd1, &endParams1);
		nosVulkan->WaitGpuEvent(&gpuevent1, UINT64_MAX);
		nosCUDABufferInfo CPUData = {};
		nosCUDA->CreateBuffer(&CPUData, TensorCUDABuffer.CreateInfo.AllocationSize);
		nosCUDA->CopyBuffers(&TensorCUDABuffer, &CPUData);

		void* InData = reinterpret_cast<void*>(CPUData.Address);
		uint8_t* vulkanCPUPointer = nosVulkan->Map(&OutputTextureBuffer);
		int a = 5;
		return NOS_RESULT_SUCCESS; 
	}

	
	nosResult PrepareResources() {

		switch (InputTensor.CreateInfo.Location) {
		case TensorMemoryLocation::MEMORY_LOCATION_CPU:
		{
			//TODO: Upload tensor data to VULKAN
			break;
		}
		case TensorMemoryLocation::MEMORY_LOCATION_CUDA:
		{
			nosResult res = nosCUDA->GetCUDABufferFromAddress(InputTensor.MemoryInfo.Address, &TensorCUDABuffer);
			CHECK_NOS_RESULT(res);

			//TODO: Upload tensor data to VULKAN
			OutputTextureBuffer.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
			OutputTextureBuffer.Info.Buffer.Size = TensorCUDABuffer.CreateInfo.AllocationSize;
			OutputTextureBuffer.Info.Buffer.Usage = nosBufferUsage(nosBufferUsage::NOS_BUFFER_USAGE_TRANSFER_SRC | nosBufferUsage::NOS_BUFFER_USAGE_TRANSFER_DST);
			OutputTextureBuffer.Memory.Handle = 0;
			OutputTextureBuffer.Memory.ExternalMemory.HandleType = NOS_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
			OutputTextureBuffer.Memory.ExternalMemory.Handle = TensorCUDABuffer.ShareInfo.ShareableHandle;
			OutputTextureBuffer.Memory.ExternalMemory.PID = getpid();
			OutputTextureBuffer.Memory.ExternalMemory.Offset = TensorCUDABuffer.CreateInfo.Offset;
			OutputTextureBuffer.Memory.ExternalMemory.AllocationSize = TensorCUDABuffer.CreateInfo.AllocationSize;
			OutputTextureBuffer.Memory.Size = TensorCUDABuffer.CreateInfo.AllocationSize;
			res = nosVulkan->ImportResource(&OutputTextureBuffer);
			CHECK_NOS_RESULT(res);
			int width = 0, height = 0;
			if (CurrentLayout == Layout_NCHW) {
				height = InputTensor.CreateInfo.ShapeInfo.Dimensions[2];
				width = InputTensor.CreateInfo.ShapeInfo.Dimensions[3];
			}
			else if (CurrentLayout == Layout_NHWC) {
				height = InputTensor.CreateInfo.ShapeInfo.Dimensions[1];
				width = InputTensor.CreateInfo.ShapeInfo.Dimensions[2];
			}
			
			OutputTexture.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
			OutputTexture.Info.Texture.Format = GetFormatFromTensor(&InputTensor);
			OutputTexture.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_SRC | NOS_IMAGE_USAGE_TRANSFER_DST);
			OutputTexture.Info.Texture.Height = height;
			OutputTexture.Info.Texture.Width = width;

			if (OutputTexture.Memory.Handle != NULL) {
				nosVulkan->DestroyResource(&OutputTexture);
				//OutputTexture.Memory.Handle = NULL;
			}
			nosVulkan->CreateResource(&OutputTexture);

			auto TTexture = nos::vkss::ConvertTextureInfo(OutputTexture);
			flatbuffers::FlatBufferBuilder fbb;
			auto TextureTable = nos::sys::vulkan::Texture::Pack(fbb, &TTexture);
			fbb.Finish(TextureTable);

			nosEngine.SetPinValueDirect(OutputUUID, { .Data = fbb.GetBufferPointer(), .Size = fbb.GetSize() });

			break;
		}
		case TensorMemoryLocation::MEMORY_LOCATION_VULKAN:
		{
			break;
		}
		}
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


	void CreateTextureOutput(nosUUID& GenUUID, nosUUID& NodeUUID, std::string name, nosResourceShareInfo* texture) {

	}
	
	nosFormat GetFormatFromTensor(nosTensorInfo* tensor) {
		switch (tensor->CreateInfo.ShapeInfo.Dimensions[ChannelCountIndex]) {
			case 1:
			{
				switch (tensor->CreateInfo.ElementType) {
					case ELEMENT_TYPE_FLOAT:
					{
						return NOS_FORMAT_R32_SFLOAT;
						break;
					}
					case ELEMENT_TYPE_UINT8:
					{
						return NOS_FORMAT_R8_UINT;
					}
					case ELEMENT_TYPE_INT8:
					{
						return NOS_FORMAT_R8_SRGB;
					}
					case ELEMENT_TYPE_UINT16:
					{
						return NOS_FORMAT_R16_UINT;
					}
					case ELEMENT_TYPE_INT16:
					{
						return NOS_FORMAT_R16_SINT;
					}
					case ELEMENT_TYPE_INT32:
					{
						return NOS_FORMAT_R32_SINT;
					}
					case ELEMENT_TYPE_FLOAT16:
					{
						return NOS_FORMAT_R16_SFLOAT;
					}
					case ELEMENT_TYPE_UINT32: 
					{
						return NOS_FORMAT_R32_UINT;
					}
					case ELEMENT_TYPE_INT64:
					case ELEMENT_TYPE_STRING:
					case ELEMENT_TYPE_BOOL:
					case ELEMENT_TYPE_DOUBLE: //May be considered as float??
					case ELEMENT_TYPE_UINT64:
					case ELEMENT_TYPE_COMPLEX64:
					case ELEMENT_TYPE_COMPLEX128:
					case ELEMENT_TYPE_BFLOAT16:
					case ELEMENT_TYPE_FLOAT8E4M3FN:
					case ELEMENT_TYPE_FLOAT8E4M3FNUZ:
					case ELEMENT_TYPE_FLOAT8E5M2:
					case ELEMENT_TYPE_FLOAT8E5M2FNUZ:
					case ELEMENT_TYPE_UNDEFINED:
					default:
					{
						return NOS_FORMAT_NONE;
					}
				}
			}
			case 2:
			{
				switch (tensor->CreateInfo.ElementType) {
				case ELEMENT_TYPE_FLOAT:
				{
					return NOS_FORMAT_R32G32_SFLOAT;
					break;
				}
				case ELEMENT_TYPE_UINT8:
				{
					return NOS_FORMAT_R8G8_UINT;
				}
				case ELEMENT_TYPE_INT8:
				{
					return NOS_FORMAT_R8G8_SRGB;
				}
				case ELEMENT_TYPE_UINT16:
				{
					return NOS_FORMAT_R16G16_UINT;
				}
				case ELEMENT_TYPE_INT16:
				{
					return NOS_FORMAT_R16G16_SINT;
				}
				case ELEMENT_TYPE_INT32:
				{
					return NOS_FORMAT_R32G32_SINT;
				}
				case ELEMENT_TYPE_FLOAT16:
				{
					return NOS_FORMAT_R16G16_SFLOAT;
				}
				case ELEMENT_TYPE_UINT32:
				{
					return NOS_FORMAT_R32G32_UINT;
				}
				case ELEMENT_TYPE_INT64:
				case ELEMENT_TYPE_STRING:
				case ELEMENT_TYPE_BOOL:
				case ELEMENT_TYPE_DOUBLE: //May be considered as float??
				case ELEMENT_TYPE_UINT64:
				case ELEMENT_TYPE_COMPLEX64:
				case ELEMENT_TYPE_COMPLEX128:
				case ELEMENT_TYPE_BFLOAT16:
				case ELEMENT_TYPE_FLOAT8E4M3FN:
				case ELEMENT_TYPE_FLOAT8E4M3FNUZ:
				case ELEMENT_TYPE_FLOAT8E5M2:
				case ELEMENT_TYPE_FLOAT8E5M2FNUZ:
				case ELEMENT_TYPE_UNDEFINED:
				default:
				{
					return NOS_FORMAT_NONE;
				}
				}
			}
			case 3:
			{
				switch (tensor->CreateInfo.ElementType) {
				case ELEMENT_TYPE_FLOAT:
				{
					return NOS_FORMAT_R32G32B32_SFLOAT;
					break;
				}
				case ELEMENT_TYPE_UINT8:
				{
					return NOS_FORMAT_R8G8B8_SRGB;
				}
				case ELEMENT_TYPE_INT8:
				{
					return NOS_FORMAT_R8G8B8_SRGB;
				}
				case ELEMENT_TYPE_UINT16:
				{
					return NOS_FORMAT_R16G16B16_UINT;
				}
				case ELEMENT_TYPE_INT16:
				{
					return NOS_FORMAT_R16G16B16_SINT;
				}
				case ELEMENT_TYPE_INT32:
				{
					return NOS_FORMAT_R32G32B32_SINT;
				}
				case ELEMENT_TYPE_FLOAT16:
				{
					return NOS_FORMAT_R16G16B16_SFLOAT;
				}
				case ELEMENT_TYPE_UINT32:
				{
					return NOS_FORMAT_R32G32B32_UINT;
				}
				case ELEMENT_TYPE_INT64:
				case ELEMENT_TYPE_STRING:
				case ELEMENT_TYPE_BOOL:
				case ELEMENT_TYPE_DOUBLE: //May be considered as float??
				case ELEMENT_TYPE_UINT64:
				case ELEMENT_TYPE_COMPLEX64:
				case ELEMENT_TYPE_COMPLEX128:
				case ELEMENT_TYPE_BFLOAT16:
				case ELEMENT_TYPE_FLOAT8E4M3FN:
				case ELEMENT_TYPE_FLOAT8E4M3FNUZ:
				case ELEMENT_TYPE_FLOAT8E5M2:
				case ELEMENT_TYPE_FLOAT8E5M2FNUZ:
				case ELEMENT_TYPE_UNDEFINED:
				default:
				{
					return NOS_FORMAT_NONE;
				}
				}
			}
			case 4: 
			{
				switch (tensor->CreateInfo.ElementType) {
				case ELEMENT_TYPE_FLOAT:
				{
					return NOS_FORMAT_R32G32B32A32_SFLOAT;
					break;
				}
				case ELEMENT_TYPE_UINT8:
				{
					return NOS_FORMAT_R8G8B8A8_SRGB;
				}
				case ELEMENT_TYPE_INT8:
				{
					return NOS_FORMAT_R8G8B8A8_SRGB;
				}
				case ELEMENT_TYPE_UINT16:
				{
					return NOS_FORMAT_R16G16B16A16_UINT;
				}
				case ELEMENT_TYPE_INT16:
				{
					return NOS_FORMAT_R16G16B16A16_SINT;
				}
				case ELEMENT_TYPE_INT32:
				{
					return NOS_FORMAT_R32G32B32A32_SINT;
				}
				case ELEMENT_TYPE_FLOAT16:
				{
					return NOS_FORMAT_R16G16B16A16_SFLOAT;
				}
				case ELEMENT_TYPE_UINT32:
				{
					return NOS_FORMAT_R32G32B32A32_UINT;
				}
				case ELEMENT_TYPE_INT64:
				case ELEMENT_TYPE_STRING:
				case ELEMENT_TYPE_BOOL:
				case ELEMENT_TYPE_DOUBLE: //May be considered as float??
				case ELEMENT_TYPE_UINT64:
				case ELEMENT_TYPE_COMPLEX64:
				case ELEMENT_TYPE_COMPLEX128:
				case ELEMENT_TYPE_BFLOAT16:
				case ELEMENT_TYPE_FLOAT8E4M3FN:
				case ELEMENT_TYPE_FLOAT8E4M3FNUZ:
				case ELEMENT_TYPE_FLOAT8E5M2:
				case ELEMENT_TYPE_FLOAT8E5M2FNUZ:
				case ELEMENT_TYPE_UNDEFINED:
				default:
				{
					return NOS_FORMAT_NONE;
				}
				}
			}
			}
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

void RegisterTensorToTexture(nosNodeFunctions* outFunctions) {
	NOS_BIND_NODE_CLASS(NSN_TensorToTexture, TensorToTexture, outFunctions);
}