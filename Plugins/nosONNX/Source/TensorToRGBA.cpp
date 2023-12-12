#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include "ONNXRTCommon.h"
#include "Tensor.h"

NOS_REGISTER_NAME(TensorToRGBA);
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME(SliceFrom);

struct TensorToRGBANodeContext : nos::NodeContext {

	nosResourceShareInfo OutputTexture;
	nos::fb::TTensor InputTensorProxy;
	nos::fb::UUID NodeID, InputID, OutputID;
	nos::fb::TensorElementType cachedType;

	TensorToRGBANodeContext(nos::fb::Node const* node) : NodeContext(node) {
		NodeID = *node->id();
	}

	void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer* value) {
		if (NSN_In.Compare(pinName.AsCStr()) == 0) {
			auto tensor = flatbuffers::GetRoot<nos::fb::Tensor>(value->Data);
			auto proxy = tensor->UnPack();
			CreateOutputTextureFromTensor(proxy);
		}
	}

	nosResult DetectWidthHeight(int& width, int& height, nos::fb::TTensor* proxy) {
		int dimension = proxy->shape.size();
		if (dimension < 2)
			return NOS_RESULT_FAILED;
		
		//We have various cases:
		// 1x4x28x28 : ideal case for CNNs
		// but there might be unexpected sitatuions
		// such as 1x1x1x..x1x4x28x28 => ignore all ones
		// 4x28x28 => create an 28x28 RGBA texture
		// 3x28x28 => still create an 28x28 RGBA texture
		// 1x28x28 => still create an 28x28 RGBA texture
		// 28x28 => still create an 28x28 RGBA texture

		height = proxy->shape[dimension - 1];
		width = proxy->shape[dimension - 2];
		return NOS_RESULT_SUCCESS;
	}

	nosResult CreateOutputTextureFromTensor(nos::fb::TTensor* proxy) {
		int width = 0, height = 0;
		if (DetectWidthHeight(width, height, proxy) == NOS_RESULT_FAILED) {
			nosEngine.LogE("This tensor is not suitable for texture conversion!");
			return NOS_RESULT_FAILED;
		}

		if (OutputTexture.Info.Texture.Width == width && OutputTexture.Info.Texture.Height == height && cachedType == proxy->type) {
			return NOS_RESULT_SUCCESS;
		}

		if (OutputTexture.Memory.Handle != NULL) {
			nosEngine.Destroy(&OutputTexture);
		}

		cachedType == proxy->type;

		OutputTexture.Info.Texture.Width = width;
		OutputTexture.Info.Texture.Height = height;
		OutputTexture.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_SRC | NOS_IMAGE_USAGE_TRANSFER_DST);
		OutputTexture.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;

		switch (proxy->type) {
		case nos::fb::TensorElementType::UNDEFINED:
			return;
		case nos::fb::TensorElementType::UINT8:
			OutputTexture.Info.Texture.Format = NOS_FORMAT_R8G8B8A8_UINT;
			break;
		case nos::fb::TensorElementType::UINT16:
			OutputTexture.Info.Texture.Format = NOS_FORMAT_R16G16B16A16_UINT;
			break;
		case nos::fb::TensorElementType::UINT32:
			OutputTexture.Info.Texture.Format = NOS_FORMAT_R32G32B32A32_UINT;
			break;
		case nos::fb::TensorElementType::UINT64:
			//TODO: what should we do for this?
			break;
		case nos::fb::TensorElementType::INT8:
			//
			break;
		case nos::fb::TensorElementType::INT16:
			OutputTexture.Info.Texture.Format = NOS_FORMAT_R16G16B16A16_SINT;
			break;
		case nos::fb::TensorElementType::INT32:
			OutputTexture.Info.Texture.Format = NOS_FORMAT_R32G32B32A32_SINT;
			break;
		case nos::fb::TensorElementType::INT64:
			break;
		case nos::fb::TensorElementType::FLOAT:
			OutputTexture.Info.Texture.Format = NOS_FORMAT_R16G16B16A16_SSCALED;
			break;
		case nos::fb::TensorElementType::FLOAT16:
			OutputTexture.Info.Texture.Format = NOS_FORMAT_R16G16B16A16_SFLOAT;
			break;
		case nos::fb::TensorElementType::DOUBLE:
			OutputTexture.Info.Texture.Format = NOS_FORMAT_R16G16B16A16_SSCALED;
			break;
		case nos::fb::TensorElementType::BOOL:
			//????
			break;
		case nos::fb::TensorElementType::STRING:
			//???? dont allow connection
			//SetPinValues<std::string>(proxy);
			break;
		}

		nosEngine.Create(&OutputTexture);
		nosEngine.ImageLoad(reinterpret_cast<void*>(proxy->buffer), nosVec2u(width, height), OutputTexture.Info.Texture.Format ,&OutputTexture);
		nosEngine.SetPinValue(OutputID, nos::Buffer::From(OutputTexture));
		return NOS_RESULT_SUCCESS;
	}
};

void RegisterTensorToRGBA(nosNodeFunctions* outFunctions) {
	NOS_BIND_NODE_CLASS(NSN_TensorToRGBA, TensorToRGBANodeContext, outFunctions);
}