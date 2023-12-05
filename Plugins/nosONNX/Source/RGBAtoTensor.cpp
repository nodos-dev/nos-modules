#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include <onnxruntime_cxx_api.h>
#include "ONNXRTCommon.h"

NOS_REGISTER_NAME(RGBAtoTensor);
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME(TensorShape);
NOS_REGISTER_NAME(Out);

struct RGBAtoTensorNodeContext : nos::NodeContext
{
	nosResourceShareInfo InputImage;
	nos::fb::TTensor nosOutputTensor;
	nosTensor OutputTensor;
	nosUUID NodeID;
	nosResourceShareInfo DummyInput = {};
	//This is required for changing the size (not sure will use it?)
	nosResourceShareInfo InputBuffer = {};
	std::atomic_bool shouldStop = false;
	std::condition_variable WaitInput;
	std::mutex InputMutex;
	RGBAtoTensorNodeContext(nos::fb::Node const* node) : NodeContext(node) {

		DummyInput.Info.Texture.Format = NOS_FORMAT_B8G8R8A8_SRGB;
		DummyInput.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;

		NodeID = *node->id();
		for (auto pin : *node->pins()) {
			if (NSN_TensorShape.Compare(pin->name()->c_str()) == 0) {
				auto tensor = flatbuffers::GetRoot<nos::fb::Tensor>(pin->data());
				if (tensor->shape() != nullptr) {
					OutputTensor.SetShape(std::vector<int64_t>(tensor->shape()->data(), tensor->shape()->data() + tensor->shape()->Length()));
				}
			}
		}
	}

	void  OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer* value) override {
		if (pinName == NSN_In) {
			DummyInput = nos::DeserializeTextureInfo(value->Data);
			WaitInput.notify_one();
		}
		if (NSN_TensorShape.Compare(pinName.AsCStr()) == 0) {
			auto tensor = flatbuffers::GetRoot<nos::fb::Tensor>(value->Data);
			if (tensor->shape() != nullptr) {
				OutputTensor.SetShape(std::vector<int64_t>(tensor->shape()->data(), tensor->shape()->data() + tensor->shape()->Length()));
			}
			//For now dont allow to create data from editor
			/*if (tensor->buffer() != nullptr) {
				OutputTensor.SetData(std::vector<int64_t>(tensor->buffer()->data(), tensor->buffer()->data() + tensor->buffer()->Length()));
			}*/
		}
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* args) {

		if (DummyInput.Info.Texture.Width != OutputTensor.GetShape()[2] && DummyInput.Info.Texture.Height != OutputTensor.GetShape()[3]) {
			if (InputBuffer.Memory.Handle != NULL)
				nosEngine.Destroy(&InputBuffer);

			InputBuffer.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
			InputBuffer.Info.Buffer.Size = DummyInput.Info.Texture.Width * DummyInput.Info.Texture.Height * sizeof(uint8_t);
			InputBuffer.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_TRANSFER_DST);
			nosEngine.Create(&InputBuffer);
			std::vector<int64_t> tensorShape = { 1, 1, DummyInput.Info.Texture.Width, DummyInput.Info.Texture.Height };
			OutputTensor.SetShape(std::move(tensorShape));
		}

		nosCmd downloadCmd;

		nosEngine.Begin(&downloadCmd);
		nosEngine.Download(&downloadCmd, &DummyInput, &InputBuffer); 
		nosEngine.End(&downloadCmd);

		auto data = nosEngine.Map(&InputBuffer);
		OutputTensor.SetData<uint8_t>(data, OutputTensor.GetLength(), false);
		nosOutputTensor.buffer = data;


		return NOS_RESULT_SUCCESS;
	}

	void CreateTensor() {
		
		while (!shouldStop) {
			if (DummyInput.Info.Texture.Width != OutputTensor.GetShape()[2] && DummyInput.Info.Texture.Height != OutputTensor.GetShape()[3]) {
				if (InputBuffer.Memory.Handle != NULL)
					nosEngine.Destroy(&InputBuffer);

				InputBuffer.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
				InputBuffer.Info.Buffer.Size = DummyInput.Info.Texture.Width * DummyInput.Info.Texture.Height * sizeof(uint8_t);
				InputBuffer.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_TRANSFER_DST);
				nosEngine.Create(&InputBuffer);
				std::vector<int64_t> tensorShape = { 1, 1, DummyInput.Info.Texture.Width, DummyInput.Info.Texture.Height };
				OutputTensor.SetShape(std::move(tensorShape));
			}

			nosCmd downloadCmd;
			
			nosEngine.Begin(&downloadCmd);
			nosEngine.Download(&downloadCmd, &DummyInput, &InputBuffer);
			nosEngine.End(&downloadCmd);

			auto data = nosEngine.Map(&InputBuffer);
			OutputTensor.SetData<uint8_t>(data, OutputTensor.GetLength());

			std::unique_lock<std::mutex> inputLock(InputMutex);
			WaitInput.wait(inputLock);
		}
	}
	
	void OnPinConnected(nos::Name pinName, nosUUID connectedPin) {
	}

	/*
	nosResult BeginCopyTo(nosCopyInfo* cpy) override {

	}

	void EndCopyTo(nosCopyInfo* cpy) override {

	}*/
};

void RegisterRGBAtoTensor(nosNodeFunctions* outFunctions) {
		NOS_BIND_NODE_CLASS(NSN_RGBAtoTensor, RGBAtoTensorNodeContext, outFunctions);
}
