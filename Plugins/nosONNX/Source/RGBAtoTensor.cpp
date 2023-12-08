#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include <onnxruntime_cxx_api.h>
#include "ONNXRTCommon.h"
#include "Tensor.h"

NOS_REGISTER_NAME(RGBAtoTensor);
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME(TensorShape);
NOS_REGISTER_NAME(Out);

struct RGBAtoTensorNodeContext : nos::NodeContext
{
	nosResourceShareInfo InputImage;
	nos::fb::TTensor nosOutputTensor;
	nosTensor OutputTensor;
	nosUUID NodeID, OutputID;
	nosResourceShareInfo DummyInput = {};
	nosResourceShareInfo InputFormatted = {};
	//This is required for changing the size (not sure will use it?)
	nosResourceShareInfo InputBuffer = {};
	std::atomic_bool shouldStop = false, isTensorReadable = false;
	std::condition_variable WaitInput;
	std::mutex InputMutex;
	std::thread TensorConverter;
	RGBAtoTensorNodeContext(nos::fb::Node const* node) : NodeContext(node) {

		DummyInput.Info.Texture.Format = NOS_FORMAT_R8G8B8A8_UNORM;
		DummyInput.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;

		NodeID = *node->id();
		for (auto pin : *node->pins()) {
			if (NSN_TensorShape.Compare(pin->name()->c_str()) == 0) {
				auto tensor = flatbuffers::GetRoot<nos::fb::Tensor>(pin->data());
				if (tensor->shape() != nullptr) {
					OutputTensor.SetShape(std::vector<int64_t>(tensor->shape()->data(), tensor->shape()->data() + tensor->shape()->Length()));
				}
			}
			if (NSN_Out.Compare(pin->name()->c_str()) == 0) {
				OutputID = *pin->id();
			}
		}

		TensorConverter = std::thread([this]() {this->ConvertRGBAtoTensor(); });
	}

	~RGBAtoTensorNodeContext() {
		if (TensorConverter.joinable()) {
			shouldStop = true;
			WaitInput.notify_one();
			TensorConverter.join();
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

	void ConvertRGBAtoTensor() {
		
		while (!shouldStop) {

			{
				std::unique_lock<std::mutex> inputLock(InputMutex);
				WaitInput.wait(inputLock);
				if (shouldStop)
					return;
			}

			isTensorReadable = false;

			if (OutputTensor.GetShape().size() != 4 || DummyInput.Info.Texture.Width != OutputTensor.GetShape()[2] && DummyInput.Info.Texture.Height != OutputTensor.GetShape()[3]) {
				
				if (InputBuffer.Memory.Handle != NULL)
					nosEngine.Destroy(&InputBuffer);

				if (InputFormatted.Memory.Handle != NULL)
					nosEngine.Destroy(&InputFormatted);

				InputFormatted.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
				InputFormatted.Info.Texture.Width = DummyInput.Info.Texture.Width;
				InputFormatted.Info.Texture.Height = DummyInput.Info.Texture.Height;
				//TODO: Make use of tensor type before deciding to this
				InputFormatted.Info.Texture.Format = NOS_FORMAT_R8G8B8A8_SRGB;
				InputFormatted.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_SRC | NOS_IMAGE_USAGE_TRANSFER_DST);
				nosEngine.Create(&InputFormatted);


				InputBuffer.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
				InputBuffer.Info.Buffer.Size = DummyInput.Info.Texture.Width * DummyInput.Info.Texture.Height * sizeof(uint8_t) * 4;
				InputBuffer.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_TRANSFER_DST);
				nosEngine.Create(&InputBuffer);
				std::vector<int64_t> tensorShape = { 1, 4, DummyInput.Info.Texture.Width, DummyInput.Info.Texture.Height };
				OutputTensor.SetShape(std::move(tensorShape));


				flatbuffers::FlatBufferBuilder fbb;
				std::vector<flatbuffers::Offset<nos::PartialPinUpdate>> Offsets;
				Offsets.push_back(nos::CreatePartialPinUpdateDirect(fbb, &OutputID, 0, 0, nos::Action::NOP, nos::Action::NOP, 0, OutputTensor.GetShapeStr().c_str()));
				HandleEvent(
					nos::CreateAppEvent(fbb,
						nos::CreatePartialNodeUpdateDirect(fbb, &NodeID, nos::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, &Offsets)));
			}

			nosCmd blitCmd;
			nosEngine.Begin(&blitCmd);
			nosEngine.Copy(blitCmd, &DummyInput, &InputFormatted, nullptr);
			nosEngine.End(blitCmd);

			nosCmd downloadCmd;
			nosEngine.Begin(&downloadCmd);
			nosEngine.Copy(downloadCmd, &InputFormatted, &InputBuffer, nullptr);
			nosEngine.End(downloadCmd);

			auto data = nosEngine.Map(&InputBuffer);
			
			nosOutputTensor.buffer = (uint64_t)(data);
			nosOutputTensor.shape = OutputTensor.GetShape();
			nosOutputTensor.type = nos::fb::TensorElementType::UINT8;

			nosEngine.SetPinValue(OutputID, nos::Buffer::From(nosOutputTensor));

			isTensorReadable = true;
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
