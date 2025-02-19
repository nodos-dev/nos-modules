// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginAPI.h>

#include <Builtins_generated.h>

#include <Nodos/PluginHelpers.hpp>

#include <nosVulkanSubsystem/nosVulkanSubsystem.h>
#include <nosVulkanSubsystem/Helpers.hpp>

#include "Window/WindowNode.h"
#include <cstdint>

NOS_INIT()

NOS_REGISTER_NAME(in1)
NOS_REGISTER_NAME(in2)
NOS_REGISTER_NAME(out)

NOS_VULKAN_INIT()

NOS_BEGIN_IMPORT_DEPS()
	NOS_VULKAN_IMPORT()
NOS_END_IMPORT_DEPS()

#define STRINGIZE(A, B) A##B

namespace nos::test
{


class TestNode : public nos::NodeContext
{
public:
	TestNode(nosFbNodePtr node) : nos::NodeContext(node)
	{
		nosEngine.LogI("TestNode: %s", __FUNCTION__);
		AddPinValueWatcher(NOS_NAME_STATIC("double_prop"), [this](nos::Buffer const& newVal, std::optional<nos::Buffer> oldVal) {
			double optOldVal = 0.0f;
			if (oldVal)
				optOldVal = *oldVal->As<double>();
			nosEngine.LogI("TestNode: double_prop changed to %f from %f", *newVal.As<double>(), optOldVal);
			});
	}
	~TestNode()
	{
		nosEngine.LogI("TestNode: %s", __FUNCTION__);
	}
	void OnNodeUpdated(const nosNodeUpdate* updatedNode) override { nosEngine.LogI("TestNode: %s", __FUNCTION__); }
	void OnPinValueChanged(nos::Name pinName, uuid const& pinId, nosBuffer value) override
	{
		nosEngine.LogI("TestNode: %s", __FUNCTION__);
	}
	virtual void OnPinConnected(nos::Name pinName, uuid const& connectedPin) override { nosEngine.LogI("TestNode: %s", __FUNCTION__); }
	virtual void OnPinDisconnected(nos::Name pinName) override { nosEngine.LogI("TestNode: %s", __FUNCTION__); }
	virtual void OnPathCommand(const nosPathCommand* command) override { nosEngine.LogI("TestNode: %s", __FUNCTION__); }
	virtual nosResult CanRemoveOrphanPin(nos::Name pinName, uuid const& pinId) override
	{
		nosEngine.LogI("TestNode: %s", __FUNCTION__);
		return NOS_RESULT_SUCCESS;
	}

	// Execution
	virtual nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		nosEngine.LogI("TestNode: %s", __FUNCTION__);
		return NOS_RESULT_SUCCESS;
	}
	virtual nosResult CopyFrom(nosCopyInfo* copyInfo) override
	{
		nosEngine.LogI("TestNode: %s", __FUNCTION__);
		return NOS_RESULT_SUCCESS;
	}

	// Menu & key events
	virtual void OnMenuRequested(nosContextMenuRequestPtr request) override
	{
		nosEngine.LogI("TestNode: %s", __FUNCTION__);
	}
	virtual void OnMenuCommand(uuid const& itemID, uint32_t cmd) override
	{
		nosEngine.LogI("TestNode: %s", __FUNCTION__);
	}
	virtual void OnKeyEvent(const nosKeyEvent* keyEvent) override { nosEngine.LogI("TestNode: %s", __FUNCTION__); }

	virtual void OnPinDirtied(uuid const& pinID, uint64_t frameCount) override
	{
		nosEngine.LogI("TestNode: %s", __FUNCTION__);
	}
	virtual void OnPathStateChanged(nosPathState pathState) override { nosEngine.LogI("TestNode: %s", __FUNCTION__); }
	virtual void OnEnterRunnerThread(nosEnterRunnerThreadParams const& params) override
	{
		nosEngine.LogI("TestNode: %s", __FUNCTION__);
	}
	virtual void OnExitRunnerThread(nosExitRunnerThreadParams const& params) override
	{
		nosEngine.LogI("TestNode: %s", __FUNCTION__);
	}

	static nosResult TestFunction(void* ctx, nosFunctionExecuteParams* params)
	{
		auto args = nos::GetPinValues(params->FunctionNodeExecuteParams);

		auto a = *GetPinValue<double>(args, NSN_in1);
		auto b = *GetPinValue<double>(args, NSN_in2);
		auto c = a + b;
		nosEngine.SetPinValue(params->FunctionNodeExecuteParams->Pins[2].Id, { .Data = &c, .Size = sizeof(c) });
		
		TestNode* node = static_cast<TestNode*>(ctx);
		if (node->SecondFunc)
		{
			TPartialNodeUpdate update{};
			update.node_id = node->NodeId;
			update.functions_to_delete = { *node->SecondFunc };
			flatbuffers::FlatBufferBuilder fbb;
			auto event = CreateAppEvent(fbb, CreatePartialNodeUpdate(fbb, &update));
			HandleEvent(event);
			node->SecondFunc = std::nullopt;
		}
		else
		{
			node->SecondFunc = nosEngine.GenerateID();
			TPartialNodeUpdate update{};
			update.node_id = node->NodeId;
			std::unique_ptr<fb::TNode> functionNode = std::make_unique<fb::TNode>();
			functionNode->id = *node->SecondFunc;
			functionNode->class_name = NOS_NAME("DynamicFunction");
			fb::TJob job{};
			functionNode->contents.Set(job);
			update.functions_to_add.push_back(std::move(functionNode));
			flatbuffers::FlatBufferBuilder fbb;
			auto event = CreateAppEvent(fbb, CreatePartialNodeUpdate(fbb, &update));
			HandleEvent(event);
		}
		return NOS_RESULT_SUCCESS;
	}

	static nosResult DynamicFunction(void* ctx, nosFunctionExecuteParams* params) 
	{
		nosEngine.LogI("DynamicFunction executed");
		return NOS_RESULT_SUCCESS;
	}

	static nosResult GetFunctions(size_t* outCount, nosName* pName, nosPfnNodeFunctionExecute* fns)
	{
		*outCount = 2;
		if (!pName || !fns)
			return NOS_RESULT_SUCCESS;

		fns[0] = TestFunction;
		pName[0] = NOS_NAME_STATIC("TestFunction");
		fns[1] = DynamicFunction;
		pName[1] = NOS_NAME_STATIC("DynamicFunction");
		return NOS_RESULT_SUCCESS;
	}

	std::optional<uuid> SecondFunc = std::nullopt;
};

struct AlwaysDirtyNode : nos::NodeContext
{
	using NodeContext::NodeContext;
	void OnPinValueChanged(nos::Name pinName, uuid const& pinId, nosBuffer value) override
	{
		if (pinName == NOS_NAME_STATIC("OutLive"))
			ChangePinLiveness(NOS_NAME_STATIC("Output"), *InterpretPinValue<bool>(value));
	}
};

struct PrintNode : nos::NodeContext
{
	using NodeContext::NodeContext;
	nosResult ExecuteNode(nosNodeExecuteParams* execParams) override
	{
		NodeExecuteParams params(execParams);
		const char* log = InterpretPinValue<const char>(params[NOS_NAME("Log")].Data->Data);
		nosEngine.LogI("Print: %s", log);
		return NOS_RESULT_SUCCESS;
	}
};

nosResult RegisterFrameInterpolator(nosNodeFunctions* nodeFunctions);

struct TestPluginFunctions : PluginFunctions
{
	nosResult Initialize() override
	{
		nosTypeInfo* leakedTypeInfo{};
		auto res  = nosEngine.GetTypeInfo(NOS_NAME_STATIC("nos.test.TestStruct"), &leakedTypeInfo);
		assert(res == NOS_RESULT_SUCCESS);

		auto vkResOpt = vkss::Resource::Create(
			{.Info = {.Type = NOS_RESOURCE_TYPE_TEXTURE,
					  .Texture = {.Width = 1, .Height = 1, .Format = NOS_FORMAT_R8G8B8A8_UNORM}}},
			"Leaked Texture");

		assert(vkResOpt);
		vkResOpt->Release();

		vkResOpt = vkss::Resource::Create(
			{.Info = {.Type = NOS_RESOURCE_TYPE_BUFFER, .Buffer = {.Size = 1, .Usage = NOS_BUFFER_USAGE_TRANSFER_DST}}},
			"Leaked Buffer");
		assert(vkResOpt);
		vkResOpt->Release();

		nosGPUEventResource leakedEventResource{};
		res = nosVulkan->CreateGPUEventResource(&leakedEventResource);
		assert(res == NOS_RESULT_SUCCESS);
		res = nosVulkan->IncreaseGPUEventResourceRefCount(leakedEventResource);
		assert(res == NOS_RESULT_SUCCESS);
		res = nosVulkan->DestroyGPUEventResource(&leakedEventResource);
		assert(res == NOS_RESULT_SUCCESS);
		return NOS_RESULT_SUCCESS;
	}
	nosResult ExportNodeFunctions(size_t& outCount, nosNodeFunctions** outFunctions) override
	{
		outCount = 13;
		if (!outFunctions)
			return NOS_RESULT_SUCCESS;

		nosModuleStatusMessage msg;
		msg.ModuleId = nosEngine.Module->Id;
		msg.Message = "Test module loaded";
		msg.MessageType = NOS_MODULE_STATUS_MESSAGE_TYPE_INFO;
		msg.UpdateType = NOS_MODULE_STATUS_MESSAGE_UPDATE_TYPE_REPLACE;
		nosEngine.SendModuleStatusMessageUpdate(&msg);

		NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.test.NodeTest"), TestNode, outFunctions[0]);
		outFunctions[1]->ClassName = NOS_NAME_STATIC("nos.test.NodeWithCategories");
		outFunctions[2]->ClassName = NOS_NAME_STATIC("nos.test.NodeWithFunctions");
		outFunctions[2]->GetFunctions = [](size_t* outCount, nosName* pName, nosPfnNodeFunctionExecute* fns) {
			*outCount = 1;
			if (!pName || !fns)
				return NOS_RESULT_SUCCESS;

			fns[0] = [](void* ctx, nosFunctionExecuteParams* params) {
				NodeExecuteParams execParams(params->FunctionNodeExecuteParams);

				nosEngine.LogI("NodeWithFunctions: TestFunction executed");

				double res = *InterpretPinValue<double>(execParams[NOS_NAME("in1")].Data->Data) +
							 *InterpretPinValue<double>(execParams[NOS_NAME("in2")].Data->Data);

				nosEngine.SetPinValue(execParams[NOS_NAME("out")].Id, nos::Buffer::From(res));
				nosEngine.SetPinDirty(execParams[NOS_NAME("OutTrigger")].Id);
				return NOS_RESULT_SUCCESS;
			};
			pName[0] = NOS_NAME_STATIC("TestFunction");
			return NOS_RESULT_SUCCESS;
		};

		outFunctions[3]->ClassName = NOS_NAME_STATIC("nos.test.NodeWithCustomTypes");
		outFunctions[4]->ClassName = NOS_NAME_STATIC("nos.test.CopyTest");
		outFunctions[4]->ExecuteNode = [](void* ctx, nosNodeExecuteParams* params) {
			nosCmd cmd = nos::vkss::BeginCmd(NOS_NAME("(nos.test.CopyTest) Copy"), params->NodeId);
			auto values = nos::GetPinValues(params);
			nosResourceShareInfo input = nos::vkss::DeserializeTextureInfo(values[NOS_NAME_STATIC("Input")]);
			nosResourceShareInfo output = nos::vkss::DeserializeTextureInfo(values[NOS_NAME_STATIC("Output")]);
			nosVulkan->Copy(cmd, &input, &output, 0);
			nosVulkan->End(cmd, NOS_FALSE);
			return NOS_RESULT_SUCCESS;
		};

		outFunctions[5]->ClassName = NOS_NAME_STATIC("nos.test.CopyTestLicensed");
		outFunctions[5]->OnNodeCreated = [](nosFbNodePtr node, void** outCtxPtr) {
			nosEngine.RegisterFeature(
				uuid(*node->id()), "Nodos.CopyTestLicensed", 1, "Nodos.CopyTestLicensed required");
		};
		outFunctions[5]->OnNodeDeleted = [](void* ctx, nosUUID nodeId) {
			nosEngine.UnregisterFeature(nodeId, "Nodos.CopyTestLicensed");
		};
		outFunctions[5]->ExecuteNode = [](void* ctx, nosNodeExecuteParams* params) {
			nosCmd cmd = nos::vkss::BeginCmd(NOS_NAME("(nos.test.CopyTest) Copy"), params->NodeId);
			auto values = nos::GetPinValues(params);
			nosResourceShareInfo input = nos::vkss::DeserializeTextureInfo(values[NOS_NAME_STATIC("Input")]);
			nosResourceShareInfo output = nos::vkss::DeserializeTextureInfo(values[NOS_NAME_STATIC("Output")]);
			nosVulkan->Copy(cmd, &input, &output, 0);
			nosVulkan->End(cmd, nullptr);
			return NOS_RESULT_SUCCESS;
		};
		outFunctions[6]->ClassName = NOS_NAME_STATIC("nos.test.CopyBuffer");
		outFunctions[6]->ExecuteNode = [](void* ctx, nosNodeExecuteParams* params) {
			auto inBuf = nos::GetPinValue<sys::vulkan::Buffer>(nos::GetPinValues(params), NOS_NAME_STATIC("Input"));
			auto outBuf = nos::GetPinValue<sys::vulkan::Buffer>(nos::GetPinValues(params), NOS_NAME_STATIC("Output"));
			auto in = vkss::ConvertToResourceInfo(*inBuf);
			if (in.Memory.Handle == 0)
				return NOS_RESULT_INVALID_ARGUMENT;
			auto out = vkss::ConvertToResourceInfo(*outBuf);
			if (out.Info.Buffer.Size != in.Info.Buffer.Size)
			{
				out = in;
				out.Info.Buffer.Usage = nosBufferUsage(out.Info.Buffer.Usage | NOS_BUFFER_USAGE_TRANSFER_DST);
				auto outRes = vkss::Resource::CreateWithSameInfo(out, "CopyBuffer");
				out = *outRes;
				nosEngine.SetPinValue(params->Pins[1].Id, outRes->ToPinData());
			}
			nosCmd cmd = nos::vkss::BeginCmd(NOS_NAME("(nos.test.CopyBuffer) Copy"), params->NodeId);
			nosVulkan->Copy(cmd, &in, &out, 0);
			nosVulkan->End(cmd, nullptr);
			return NOS_RESULT_SUCCESS;
		};
		RegisterFrameInterpolator(outFunctions[7]);
		nos::test::RegisterWindowNode(outFunctions[8]);
		outFunctions[9]->ClassName = NOS_NAME_STATIC("nos.test.BypassTexture");
		outFunctions[9]->ExecuteNode = [](void* ctx, nosNodeExecuteParams* params) {
			auto values = nos::GetPinValues(params);
			nos::sys::vulkan::TTexture in, out;
			auto intex = flatbuffers::GetRoot<nos::sys::vulkan::Texture>(values[NOS_NAME_STATIC("Input")]);
			intex->UnPackTo(&in);
			out = in;
			out.unmanaged = true;
			auto ids = nos::GetPinIds(params);
			nosEngine.SetPinValue(ids[NOS_NAME_STATIC("Output")], nos::Buffer::From(out));
			return NOS_RESULT_SUCCESS;
		};
		outFunctions[10]->ClassName = NOS_NAME_STATIC("nos.test.LiveOutWithInput");
		outFunctions[10]->CopyFrom = [](void* ctx, nosCopyInfo* copyInfo) {
			nosEngine.LogD("LiveOutWithInput: CopyFrom");
			return NOS_RESULT_SUCCESS;
		};
		NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.test.AlwaysDirty"), AlwaysDirtyNode, outFunctions[11]);
		NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("Print"), PrintNode, outFunctions[12]);
		return NOS_RESULT_SUCCESS;
	}
};

NOS_EXPORT_PLUGIN_FUNCTIONS(TestPluginFunctions)

} // namespace nos::test
