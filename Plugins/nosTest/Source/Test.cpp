// Copyright Nodos AS. All Rights Reserved.

#include <Nodos/PluginAPI.h>

#include <Builtins_generated.h>

#include <Nodos/PluginHelpers.hpp>

#include <nosVulkanSubsystem/nosVulkanSubsystem.h>
#include <nosVulkanSubsystem/Helpers.hpp>

#include "Window/WindowNode.h"

NOS_INIT_WITH_MIN_REQUIRED_MINOR(0) // Do not forget to remove this minimum required minor version on major version
									// changes, or we might not be loaded.
NOS_REGISTER_NAME(in1)
NOS_REGISTER_NAME(in2)
NOS_REGISTER_NAME(out)

NOS_VULKAN_INIT();

namespace nos::test
{


class TestNode : public nos::NodeContext
{
public:
	TestNode(const nosFbNode* node) : nos::NodeContext(node) 
	{ 
		nosEngine.LogI("TestNode: " __FUNCTION__);
		AddPinValueWatcher(NOS_NAME_STATIC("double_prop"), [this](nos::Buffer const& newVal, nos::Buffer const& oldVal) {
			double optOldVal = 0.0f;
			if (oldVal.Size() > 0)
				optOldVal = *oldVal.As<double>();
			nosEngine.LogI("TestNode: double_prop changed to %f from %f", *newVal.As<double>(), optOldVal);
		});
	}
	~TestNode() 
	{ 
		nosEngine.LogI("TestNode: " __FUNCTION__);
	}
	virtual void OnNodeUpdated(const nosFbNode* updatedNode) { nosEngine.LogI("TestNode: " __FUNCTION__); }
	virtual void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer value)
	{
		nosEngine.LogI("TestNode: " __FUNCTION__);
	}
	virtual void OnPinConnected(nos::Name pinName, nosUUID connectedPin) { nosEngine.LogI("TestNode: " __FUNCTION__); }
	virtual void OnPinDisconnected(nos::Name pinName) { nosEngine.LogI("TestNode: " __FUNCTION__); }
	virtual void OnPinShowAsChanged(nos::Name pinName, nos::fb::ShowAs showAs)
	{
		nosEngine.LogI("TestNode: " __FUNCTION__);
	}
	virtual void OnPathCommand(const nosPathCommand* command) { nosEngine.LogI("TestNode: OnNodeUpdated"); }
	virtual nosResult CanRemoveOrphanPin(nos::Name pinName, nosUUID pinId)
	{
		nosEngine.LogI("TestNode: " __FUNCTION__);
		return NOS_RESULT_SUCCESS;
	}
	virtual nosResult OnOrphanPinRemoved(nos::Name pinName, nosUUID pinId)
	{
		nosEngine.LogI("TestNode: " __FUNCTION__);
		return NOS_RESULT_SUCCESS;
	}

	// Execution
	virtual nosResult ExecuteNode(const nosNodeExecuteArgs* args)
	{
		nosEngine.LogI("TestNode: " __FUNCTION__);
		return NOS_RESULT_SUCCESS;
	}
	virtual nosResult BeginCopyFrom(nosCopyInfo* copyInfo)
	{
		nosEngine.LogI("TestNode: " __FUNCTION__);
		return NOS_RESULT_SUCCESS;
	}
	virtual nosResult BeginCopyTo(nosCopyInfo* copyInfo)
	{
		nosEngine.LogI("TestNode: " __FUNCTION__);
		return NOS_RESULT_SUCCESS;
	}
	virtual void EndCopyFrom(nosCopyInfo* copyInfo) { nosEngine.LogI("TestNode: " __FUNCTION__); }
	virtual void EndCopyTo(nosCopyInfo* copyInfo) { nosEngine.LogI("TestNode: " __FUNCTION__); }

	// Menu & key events
	virtual void OnMenuRequested(const nosContextMenuRequest* request) { nosEngine.LogI("TestNode: " __FUNCTION__); }
	virtual void OnMenuCommand(nosUUID itemID, uint32_t cmd) { nosEngine.LogI("TestNode: " __FUNCTION__); }
	virtual void OnKeyEvent(const nosKeyEvent* keyEvent) { nosEngine.LogI("TestNode: " __FUNCTION__); }

	virtual void OnPinDirtied(nosUUID pinID, uint64_t frameCount) { nosEngine.LogI("TestNode: " __FUNCTION__); }
	virtual void OnPathStateChanged(nosPathState pathState) { nosEngine.LogI("TestNode: " __FUNCTION__); }

	
	static void TestFunction(void* ctx, const nosNodeExecuteArgs* nodeArgs, const nosNodeExecuteArgs* functionArgs)
	{
		auto args = nos::GetPinValues(functionArgs);

		auto a = *GetPinValue<double>(args, NSN_in1);
		auto b = *GetPinValue<double>(args, NSN_in2);
		auto c = a + b;
		nosEngine.SetPinValue(functionArgs->Pins[2].Id, {.Data = &c, .Size = sizeof(c)});
	}

	static nosResult GetFunctions(size_t* outCount, nosName* pName, nosPfnNodeFunctionExecute* fns)
	{
		*outCount = 1;
		if (!pName || !fns)
			return NOS_RESULT_SUCCESS;

		*fns = TestFunction;
		*pName = NOS_NAME_STATIC("TestFunction");
		return NOS_RESULT_SUCCESS;
	}
};

nosResult RegisterFrameInterpolator(nosNodeFunctions* nodeFunctions);

extern "C"
{


	NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outCount, nosNodeFunctions** outFunctions)
	{
		*outCount = (size_t)(9);
		if (!outFunctions)
			return NOS_RESULT_SUCCESS;

		auto ret = RequestVulkanSubsystem();
		if (ret != NOS_RESULT_SUCCESS)
			return ret;
		
		NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.test.NodeTest"), TestNode, outFunctions[0]);
		outFunctions[1]->ClassName = NOS_NAME_STATIC("nos.test.NodeWithCategories");
		outFunctions[2]->ClassName = NOS_NAME_STATIC("nos.test.NodeWithFunctions");
		outFunctions[3]->ClassName = NOS_NAME_STATIC("nos.test.NodeWithCustomTypes");
		outFunctions[4]->ClassName = NOS_NAME_STATIC("nos.test.CopyTest");
		outFunctions[4]->ExecuteNode = [](void* ctx, const nosNodeExecuteArgs* args)
		{
			nosCmd cmd;
			nosVulkan->Begin("(nos.test.CopyTest) Copy", &cmd);
			auto values = nos::GetPinValues(args);
			nosResourceShareInfo input = nos::vkss::DeserializeTextureInfo(values[NOS_NAME_STATIC("Input")]);
			nosResourceShareInfo output = nos::vkss::DeserializeTextureInfo(values[NOS_NAME_STATIC("Output")]);
			nosVulkan->Copy(cmd, &input, &output, 0);
			nosVulkan->End(cmd, NOS_FALSE);
			return NOS_RESULT_SUCCESS;
		};
		
		outFunctions[5]->ClassName = NOS_NAME_STATIC("nos.test.CopyTestLicensed");
		outFunctions[5]->OnNodeCreated = [] (const nosFbNode* node, void** outCtxPtr) {
			nosEngine.RegisterFeature(*node->id(), "Nodos.CopyTestLicensed", 1, "Nodos.CopyTestLicensed required");	
		};
		outFunctions[5]->OnNodeDeleted = [] (void* ctx, nosUUID nodeId) {
			nosEngine.UnregisterFeature(nodeId, "Nodos.CopyTestLicensed");
		};
		outFunctions[5]->ExecuteNode = [](void* ctx, const nosNodeExecuteArgs* args)
		{
			nosCmd cmd;
			nosVulkan->Begin("(nos.test.CopyTest) Copy", &cmd);
			auto values = nos::GetPinValues(args);
			nosResourceShareInfo input = nos::vkss::DeserializeTextureInfo(values[NOS_NAME_STATIC("Input")]);
			nosResourceShareInfo output = nos::vkss::DeserializeTextureInfo(values[NOS_NAME_STATIC("Output")]);
			nosVulkan->Copy(cmd, &input, &output, 0);
			nosVulkan->End(cmd, nullptr);
			return NOS_RESULT_SUCCESS;
		};
		outFunctions[6]->ClassName = NOS_NAME_STATIC("nos.test.CopyBuffer");
		outFunctions[6]->ExecuteNode = [](void* ctx, const nosNodeExecuteArgs* args) {
			auto inBuf = nos::GetPinValue<sys::vulkan::Buffer>(nos::GetPinValues(args), NOS_NAME_STATIC("Input"));
			auto outBuf = nos::GetPinValue<sys::vulkan::Buffer>(nos::GetPinValues(args), NOS_NAME_STATIC("Output"));
			auto in = vkss::ConvertToResourceInfo(*inBuf);
			if (in.Memory.Handle == 0)
				return NOS_RESULT_INVALID_ARGUMENT;
			auto out = vkss::ConvertToResourceInfo(*outBuf);
			if (out.Info.Buffer.Size != in.Info.Buffer.Size)
			{
				out = in;
				nosVulkan->CreateResource(&out);
				out.Info.Buffer.Usage = nosBufferUsage(out.Info.Buffer.Usage | NOS_BUFFER_USAGE_TRANSFER_DST);
				auto newBuf = nos::Buffer::From(vkss::ConvertBufferInfo(out));
				nosEngine.SetPinValue(args->Pins[1].Id, newBuf);
			}
			nosCmd cmd{};
			nosVulkan->Begin("(nos.test.CopyBuffer) Copy", &cmd);
			nosVulkan->Copy(cmd, &in, &out, 0);
			nosVulkan->End(cmd, nullptr);
			return NOS_RESULT_SUCCESS;
		};
		RegisterFrameInterpolator(outFunctions[7]);
		nos::test::RegisterWindowNode(outFunctions[8]);
		return NOS_RESULT_SUCCESS;

	}

}
} // namespace nos::test
