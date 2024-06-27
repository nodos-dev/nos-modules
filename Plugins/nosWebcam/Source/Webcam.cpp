// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginAPI.h>

#include <Builtins_generated.h>

#include <Nodos/PluginHelpers.hpp>

#include <nosVulkanSubsystem/nosVulkanSubsystem.h>
#include <nosVulkanSubsystem/Helpers.hpp>

#include "MFCapture.h"
#include "nosUtil/Stopwatch.hpp"

NOS_INIT_WITH_MIN_REQUIRED_MINOR(0) // Do not forget to remove this minimum required minor version on major version
									// changes, or we might not be loaded.
NOS_VULKAN_INIT();

namespace nos::webcam
{
struct WebcamInNode : public nos::NodeContext
{
	WebcamInNode(const nosFbNode* node) : nos::NodeContext(node), Cap()
	{
		auto devices = Cap.EnumerateDevices();
		auto kv = std::views::keys(devices);
		auto id = devices[kv.front()];
		SelectedDeviceInfo = Cap.CreateDeviceFromName(id);
		UpdateDeviceInfo();
	}

	void UpdateDeviceInfo()
	{
		nosEngine.SetPinValueByName(NodeId, NOS_NAME("Resolution"), nos::Buffer::From(SelectedDeviceInfo.Resolution));
		nosEngine.SetPinValueByName(NodeId, NOS_NAME("Format"), nosBuffer{ .Data = (void*)SelectedDeviceInfo.Format.c_str(), .Size = SelectedDeviceInfo.Format.size() + 1 });
		nosEngine.SetPinValueByName(NodeId, NOS_NAME("DeltaSeconds"), nos::Buffer::From(SelectedDeviceInfo.DeltaSeconds));
	}

	// Execution
	virtual nosResult ExecuteNode(const nosNodeExecuteArgs* args)
	{
		if (SelectedDeviceInfo.Resolution.x() == 0)
			return NOS_RESULT_FAILED;

		auto sample = Cap.ReadSample();

		// First sample is not valid, try again
		if (sample.Size == 0)
		{
			sample = Cap.ReadSample();
			if (sample.Size == 0)
				return NOS_RESULT_FAILED;
		}

		nos::NodeExecuteArgs execArgs(args);

		auto buffer = vkss::ConvertToResourceInfo(*InterpretPinValue<nos::sys::vulkan::Buffer>(*execArgs[NOS_NAME("Output")].Data));

		if (buffer.Info.Buffer.Size != sample.Size)
		{
			nosResourceShareInfo bufInfo{};
			bufInfo.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
			bufInfo.Info.Buffer.Size = sample.Size;
			bufInfo.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_STORAGE_BUFFER);
			bufInfo.Info.Buffer.MemoryFlags = nosMemoryFlags(NOS_MEMORY_FLAGS_HOST_VISIBLE);

			nosEngine.SetPinValue(execArgs[NOS_NAME("Output")].Id, nos::Buffer::From(vkss::ConvertBufferInfo(bufInfo)));
			buffer = vkss::ConvertToResourceInfo(*InterpretPinValue<nos::sys::vulkan::Buffer>(*execArgs[NOS_NAME("Output")].Data));
		}

		uint8_t* mapped = nosVulkan->Map(&buffer);

		memcpy(mapped, sample.Data, sample.Size);


		return NOS_RESULT_SUCCESS;
	}

	Capturer Cap;
	DeviceInfo SelectedDeviceInfo;
	int WebCamIndex = 0;

	nosResourceShareInfo _nosIntermediateTexture = {};
	nosResourceShareInfo _nosMemoryBuffer = {};
};

extern "C"
{


NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outCount, nosNodeFunctions** outFunctions)
{
	*outCount = (size_t)(1);
	if (!outFunctions)
		return NOS_RESULT_SUCCESS;

	NOS_RETURN_ON_FAILURE(RequestVulkanSubsystem());
		
	NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.webcam.WebcamIn"), WebcamInNode, outFunctions[0]);
	return NOS_RESULT_SUCCESS;
}

}
} // namespace nos::test
