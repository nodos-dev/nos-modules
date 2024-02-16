#include <Nodos/PluginHelpers.hpp>
#include <nosVulkanSubsystem/Helpers.hpp>
#include <nosVulkanSubsystem/Types_generated.h>
#include <nosCUDASubsystem/nosCUDASubsystem.h>
#include <nosCUDASubsystem/Types_generated.h>
#include "InteropCommon.h"
#include "InteropNames.h"

struct VulkanBufferToCUDABuffer : nos::NodeContext
{
	BufferPin VulkanBufferPinProxy = {}, CUDABufferPinProxy = {};
	nosResourceShareInfo Buffer = {};
	nosUUID NodeUUID = {}, InputBufferUUID = {}, OutputBufferUUID = {};
	nosCUDABufferInfo CUDABuffer = {};
	VulkanBufferToCUDABuffer(nosFbNode const* node) : NodeContext(node)
	{
		NodeUUID = *node->id();

		for (const auto& pin : *node->pins()) {
			if (NSN_InputBuffer.Compare(pin->name()->c_str())) {
				InputBufferUUID = *pin->id();
			}
			else if (NSN_OutputBuffer.Compare(pin->name()->c_str())) {
				OutputBufferUUID = *pin->id();
			}
		}
	}

	void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer value) override
	{
		if (InputBufferUUID == pinId) {
			auto* VulkanBuf = flatbuffers::GetRoot<nos::sys::vulkan::Buffer>(value.Data);
			if (VulkanBuf->handle() == CUDABuffer.CreateInfo.ImportedExternalHandle) {
				//Resource already imported
				return;
			}
			nosResult res = nosCUDA->ImportExternalMemoryAsCUDABuffer(VulkanBuf->external_memory().handle(), VulkanBuf->size_in_bytes(), VulkanBuf->size_in_bytes(), VulkanBuf->offset(), EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUEWIN32, &CUDABuffer);
			if (res != NOS_RESULT_SUCCESS) {
				nosEngine.LogE("Import from Vulkan to CUDA failed!");
				return;
			}
			UpdateOutputPin(VulkanBuf);
		}
	}
	

	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		return NOS_RESULT_SUCCESS;
	}



	void UpdateOutputPin(const nos::sys::vulkan::Buffer* vulkanBuf) {

		if (VulkanBufferPinProxy.Address == vulkanBuf->handle()) {
			return;
		}

		nos::sys::cuda::Buffer buffer;
		buffer.mutate_element_type(CUDABufferPinProxy.Element.CUDAElementType);
		buffer.mutate_handle(CUDABufferPinProxy.Address);
		buffer.mutate_size_in_bytes(CUDABufferPinProxy.Size);
		buffer.mutate_offset(CUDABufferPinProxy.Offset);

		auto bufPin = nos::Buffer::From(buffer);

		nosEngine.SetPinValue(OutputBufferUUID, { .Data = &bufPin, .Size = sizeof(bufPin.Size()) });

		return;
	}

};

nosResult RegisterVulkanBufferToCUDABuffer(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NSN_VulkanBufferToCUDABuffer, VulkanBufferToCUDABuffer, fn);
	return NOS_RESULT_SUCCESS;
}

