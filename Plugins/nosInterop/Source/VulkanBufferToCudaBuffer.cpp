#include <Nodos/PluginHelpers.hpp>
#include <nosVulkanSubsystem/Helpers.hpp>
#include <nosVulkanSubsystem/Types_generated.h>
#include <nosCUDASubsystem/nosCUDASubsystem.h>
#include <nosCUDASubsystem/Types_generated.h>
#include "InteropCommon.h"
#include "InteropNames.h"

struct VulkanBufferToCUDABuffer : nos::NodeContext
{
	BufferPin VulkanBufferPinProxy = {};
	nosResourceShareInfo Buffer = {};
	nosUUID NodeUUID = {}, InputBufferUUID = {}, OutputBufferUUID = {};
	nosCUDABufferInfo CUDABuffer = {};
	VulkanBufferToCUDABuffer(nosFbNode const* node) : NodeContext(node)
	{
		NodeUUID = *node->id();

		for (const auto& pin : *node->pins()) {
			if (NSN_InputBuffer.Compare(pin->name()->c_str()) == 0) {
				InputBufferUUID = *pin->id();
			}
			else if (NSN_OutputBuffer.Compare(pin->name()->c_str()) == 0) {
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



	void UpdateOutputPin(const nos::sys::vulkan::Buffer* VulkanBuf) {

		if (VulkanBufferPinProxy.Address == VulkanBuf->handle()) {
			return;
		}

		nos::sys::cuda::Buffer buffer;
		buffer.mutate_element_type((nos::sys::cuda::BufferElementType)VulkanBuf->element_type());
		buffer.mutate_handle(CUDABuffer.Address);
		buffer.mutate_size_in_bytes(VulkanBuf->size_in_bytes());
		buffer.mutate_offset(VulkanBuf->offset());

		VulkanBufferPinProxy.Address = VulkanBuf->handle();
		
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

