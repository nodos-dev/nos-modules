#include <Nodos/PluginHelpers.hpp>

// External
#include <glm/glm.hpp> // TODO: Ring no longer needs glm::mat4 colormatrix. Remove this
#include <nosVulkanSubsystem/Helpers.hpp>

#include "Ring.h"
#include "nosUtil/Stopwatch.hpp"

NOS_REGISTER_NAME(Size)
NOS_REGISTER_NAME(Spare)

namespace nos::MediaIO
{
struct BufferRingNodeContext : NodeContext
{
	enum class RingMode
	{
		CONSUME,
		FILL,
	};
	std::unique_ptr<TRing<nosBufferInfo>> Ring = nullptr;
	uint32_t RequestedRingSize = 1;
	std::atomic_uint32_t SpareCount = 0;
	std::atomic<RingMode> Mode = RingMode::FILL;
	nosBufferInfo SampleBuffer =
		nosBufferInfo{ .Size = 1,
					  .Alignment = 0,
					  .Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_TRANSFER_DST),
					  .MemoryFlags = nosMemoryFlags(NOS_MEMORY_FLAGS_DOWNLOAD | NOS_MEMORY_FLAGS_HOST_VISIBLE)};
	BufferRingNodeContext(const nosFbNode* node) : NodeContext(node)
	{
		Ring = std::make_unique<TRing<nosBufferInfo>>(1, SampleBuffer);
		Ring->Stop();
		AddPinValueWatcher(NSN_Size, [this](nos::Buffer const& newSize, nos::Buffer const& oldSize, bool first) {
			uint32_t size = *newSize.As<uint32_t>();
			if (size == 0)
			{
				nosEngine.LogW("Ring size cannot be 0");
				return;
			}
			if (RequestedRingSize != size)
			{
				nosPathCommand ringSizeChange{.Event = NOS_RING_SIZE_CHANGE, .RingSize = size};
				nosEngine.SendPathCommand(PinName2Id[NOS_NAME_STATIC("Input")], ringSizeChange);
			}
		});
		AddPinValueWatcher(NOS_NAME("Alignment"), [this](nos::Buffer const& newAlignment, nos::Buffer const& oldSize, bool first) {
			uint32_t alignment = *newAlignment.As<uint32_t>();
			if (SampleBuffer.Alignment == alignment)
				return;
			SampleBuffer.Alignment = alignment;
			Ring = std::make_unique<TRing<nosBufferInfo>>(Ring->Size, SampleBuffer);
			Ring->Stop();
			SendPathRestart();
		});
		AddPinValueWatcher(NOS_NAME("Input"), [this](nos::Buffer const& newBuf, nos::Buffer const& oldBuf, bool first) {
			auto info = vkss::ConvertToResourceInfo(*InterpretPinValue<sys::vulkan::Buffer>(newBuf.Data()));
			if (Ring->Sample.Size != info.Info.Buffer.Size)
			{
				SampleBuffer.Size = info.Info.Buffer.Size; 
				Ring = std::make_unique<TRing<nosBufferInfo>>(Ring->Size, SampleBuffer);
				Ring->Stop();
				SendPathRestart();
			}
		});
	}

	void SendRingStats() const
	{
		nosEngine.WatchLog(("Ring: " + NodeName.AsString() + " Read Size").c_str(), std::to_string(Ring->Read.Pool.size()).c_str());
		nosEngine.WatchLog(("Ring: " + NodeName.AsString() + " Write Size").c_str(), std::to_string(Ring->Write.Pool.size()).c_str());
		nosEngine.WatchLog(("Ring: " + NodeName.AsString() + " Total Frame Count").c_str(), std::to_string(Ring->TotalFrameCount()).c_str());
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		if (Ring->Exit || Ring->Size == 0)
			return NOS_RESULT_FAILED;
		NodeExecuteArgs pins(args);
		SpareCount = *pins.GetPinData<uint32_t>(NSN_Spare);
		if (SpareCount >= Ring->Size)
		{
			uint32_t newSpareCount = Ring->Size - 1; 
			SpareCount = newSpareCount;
			nosEngine.LogW("Spare count must be less than ring size! Capping spare count at %u.", newSpareCount);
			nosEngine.SetPinValueByName(NodeId, NSN_Spare, nosBuffer{.Data = &newSpareCount, .Size = sizeof(newSpareCount)});
		}
		auto* output = pins.GetPinData<sys::vulkan::Buffer>(NOS_NAME_STATIC("Output"));
		nosResourceShareInfo input = vkss::ConvertToResourceInfo(*pins.GetPinData<sys::vulkan::Buffer>(NOS_NAME_STATIC("Input")));
		if (!input.Memory.Handle)
			return NOS_RESULT_FAILED;

		if (Ring->IsFull())
		{
			nosEngine.LogI("Trying to push while ring is full");
		}

		auto slot = Ring->BeginPush();
		// TODO: FieldType
		slot->FrameNumber = args->FrameNumber;
		if (slot->Params.WaitEvent)
		{
			nos::util::Stopwatch sw;
			nosVulkan->WaitGpuEvent(&slot->Params.WaitEvent, UINT64_MAX);
			auto elapsed = sw.Elapsed();
			nosEngine.WatchLog(("Ring Execute GPU Wait: " + NodeName.AsString()).c_str(),
							   nos::util::Stopwatch::ElapsedString(elapsed).c_str());
		}
		nosCmd cmd;
		nosCmdBeginParams beginParams {NOS_NAME("BufferRing: Input Buffer Copy To Ring Slot"), NodeId, &cmd};
		nosVulkan->Begin2(&beginParams);
		nosVulkan->Copy(cmd, &input, &slot->Res, 0);
		nosCmdEndParams end{.ForceSubmit = NOS_TRUE, .OutGPUEventHandle = &slot->Params.WaitEvent};
		nosVulkan->End(cmd, &end);
		Ring->EndPush(slot);
		if (Mode == RingMode::FILL && Ring->IsFull())
			Mode = RingMode::CONSUME;
		return NOS_RESULT_SUCCESS;
	}

	// Called from a different thread.
	nosResult CopyFrom(nosCopyInfo* cpy) override
	{
		if (!Ring || Ring->Exit)
			return NOS_RESULT_FAILED;
		SendRingStats();
		if (Mode == RingMode::FILL)
			return NOS_RESULT_PENDING;

		auto outputBufferDesc = static_cast<sys::vulkan::Buffer*>(cpy->PinData->Data);
		auto output = vkss::ConvertToResourceInfo(*outputBufferDesc);
		auto effectiveSpareCount = SpareCount.load(); // TODO: * (1 + u32(th->Interlaced()));
		auto* slot = Ring->BeginPop();
		if (!slot)
			return Ring->Exit ? NOS_RESULT_FAILED : NOS_RESULT_PENDING;
		if (slot->Res.Info.Buffer.Size != output.Info.Buffer.Size)
		{
			output.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
			output.Info.Buffer = slot->Res.Info.Buffer;
			nosEngine.SetPinValueByName(NodeId, NOS_NAME_STATIC("Output"), Buffer::From(vkss::ConvertBufferInfo(output)));
			outputBufferDesc = static_cast<sys::vulkan::Buffer*>(cpy->PinData->Data);
			output = vkss::ConvertToResourceInfo(*outputBufferDesc);
		}
		if (slot->Params.WaitEvent)
		{
			nos::util::Stopwatch sw;
			nosVulkan->WaitGpuEvent(&slot->Params.WaitEvent, UINT64_MAX);
			auto elapsed = sw.Elapsed();
			nosEngine.WatchLog(("Ring Copy From GPU Wait: " + NodeName.AsString()).c_str(),
							   nos::util::Stopwatch::ElapsedString(elapsed).c_str());
		}
		nosCmd cmd;
		nosCmdBeginParams beginParams {NOS_NAME("BufferRing: Ring Slot Copy To Output Buffer"), NodeId, &cmd};
		nosVulkan->Begin2(&beginParams);
		nosVulkan->Copy(cmd, &slot->Res, &output, 0);
		nosCmdEndParams end{.ForceSubmit = NOS_TRUE, .OutGPUEventHandle = &slot->Params.WaitEvent};
		nosVulkan->End(cmd, &end);
		//nosEngine.SetPinValueDirect(cpy->ID, Buffer::From(vkss::ConvertBufferInfo(slot->Res)));
		Ring->EndPop(slot);
		SendScheduleRequest(1);
		return NOS_RESULT_SUCCESS;
	}

	void OnPathCommand(const nosPathCommand* command) override
	{
		switch (command->Event)
		{
		case NOS_RING_SIZE_CHANGE: {
			if (command->RingSize == 0)
			{
				nosEngine.LogW("Ring size cannot be 0");
				return;
			}
			if (RequestedRingSize != command->RingSize)
			{
				RequestedRingSize = command->RingSize;
				SendPathRestart();
			}
			break;
		}
		default: return;
		}
	}

	void SendScheduleRequest(uint32_t count, bool reset = false) const
	{
		nosScheduleNodeParams schedule {
			.NodeId = NodeId,
			.AddScheduleCount = count,
			.Reset = reset
		};
		nosEngine.ScheduleNode(&schedule);
	}

	void OnPathStop() override
	{
		Mode = RingMode::FILL;
		if (Ring)
			Ring->Stop();
	}

	void OnPathStart() override
	{
		assert(RequestedRingSize != 0);
		if (RequestedRingSize != Ring->Size)
			Ring->Resize(RequestedRingSize);
		auto emptySlotCount = Ring->Write.Pool.size();
		nosScheduleNodeParams schedule{.NodeId = NodeId, .AddScheduleCount = emptySlotCount};
		nosEngine.ScheduleNode(&schedule);
		if (emptySlotCount == 0)
			Mode = RingMode::CONSUME;
		Ring->Exit = false;
	}

	void SendPathRestart()
	{
		nosEngine.SendPathRestart(PinName2Id[NOS_NAME_STATIC("Input")]);
	}
};

nosResult RegisterBufferRing(nosNodeFunctions* functions)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("BufferRing"), BufferRingNodeContext, functions)
	return NOS_RESULT_SUCCESS;
}

}