#include <Nodos/PluginHelpers.hpp>

// External
#include <glm/glm.hpp> // TODO: Ring no longer needs glm::mat4 colormatrix. Remove this
#include <nosVulkanSubsystem/Helpers.hpp>

#include "Ring.h"

namespace nos::utilities
{
struct BoundedTextureQueueNodeContext : NodeContext
{
	enum class RingMode
	{
		CONSUME,
		FILL,
	};
	std::unique_ptr<TRing<nosTextureInfo>> Ring = nullptr;
	uint32_t RequestedRingSize = 1;
	std::atomic_uint32_t SpareCount = 0;
	std::atomic<RingMode> Mode = RingMode::CONSUME;

	BoundedTextureQueueNodeContext(const nosFbNode* node) : NodeContext(node)
	{
		Ring = std::make_unique<TRing<nosTextureInfo>>(nosVec2u(1920, 1080), 1, 
			nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_SRC | NOS_IMAGE_USAGE_TRANSFER_DST), NOS_FORMAT_R16G16B16A16_SFLOAT);

		Ring->Stop();
		AddPinValueWatcher(NOS_NAME_STATIC("Size"), [this](nos::Buffer const& newSize, nos::Buffer const& oldSize, bool first) {
			uint32_t size = *newSize.As<uint32_t>();
			if (size == 0)
			{
				nosEngine.LogW("Bounded Queue size cannot be 0");
				return;
			}
			if (RequestedRingSize != size)
			{
				nosPathCommand ringSizeChange{.Event = NOS_RING_SIZE_CHANGE, .RingSize = size};
				nosEngine.SendPathCommand(PinName2Id[NOS_NAME_STATIC("Input")], ringSizeChange);
				SendPathRestart();
				RequestedRingSize = size;
			}
		});
		AddPinValueWatcher(NOS_NAME_STATIC("Input"), [this](nos::Buffer const& newBuf, nos::Buffer const& oldBuf, bool first) {
			auto info = vkss::DeserializeTextureInfo(newBuf.Data());
			if (Ring->Sample.Width != info.Info.Texture.Width ||
			Ring->Sample.Height != info.Info.Texture.Height ||
			Ring->Sample.Format != info.Info.Texture.Format)
			{
				Ring = std::make_unique<TRing<nosTextureInfo>>(nosVec2u(info.Info.Texture.Width, info.Info.Texture.Height), Ring->Size, 
					nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_SRC | NOS_IMAGE_USAGE_TRANSFER_DST), info.Info.Texture.Format);
				Ring->Stop();
				SendPathRestart();
			}
		});
	}

	void SendRingStats() const
	{
		nosEngine.WatchLog(("Bounded Queue: " + NodeName.AsString() + " Read Size").c_str(), std::to_string(Ring->Read.Pool.size()).c_str());
		nosEngine.WatchLog(("Bounded Queue: " + NodeName.AsString() + " Write Size").c_str(), std::to_string(Ring->Write.Pool.size()).c_str());
		nosEngine.WatchLog(("Bounded Queue: " + NodeName.AsString() + " Total Frame Count").c_str(), std::to_string(Ring->TotalFrameCount()).c_str());
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		if (Ring->Exit || Ring->Size == 0)
			return NOS_RESULT_FAILED;
		NodeExecuteArgs pins(args);

		//TODO: AJA spare count
		//SpareCount = *pins.GetPinData<uint32_t>(NSN_Spare);
		//if (SpareCount >= Ring->Size)
		//{
		//	uint32_t newSpareCount = Ring->Size - 1; 
		//	SpareCount = newSpareCount;
		//	nosEngine.LogW("Spare count must be less than ring size! Capping spare count at %u.", newSpareCount);
		//	nosEngine.SetPinValueByName(NodeId, NSN_Spare, nosBuffer{.Data = &newSpareCount, .Size = sizeof(newSpareCount)});
		//}
		//auto* output = pins.GetPinData<sys::vulkan::Texture>(NOS_NAME_STATIC("Output"));
		//nosResourceShareInfo input = vkss::ConvertToResourceInfo(*pins.GetPinData<sys::vulkan::Buffer>(NOS_NAME_STATIC("Input")));
		
		auto input = vkss::DeserializeTextureInfo(pins[NOS_NAME_STATIC("Input")].Data->Data);
		if (!input.Memory.Handle)
			return NOS_RESULT_FAILED;

		if (Ring->IsFull())
		{
			nosEngine.LogI("Trying to push while ring is full");
		}

		auto slot = Ring->BeginPush();
		// TODO: FieldType
		slot->FrameNumber = args->FrameNumber;
		nosCmd cmd;
		nosCmdBeginParams params {NOS_NAME("BoundedTextureQueue: Input Buffer Copy To Queue Slot"), NodeId, &cmd};
		nosVulkan->Begin2(&params);
		nosVulkan->Copy(cmd, &input, &slot->Res, 0);
		nosCmdEndParams end{.ForceSubmit = NOS_TRUE, .OutGPUEventHandle = &slot->Params.WaitEvent};
		nosVulkan->End(cmd, &end);
		nosVulkan->WaitGpuEvent(&slot->Params.WaitEvent, UINT64_MAX);
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

		auto outputTextureDesc = static_cast<sys::vulkan::Texture*>(cpy->PinData->Data);
		auto output = vkss::DeserializeTextureInfo(outputTextureDesc);
		//auto effectiveSpareCount = SpareCount.load(); // TODO: * (1 + u32(th->Interlaced()));
		auto* slot = Ring->BeginPop();
		if (!slot)
			return Ring->Exit ? NOS_RESULT_FAILED : NOS_RESULT_PENDING;
		if (slot->Res.Info.Texture.Height != output.Info.Texture.Height || 
			slot->Res.Info.Texture.Width != output.Info.Texture.Width || 
			slot->Res.Info.Texture.Format != output.Info.Texture.Format)
		{
			output.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
			output.Info.Texture = slot->Res.Info.Texture;
			output.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_SRC | NOS_IMAGE_USAGE_TRANSFER_DST | NOS_IMAGE_USAGE_SAMPLED);

			sys::vulkan::TTexture texDef = vkss::ConvertTextureInfo(output);
			texDef.unscaled = true;

			nosEngine.SetPinValueByName(NodeId, NOS_NAME_STATIC("Output"), Buffer::From(texDef));

			outputTextureDesc = static_cast<sys::vulkan::Texture*>(cpy->PinData->Data);
			output = vkss::DeserializeTextureInfo(outputTextureDesc);
		}
		nosCmd cmd;
		nosCmdBeginParams params {NOS_NAME("BoundedTextureQueue: Slot Copy To Output Texture"), NodeId, &cmd};
		nosVulkan->Begin2(&params);
		nosVulkan->Copy(cmd, &slot->Res, &output, 0);
		nosCmdEndParams end{ .ForceSubmit = NOS_TRUE, .OutGPUEventHandle = &slot->Params.WaitEvent };
		nosVulkan->End(cmd, &end);
		nosVulkan->WaitGpuEvent(&slot->Params.WaitEvent, UINT64_MAX);
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
				nosEngine.LogW("Bounded Queue size cannot be 0");
				return;
			}
			if (RequestedRingSize != command->RingSize)
				RequestedRingSize = command->RingSize;
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
		//Do not fill the queue
		//Mode = RingMode::FILL;
		if (Ring)
		{
			Ring->Reset(false);
			Ring->Stop();
		}
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

nosResult RegisterBoundedTextureQueue(nosNodeFunctions* functions)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("BoundedTextureQueue"), BoundedTextureQueueNodeContext, functions)
	return NOS_RESULT_SUCCESS;
}

}