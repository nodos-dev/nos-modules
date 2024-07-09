// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#pragma once

#include <Nodos/PluginHelpers.hpp>

// External
#include <glm/glm.hpp> // TODO: Ring no longer needs glm::mat4 colormatrix. Remove this
#include <nosVulkanSubsystem/Helpers.hpp>

#include "Ring.h"
#include "nosUtil/Stopwatch.hpp"

namespace nos::mediaio
{

template<typename T, bool CheckInterlace>
requires std::is_same_v<T, nosBufferInfo> || std::is_same_v<T, nosTextureInfo>
struct RingNodeBase : NodeContext
{
	enum class RingMode
	{
		CONSUME,
		FILL,
	};
	std::unique_ptr<TRing<T>> Ring = nullptr;
	
	// If copy ring, then full copy the slot resource to output pin & wait for copy before copying from input to slot
	// If download ring, then set the output pin to the slot resource, do not copy & wait until path execution finished to put the slot back in ring
	enum class RingType
	{
		COPY_RING,
		DOWNLOAD_RING
	} Type;

	// If reset, then reset the ring on path stop
	// If wait until full, do not output until ring is full & then start consuming
	enum class OnRestartType
	{
		RESET,
		WAIT_UNTIL_FULL
	} OnRestart;

	std::optional<uint32_t> RequestedRingSize = std::nullopt;
	bool NeedsRecreation = false;

	std::atomic_uint32_t SpareCount = 0;

	std::condition_variable ModeCV;
	std::mutex ModeMutex;
	std::atomic<RingMode> Mode = RingMode::CONSUME;

	nosTextureFieldType WantedField = NOS_TEXTURE_FIELD_TYPE_UNKNOWN;

	TRing<T>::Resource* LastPopped = nullptr;

	RingNodeBase(const nosFbNode* node, T baseInfo, RingType type, OnRestartType onRestart) : NodeContext(node), Type(type), OnRestart(onRestart)
	{
		Ring = std::make_unique<TRing<T>>(1, baseInfo);

		Ring->Stop();
		AddPinValueWatcher(NOS_NAME_STATIC("Size"), [this](nos::Buffer const& newSize, std::optional<nos::Buffer> oldVal) {
			uint32_t size = *newSize.As<uint32_t>();
			if (size == 0)
			{
				nosEngine.LogW((GetName() + " size cannot be 0").c_str());
				return;
			}
			if (Ring->Size != size && (!RequestedRingSize.has_value() || *RequestedRingSize != size))
			{
				nosPathCommand ringSizeChange{ .Event = NOS_RING_SIZE_CHANGE, .RingSize = size };
				nosEngine.SendPathCommand(PinName2Id[NOS_NAME_STATIC("Input")], ringSizeChange);
				SendPathRestart();
				RequestedRingSize = size;
				Ring->Stop();
			}
			});
		AddPinValueWatcher(NOS_NAME_STATIC("Input"), [this](nos::Buffer const& newBuf, std::optional<nos::Buffer> oldVal) {
			if constexpr (std::is_same_v<T, nosTextureInfo>)
			{
				auto info = vkss::DeserializeTextureInfo(newBuf.Data());
				if (Ring->Sample.Width != info.Info.Texture.Width ||
					Ring->Sample.Height != info.Info.Texture.Height ||
					Ring->Sample.Format != info.Info.Texture.Format)
				{
					Ring->Sample.Format = info.Info.Texture.Format;
					Ring->Sample.Width = info.Info.Texture.Width;
					Ring->Sample.Height = info.Info.Texture.Height;
					NeedsRecreation = true;
				}
			}
			else if (std::is_same_v<T, nosBufferInfo>)
			{
				auto info = vkss::ConvertToResourceInfo(*InterpretPinValue<sys::vulkan::Buffer>(newBuf.Data()));
				if (Ring->Sample.Size != info.Info.Buffer.Size)
				{
					Ring->Sample.Size = info.Info.Buffer.Size;
					NeedsRecreation = true;
				}
			}

			if (NeedsRecreation)
			{
				SendPathRestart();
				Ring->Stop();
			}

			});
		if constexpr (std::is_same_v<T, nosBufferInfo>)
		{
			AddPinValueWatcher(NOS_NAME_STATIC("Alignment"), [this](nos::Buffer const& newAlignment, std::optional<nos::Buffer> oldVal) {
				uint32_t alignment = *newAlignment.As<uint32_t>();
				if (Ring->Sample.Alignment == alignment)
					return;
				Ring->Sample.Alignment = alignment;
				NeedsRecreation = true;
				SendPathRestart();
				Ring->Stop();
			});
		}
	}

	virtual std::string GetName() const = 0;

	void SendRingStats() const
	{
		nosEngine.WatchLog((NodeName.AsString() + " Read Size").c_str(), std::to_string(Ring->Read.Pool.size()).c_str());
		nosEngine.WatchLog((NodeName.AsString() + " Write Size").c_str(), std::to_string(Ring->Write.Pool.size()).c_str());
		nosEngine.WatchLog((NodeName.AsString() + " Total Frame Count").c_str(), std::to_string(Ring->TotalFrameCount()).c_str());
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

		nosResourceShareInfo input = {};
		if constexpr (std::is_same_v<T, nosBufferInfo>)
		{
			input = vkss::ConvertToResourceInfo(*pins.GetPinData<sys::vulkan::Buffer>(NOS_NAME_STATIC("Input")));
		}
		else if constexpr (std::is_same_v<T, nosTextureInfo>)
		{
			input = vkss::DeserializeTextureInfo(pins[NOS_NAME_STATIC("Input")].Data->Data);
		}
		if (!input.Memory.Handle)
			return NOS_RESULT_FAILED;

		if (Ring->IsFull())
		{
			nosEngine.LogI("Trying to push while ring is full");
		}

		nos::util::Stopwatch sw; 
		auto slot = Ring->BeginPush();
		nosEngine.WatchLog((GetName() + " Begin Push").c_str(), nos::util::Stopwatch::ElapsedString(sw.Elapsed()).c_str());
		if constexpr (CheckInterlace)
		{
			nosTextureFieldType incomingField;
			if constexpr (std::is_same_v<T, nosBufferInfo>)
				incomingField = input.Info.Buffer.FieldType;
			else if constexpr (std::is_same_v<T, nosTextureInfo>)
				incomingField = input.Info.Texture.FieldType;

			if (WantedField == NOS_TEXTURE_FIELD_TYPE_UNKNOWN)
				WantedField = incomingField;
			
			auto outInterlaced = vkss::IsTextureFieldTypeInterlaced(WantedField);
			auto inInterlaced = vkss::IsTextureFieldTypeInterlaced(incomingField);
			if ((inInterlaced && outInterlaced) && incomingField != WantedField)
			{
				nosEngine.LogW("BufferRing: Field mismatch. Waiting for a new frame.");
				Ring->CancelPush(slot);
				SendScheduleRequest(0);
				return NOS_RESULT_FAILED;
			}
			if constexpr (std::is_same_v<T, nosBufferInfo>)
				slot->Res.Info.Buffer.FieldType = incomingField;
			else if constexpr (std::is_same_v<T, nosTextureInfo>)
				slot->Res.Info.Texture.FieldType = incomingField;

			WantedField = vkss::FlippedField(WantedField);
		}
		slot->FrameNumber = args->FrameNumber;
		if (slot->Params.WaitEvent)
		{
			nos::util::Stopwatch sw;
			nosVulkan->WaitGpuEvent(&slot->Params.WaitEvent, UINT64_MAX);
			auto elapsed = sw.Elapsed();
			nosEngine.WatchLog((GetName() + " Execute GPU Wait: " + NodeName.AsString()).c_str(),
				nos::util::Stopwatch::ElapsedString(elapsed).c_str());
		}
		nosCmd cmd;
		nosCmdBeginParams beginParams;
		if constexpr (std::is_same_v<T, nosTextureInfo>)
			beginParams = { NOS_NAME("BoundedTextureQueue: Input Texture Copy To Queue Slot"), NodeId, &cmd };
		else if constexpr (std::is_same_v<T, nosBufferInfo>)
			beginParams = { NOS_NAME("BufferRing: Input Buffer Copy To Ring Slot"), NodeId, &cmd };

		nosVulkan->Begin2(&beginParams);
		nosVulkan->Copy(cmd, &input, &slot->Res, 0);
		nosCmdEndParams end{ .ForceSubmit = NOS_TRUE, .OutGPUEventHandle = &slot->Params.WaitEvent };
		nosVulkan->End(cmd, &end);
		Ring->EndPush(slot);
		if (Mode == RingMode::FILL && Ring->IsFull())
		{
			Mode = RingMode::CONSUME;
			ModeCV.notify_all();
		}
		return NOS_RESULT_SUCCESS;
	}

	// Called from a different thread.
	nosResult CopyFrom(nosCopyInfo* cpy) override
	{
		if (LastPopped != nullptr)
		{
			DEBUG_BREAK
		}
		if (!Ring || Ring->Exit)
			return NOS_RESULT_FAILED;
		SendRingStats();
		if (Mode == RingMode::FILL)
		{
			//Sleep for 20 ms & if still Fill, return pending
			std::unique_lock<std::mutex> lock(ModeMutex);
			if(!ModeCV.wait_for(lock, std::chrono::milliseconds(100), [this] { return Mode != RingMode::FILL; }))
				return NOS_RESULT_PENDING;
		}

		auto effectiveSpareCount = SpareCount.load(); // TODO: * (1 + u32(th->Interlaced()));
		auto* slot = Ring->BeginPop(100);
		// If timeout or exit
		if (!slot)
			return Ring->Exit ? NOS_RESULT_FAILED : NOS_RESULT_PENDING;

		nosResourceShareInfo output;

		if constexpr (std::is_same_v<T, nosBufferInfo>)
		{
			auto outputBufferDesc = *static_cast<sys::vulkan::Buffer*>(cpy->PinData->Data);
			output = vkss::ConvertToResourceInfo(outputBufferDesc);
			if (slot->Res.Info.Buffer.Size != output.Info.Buffer.Size)
			{
				output.Memory = {};
				output.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
				output.Info.Buffer = slot->Res.Info.Buffer;
				nosEngine.SetPinValueByName(NodeId, NOS_NAME_STATIC("Output"), Buffer::From(vkss::ConvertBufferInfo(output)));
				outputBufferDesc = *static_cast<sys::vulkan::Buffer*>(cpy->PinData->Data);
				output = vkss::ConvertToResourceInfo(outputBufferDesc);
			}
		}
		else if constexpr (std::is_same_v<T, nosTextureInfo>)
		{
			auto outputTextureDesc = static_cast<sys::vulkan::Texture*>(cpy->PinData->Data); 
			output = vkss::DeserializeTextureInfo(outputTextureDesc);
			if (slot->Res.Info.Texture.Height != output.Info.Texture.Height ||
				slot->Res.Info.Texture.Width != output.Info.Texture.Width ||
				slot->Res.Info.Texture.Format != output.Info.Texture.Format)
			{
				output.Memory = {};
				output.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
				output.Info.Texture = slot->Res.Info.Texture;
				output.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_SRC | NOS_IMAGE_USAGE_TRANSFER_DST | NOS_IMAGE_USAGE_SAMPLED);

				sys::vulkan::TTexture texDef = vkss::ConvertTextureInfo(output);
				texDef.unscaled = true;

				nosEngine.SetPinValueByName(NodeId, NOS_NAME_STATIC("Output"), Buffer::From(texDef));

				outputTextureDesc = static_cast<sys::vulkan::Texture*>(cpy->PinData->Data);
				output = vkss::DeserializeTextureInfo(outputTextureDesc);
			}
		}
		if (slot->Params.WaitEvent)
		{
			nos::util::Stopwatch sw;
			nosVulkan->WaitGpuEvent(&slot->Params.WaitEvent, UINT64_MAX);
			auto elapsed = sw.Elapsed();
			nosEngine.WatchLog((GetName() + " Copy From GPU Wait: " + NodeName.AsString()).c_str(),
				nos::util::Stopwatch::ElapsedString(elapsed).c_str());
		}
		nosCmd cmd;
		nosCmdBeginParams beginParams;
		if constexpr (std::is_same_v<T, nosTextureInfo>)
			beginParams = { NOS_NAME("BoundedTextureQueue: Slot Copy To Output Texture"), NodeId, & cmd };
		else if constexpr (std::is_same_v<T, nosBufferInfo>)
			beginParams = { NOS_NAME("BufferRing: Ring Slot Copy To Output Buffer"), NodeId, & cmd };
		if (Type == RingType::COPY_RING)
		{
			nosVulkan->Begin2(&beginParams);
			nosVulkan->Copy(cmd, &slot->Res, &output, 0);
			nosCmdEndParams end{ .ForceSubmit = NOS_TRUE, .OutGPUEventHandle = &slot->Params.WaitEvent };
			nosVulkan->End(cmd, &end);
			if constexpr (std::is_same_v<T, nosBufferInfo>)
			{
				nosTextureFieldType outFieldType = slot->Res.Info.Buffer.FieldType;
				auto outputBufferDesc = *static_cast<sys::vulkan::Buffer*>(cpy->PinData->Data);
				outputBufferDesc.mutate_field_type((sys::vulkan::FieldType)outFieldType);
				nosEngine.SetPinValue(cpy->ID, nos::Buffer::From(outputBufferDesc));
			}
			else if constexpr (std::is_same_v<T, nosTextureInfo>)
			{
				nosTextureFieldType outFieldType = slot->Res.Info.Texture.FieldType;
				auto outputTextureDesc = static_cast<sys::vulkan::Texture*>(cpy->PinData->Data);
				auto output = vkss::DeserializeTextureInfo(outputTextureDesc);
				output.Info.Texture.FieldType = slot->Res.Info.Texture.FieldType;
				sys::vulkan::TTexture texDef = vkss::ConvertTextureInfo(output);
				texDef.unscaled = true;
				nosEngine.SetPinValue(cpy->ID, Buffer::From(texDef));
			}
		}
		else if (Type == RingType::DOWNLOAD_RING)
		{
			nosEngine.SetPinValue(cpy->ID, nos::Buffer::From(vkss::ConvertBufferInfo(slot->Res)));
		}

		cpy->CopyFromOptions.ShouldSetSourceFrameNumber = true;
		cpy->FrameNumber = slot->FrameNumber;
		

		if (Type == RingType::DOWNLOAD_RING)
			LastPopped = slot;
		else if (Type == RingType::COPY_RING)
		{
			Ring->EndPop(slot);
			SendScheduleRequest(1);
		}
		return NOS_RESULT_SUCCESS;
	}

	void OnPathCommand(const nosPathCommand* command) override
	{
		switch (command->Event)
		{
		case NOS_RING_SIZE_CHANGE: {
			RequestedRingSize = command->RingSize;
			nosEngine.SetPinValue(*GetPinId(NOS_NAME("Size")), nos::Buffer::From(command->RingSize));
			break;
		}
		default: return;
		}
	}

	void SendScheduleRequest(uint32_t count, bool reset = false) const
	{
		nosScheduleNodeParams schedule{
			.NodeId = NodeId,
			.AddScheduleCount = count,
			.Reset = reset
		};
		nosEngine.ScheduleNode(&schedule);
	}

	void OnPathStop() override
	{
		if(OnRestart == OnRestartType::WAIT_UNTIL_FULL)
			Mode = RingMode::FILL;
		if (Ring)
		{
			Ring->Stop();
		}
	}

	void OnPathStart() override
	{
		if (Ring && OnRestart == OnRestartType::RESET)
			Ring->Reset(false);
		if (RequestedRingSize)
		{
			Ring->Resize(*RequestedRingSize);
			RequestedRingSize = std::nullopt;
		}
		if (NeedsRecreation)
		{
			Ring = std::make_unique<TRing<T>>(Ring->Size, Ring->Sample);
			Ring->Exit = true;
			NeedsRecreation = false;
		}
		auto emptySlotCount = Ring->Write.Pool.size();
		nosScheduleNodeParams schedule{ .NodeId = NodeId, .AddScheduleCount = emptySlotCount };
		nosEngine.ScheduleNode(&schedule);
		if (emptySlotCount == 0)
			Mode = RingMode::CONSUME;
		Ring->Exit = false;
	}

	void OnPathExecutionFinished(nosUUID pinId, bool causedByCancel) override
	{
		if (Type != RingType::DOWNLOAD_RING)
			return;
		if (pinId == PinName2Id[NOS_NAME_STATIC("Output")])
		{
			if (!LastPopped)
				return;
			Ring->EndPop(LastPopped);
			LastPopped = nullptr;
			SendScheduleRequest(1);
		}
	}

	void SendPathRestart()
	{
		nosEngine.SendPathRestart(PinName2Id[NOS_NAME_STATIC("Input")]);
	}
};




}