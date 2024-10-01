/*
 * Copyright MediaZ Teknoloji A.S. All Rights Reserved.
 */

#pragma once
#include <Nodos/PluginHelpers.hpp>

 // External
#include <nosVulkanSubsystem/Helpers.hpp>

#include "nosUtil/Stopwatch.hpp"

namespace nos
{

enum class ResourceType {
	Buffer,
	Texture,
	Generic
};


struct TRing
{
    nos::Buffer Sample;
	ResourceType type = ResourceType::Generic;
    
    struct Resource
    {
        nosResourceShareInfo Res;
	    struct {
		    nosTextureFieldType FieldType = NOS_TEXTURE_FIELD_TYPE_UNKNOWN;
			glm::mat4 ColorspaceMatrix = {};
			nosGPUEvent WaitEvent = 0;
	    } Params {};

        Resource(nosBuffer r, ResourceType type) : Res{}
        {
            if (type == ResourceType::Buffer)
            {
                Res.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
                Res.Info.Buffer = vkss::ConvertToResourceInfo(*InterpretPinValue<sys::vulkan::Buffer>(r.Data)).Info.Buffer;
            }
            else if (type == ResourceType::Texture)
            {
                Res.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
                Res.Info.Texture = vkss::DeserializeTextureInfo(r.Data).Info.Texture;
            }
			else {
				assert(0);
			}
            nosVulkan->CreateResource(&Res);
        }
        
        ~Resource()
        { 
            if (Params.WaitEvent)
				nosVulkan->WaitGpuEvent(&Params.WaitEvent, UINT64_MAX);
            nosVulkan->DestroyResource(&Res);
        }

        void Reset()
		{
			if (Params.WaitEvent)
				nosVulkan->WaitGpuEvent(&Params.WaitEvent, UINT64_MAX);
            Params = {};
			FrameNumber = 0;
        }

        std::atomic_uint64_t FrameNumber;
    };

    void Resize(u32 size)
    {
        Write.Pool = {};
        Read.Pool = {};
        Resources.clear();
        for (u32 i = 0; i < size; ++i)
		{
            auto res = MakeShared<Resource>(Sample, type);
			Resources.push_back(res);
            Write.Pool.push_back(res.get());
        }
        Size = size;
    }
    
    TRing(u32 ringSize, ResourceType resourceType, nosBuffer resourceSample) : type(resourceType), Sample(resourceSample)
    {
        Resize(ringSize);
    }

    struct
    {
        std::deque<Resource *> Pool;
        std::mutex Mutex;
        std::condition_variable CV;
    } Write, Read;

    std::vector<rc<Resource>> Resources;

    u32 Size = 0;
    nosVec2u Extent;
    std::atomic_bool Exit = false;
    std::atomic_bool ResetFrameCount = true;

    ~TRing()
    {
        Stop();
        Resources.clear();
    }

    void Stop()
    {
        {
            std::unique_lock l1(Write.Mutex);
            std::unique_lock l2(Read.Mutex);
            Exit = true;
        }
		Write.CV.notify_all();
		Read.CV.notify_all();
    }

    bool IsFull()
    {
        std::unique_lock lock(Read.Mutex);
		return Read.Pool.size() == Resources.size(); 
    }

	bool HasEmptySlots()
	{
		return EmptyFrames() != 0;
	}

	u32 EmptyFrames()
	{
		std::unique_lock lock(Write.Mutex);
		return Write.Pool.size();
	}

    bool IsEmpty()
    {
        std::unique_lock lock(Read.Mutex);
        return Read.Pool.empty();
    }

    u32 ReadyFrames()
    {
        std::unique_lock lock(Read.Mutex);
        return Read.Pool.size();
    }

    u32 TotalFrameCount()
    {
        std::unique_lock lock(Write.Mutex);
        return Size - Write.Pool.size();
    }

    Resource *BeginPush()
    {
        std::unique_lock lock(Write.Mutex);
        while (Write.Pool.empty() && !Exit)
        {
            Write.CV.wait(lock);
        }
        if (Exit)
            return 0;
        Resource *res = Write.Pool.front();
        Write.Pool.pop_front();
        return res;
    }

    void EndPush(Resource *res)
    {
        {
            std::unique_lock lock(Read.Mutex);
            Read.Pool.push_back(res);
			assert(Read.Pool.size() <= Resources.size());
        }
        Read.CV.notify_one();
    }

    void CancelPush(Resource* res)
	{
		{
			std::unique_lock lock(Write.Mutex);
			res->FrameNumber = 0;
			Write.Pool.push_front(res);
			assert(Write.Pool.size() <= Resources.size());
		}
		Write.CV.notify_one();
	}
	void CancelPop(Resource* res)
	{
		{
			std::unique_lock lock(Read.Mutex);
			Read.Pool.push_front(res);
			assert(Read.Pool.size() <= Resources.size());
		}
		Read.CV.notify_one();
	}

    Resource *BeginPop(uint64_t timeoutMilliseconds)
    {
        std::unique_lock lock(Read.Mutex);
        if (!Read.CV.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds), [this]() {return !Read.Pool.empty() || Exit; }))
            return 0;
        if (Exit)
            return 0;
        auto res = Read.Pool.front();
        Read.Pool.pop_front();
        return res;
    }

    void EndPop(Resource *res)
    {
        {
            std::unique_lock lock(Write.Mutex);
            res->FrameNumber = 0;
            Write.Pool.push_back(res);
			assert(Write.Pool.size() <= Resources.size());
        }
        Write.CV.notify_one();
    }

    bool CanPop(u64& frameNumber, u32 spare = 0)
    {
        std::unique_lock lock(Read.Mutex);
        if (Read.Pool.size() > spare)
        {
        	// TODO: Under current arch, schedule requests are sent for the node instead of pin, so this code shouldn't be needed, but check.
            // auto newFrameNumber = Read.Pool.front()->FrameNumber.load();
            // bool result = ResetFrameCount || !frameNumber || newFrameNumber > frameNumber;
            // frameNumber = newFrameNumber;
            // ResetFrameCount = false;
            return true;
        }

        return false;
    }

    bool CanPush()
    {
        std::unique_lock lock(Write.Mutex);
        return !Write.Pool.empty();
    }

    Resource *TryPush()
    {
        if (CanPush())
            return BeginPush();
        return 0;
    }

    Resource *TryPush(const std::chrono::milliseconds timeout)
    {
		{
            std::unique_lock lock(Write.Mutex);
		    if (Write.Pool.empty())
                Write.CV.wait_for(lock, timeout, [&]{ return CanPush(); });
		}
		return TryPush();
    }

    Resource *TryPop(u64& frameNumber, u32 spare = 0)
    {
        if (CanPop(frameNumber, spare))
            return BeginPop(20);
        return 0;
	}

    void Reset(bool fill)
    {
        auto& from = fill ? Write : Read;
		auto& to = fill ? Read : Write;
		std::unique_lock l1(Write.Mutex);
		std::unique_lock l2(Read.Mutex);
		while (!from.Pool.empty())
		{
			auto* slot = from.Pool.front();
			from.Pool.pop_front();
			slot->Reset();
			to.Pool.push_back(slot);
		}
    }
};

struct RingNodeBase : NodeContext
{
	enum class RingMode
	{
		CONSUME,
		FILL,
	};
	std::unique_ptr<TRing> Ring = nullptr;
	ResourceType type = ResourceType::Generic;

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

	TRing::Resource* LastPopped = nullptr;

	TypeInfo typeInfo;
	void Init() {
		nosBuffer sample;
		nosEngine.GetDefaultValueOfType(typeInfo->TypeName, &sample);
		Ring = std::make_unique<TRing>(1, type, sample);

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
			if (type == ResourceType::Texture)
			{
				auto info = vkss::DeserializeTextureInfo(newBuf.Data());
				auto textureInfo = vkss::ConvertTextureInfo(vkss::DeserializeTextureInfo(Ring->Sample.Data()));
				if (textureInfo.format != (nos::sys::vulkan::Format)info.Info.Texture.Format ||
					textureInfo.height != info.Info.Texture.Height ||
					textureInfo.width != info.Info.Texture.Width)
				{
					textureInfo.format = (nos::sys::vulkan::Format)info.Info.Texture.Format;
					textureInfo.width = info.Info.Texture.Width;
					textureInfo.height = info.Info.Texture.Height;
					NeedsRecreation = true;
				}
				Ring->Sample = Buffer::From(textureInfo);
			}
			else if (type == ResourceType::Buffer)
			{
				auto info = vkss::ConvertToResourceInfo(*InterpretPinValue<sys::vulkan::Buffer>(newBuf.Data())).Info.Buffer;
				auto sampleInfo = vkss::ConvertBufferInfo(vkss::ConvertToResourceInfo(*InterpretPinValue<sys::vulkan::Buffer>(Ring->Sample.Data())));
				if (sampleInfo.size_in_bytes() != info.Size)
				{
					sampleInfo.mutate_size_in_bytes(info.Size);
					NeedsRecreation = true;
				}
			}

			if (NeedsRecreation)
			{
				SendPathRestart();
				Ring->Stop();
			}

			});
		if (type == ResourceType::Buffer)
		{
			AddPinValueWatcher(NOS_NAME_STATIC("Alignment"), [this](nos::Buffer const& newAlignment, std::optional<nos::Buffer> oldVal) {
				uint32_t alignment = *newAlignment.As<uint32_t>();
				auto sampleInfo = vkss::ConvertToResourceInfo(*InterpretPinValue<sys::vulkan::Buffer>(Ring->Sample.Data())).Info.Buffer;
				if (sampleInfo.Alignment == alignment)
					return;
				sampleInfo.Alignment = alignment;
				NeedsRecreation = true;
				SendPathRestart();
				Ring->Stop();
				});
		}
	}

	RingNodeBase(const nosFbNode* node, OnRestartType onRestart) : NodeContext(node), OnRestart(onRestart), typeInfo(NOS_NAME_STATIC("nos.fb.Void")) {
		
	}

	virtual std::string GetName() const = 0;

	void SendRingStats() const
	{
		nosEngine.WatchLog((NodeName.AsString() + " Read Size").c_str(), std::to_string(Ring->Read.Pool.size()).c_str());
		nosEngine.WatchLog((NodeName.AsString() + " Write Size").c_str(), std::to_string(Ring->Write.Pool.size()).c_str());
		nosEngine.WatchLog((NodeName.AsString() + " Total Frame Count").c_str(), std::to_string(Ring->TotalFrameCount()).c_str());
	}

	nosResult OnResolvePinDataTypes(nosResolvePinDataTypesParams* params) override
	{
		if (typeInfo.TypeName != NOS_NAME_STATIC("nos.fb.Void"))
			return NOS_RESULT_FAILED;
		
		typeInfo = TypeInfo(params->IncomingTypeName);
		if (typeInfo->TypeName == NOS_NAME_STATIC("nos.sys.vulkan.Buffer"))
			type = ResourceType::Buffer;
		else if (typeInfo->TypeName == NOS_NAME_STATIC("nos.sys.vulkan.Texture"))
			type = ResourceType::Texture;
		else
			return NOS_RESULT_FAILED;

		for (size_t i = 0; i < params->PinCount; i++)
		{
			auto& pinInfo = params->Pins[i];
			std::string pinName = nosEngine.GetString(pinInfo.Name);
			if (pinName == "Input" || pinName == "Output")
				pinInfo.OutResolvedTypeName = typeInfo->TypeName;
		}

		return NOS_RESULT_SUCCESS;
	}

	void OnPinUpdated(const nosPinUpdate* pinUpdate) {
		if (typeInfo->TypeName == NOS_NAME_STATIC("nos.fb.Void") || Ring)
			return;

		Init();
	}

	nosResult ExecuteRingNode(nosNodeExecuteParams* params, bool useSlotEvent, nosName ringExecuteName, bool rejectFieldMismatch)
	{
		if (Ring->Exit || Ring->Size == 0|| !typeInfo)
			return NOS_RESULT_FAILED;

		NodeExecuteParams pins(params);

		auto it = pins.find(NOS_NAME_STATIC("Input"));
		assert(it != pins.end());
		auto& inputPin = it->second;
		assert(inputPin.Data);

		nosResourceShareInfo input = {};
		nosName inputName = NOS_NAME_STATIC("Input");
		switch (type)
		{
		case nos::ResourceType::Buffer:
			input = vkss::ConvertToResourceInfo(*pins.GetPinData<sys::vulkan::Buffer>(inputName));
			break;
		case nos::ResourceType::Texture:
			input = vkss::DeserializeTextureInfo(pins[inputName].Data->Data);
			break;
		case nos::ResourceType::Generic:
			return NOS_RESULT_FAILED;
		}

		if (!input.Memory.Handle)
			return NOS_RESULT_FAILED;

		if (Ring->IsFull())
		{
			nosEngine.LogI("Trying to push while ring is full");
		}

		typename TRing::Resource* slot = nullptr;
		{
			nos::util::Stopwatch sw;
			ScopedProfilerEvent({ .Name = "Wait For Empty Slot" });
			slot = Ring->BeginPush();
			nosEngine.WatchLog((GetName() + " Begin Push").c_str(), nos::util::Stopwatch::ElapsedString(sw.Elapsed()).c_str());
		}

		nosTextureFieldType incomingField;
		switch (type) {
		case ResourceType::Buffer:
			incomingField = input.Info.Buffer.FieldType;
			break;
		case ResourceType::Texture:
			incomingField = input.Info.Texture.FieldType;
			break;
		case ResourceType::Generic:
		default:
			return NOS_RESULT_FAILED;
		}


		if (rejectFieldMismatch)
		{
			if (WantedField == NOS_TEXTURE_FIELD_TYPE_UNKNOWN)
				WantedField = incomingField;

			auto outInterlaced = vkss::IsTextureFieldTypeInterlaced(WantedField);
			auto inInterlaced = vkss::IsTextureFieldTypeInterlaced(incomingField);
			if ((inInterlaced && outInterlaced) && incomingField != WantedField)
			{
				nosEngine.LogW("Field mismatch. Waiting for a new frame.");
				Ring->CancelPush(slot);
				SendScheduleRequest(0);
				return NOS_RESULT_FAILED;
			}
			WantedField = vkss::FlippedField(WantedField);
		}
		if (type == ResourceType::Buffer)
			slot->Res.Info.Buffer.FieldType = incomingField;
		else if (type == ResourceType::Texture)
			slot->Res.Info.Texture.FieldType = incomingField;
		else
			return NOS_RESULT_FAILED;
		slot->FrameNumber = params->FrameNumber;
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
		beginParams = { ringExecuteName, NodeId, &cmd };

		nosVulkan->Begin2(&beginParams);
		nosVulkan->Copy(cmd, &input, &slot->Res, 0);
		nosCmdEndParams end{ .ForceSubmit = NOS_TRUE, .OutGPUEventHandle = useSlotEvent ? &slot->Params.WaitEvent : nullptr};
		nosVulkan->End(cmd, &end);
		Ring->EndPush(slot);
		if (Mode == RingMode::FILL && Ring->IsFull())
		{
			Mode = RingMode::CONSUME;
			ModeCV.notify_all();
		}

		return NOS_RESULT_SUCCESS;
	}

	nosResult CopyFromBegin(nosCopyInfo* cpy, TRing::Resource** foundSlot, nosResourceShareInfo* outputResource) {
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
			if (!ModeCV.wait_for(lock, std::chrono::milliseconds(100), [this] { return Mode != RingMode::FILL; }))
				return NOS_RESULT_PENDING;
		}

		auto effectiveSpareCount = SpareCount.load(); // TODO: * (1 + u32(th->Interlaced()));

		TRing::Resource* slot = nullptr;
		{
			ScopedProfilerEvent({ .Name = "Wait For Filled Slot" });
			slot = Ring->BeginPop(100);
		}
		// If timeout or exit
		if (!slot)
			return Ring->Exit ? NOS_RESULT_FAILED : NOS_RESULT_PENDING;

		nosResourceShareInfo output;

		if (type == ResourceType::Buffer)
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
		else if (type == ResourceType::Texture)
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
		*foundSlot = slot;
		*outputResource = output;
		return NOS_RESULT_SUCCESS;
	}

	void OnPathCommand(const nosPathCommand* command) override
	{
		switch (command->Event)
		{
		case NOS_RING_SIZE_CHANGE: {
			if (command->RingSize == 0)
			{
				nosEngine.LogW((GetName() + " size cannot be 0.").c_str());
				return;
			}
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
		if (OnRestart == OnRestartType::WAIT_UNTIL_FULL)
			Mode = RingMode::FILL;
		if (Ring)
		{
			Ring->Stop();
		}
	}

	void OnPathStart() override
	{
		if (!Ring)
			return;
		if (Ring && OnRestart == OnRestartType::RESET)
			Ring->Reset(false);
		if (RequestedRingSize)
		{
			Ring->Resize(*RequestedRingSize);
			RequestedRingSize = std::nullopt;
		}
		if (NeedsRecreation)
		{
			Ring = std::make_unique<TRing>(Ring->Size, type, Ring->Sample);
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

	void SendPathRestart()
	{
		nosEngine.SendPathRestart(PinName2Id[NOS_NAME_STATIC("Input")]);
	}
};

} // namespace nos