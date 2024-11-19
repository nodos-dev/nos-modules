// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>

// External
#include <nosVulkanSubsystem/Helpers.hpp>
#include <nosUtil/Stopwatch.hpp>

NOS_REGISTER_NAME(Buffer);
NOS_REGISTER_NAME(GPUEventRef);
NOS_REGISTER_NAME(QueueSize);
NOS_REGISTER_NAME(BufferSize);
NOS_REGISTER_NAME(Alignment);
NOS_REGISTER_NAME(ForceHostMemory);


namespace nos::utilities
{
	struct UploadBuffer
	{
		nosResourceShareInfo BufferInfo = {};
		nosGPUEventResource Event = 0;
		UploadBuffer(nosResourceShareInfo sampleBufferInfo) : BufferInfo(sampleBufferInfo)
		{
			nosVulkan->CreateGPUEventResource(&Event);
			nosVulkan->CreateResource(&BufferInfo);
		}
		UploadBuffer(const UploadBuffer& other) = delete;
		UploadBuffer& operator=(const UploadBuffer& other) = delete;
		UploadBuffer(UploadBuffer&& other) 
		{
			BufferInfo = other.BufferInfo;
			Event = other.Event;
			other.Event = 0;
			other.BufferInfo = {};
		}

		~UploadBuffer()
		{
			if (Event)
			{
				nosGPUEvent* event = nullptr;
				auto res = nosVulkan->GetGPUEvent(Event, &event);

				if (res == NOS_RESULT_SUCCESS && *event)
					nosVulkan->WaitGpuEvent(event, UINT64_MAX);
			}
			nosVulkan->DestroyResource(&BufferInfo);
			nosVulkan->DestroyGPUEventResource(&Event);
		}
	};

	struct UploadBufferProviderNodeContext : NodeContext
	{
		nosResourceShareInfo SampleBuffer = {
			.Info = {
				.Type = NOS_RESOURCE_TYPE_BUFFER,
				.Buffer = nosBufferInfo{
					.Size = 0,
					.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC),
					.MemoryFlags = nosMemoryFlags(NOS_MEMORY_FLAGS_HOST_VISIBLE | NOS_MEMORY_FLAGS_FORCE_HOST_MEMORY)
				}
			}
		};
		std::vector<UploadBuffer> Buffers;
		uint64_t QueueSize = 2;

		size_t CurrentIndex = 0;

		UploadBufferProviderNodeContext(const nosFbNode* node) : NodeContext(node)
		{
			AddPinValueWatcher(NSN_QueueSize, [this](nos::Buffer const& newVal, std::optional<nos::Buffer> oldVal) 
				{
					QueueSize = *InterpretPinValue<uint32_t>(newVal);
					if (QueueSize == 0)
						return;
					if(Buffers.size() == QueueSize)
						return;
					if (SampleBuffer.Info.Buffer.Size == 0)
						return;
					Buffers.clear();
					for (size_t i = 0; i < QueueSize; i++)
						Buffers.emplace_back(SampleBuffer);
					CurrentIndex = 0;
				});
			AddPinValueWatcher(NSN_BufferSize, [this](nos::Buffer const& newVal, std::optional<nos::Buffer> oldVal)
				{
					uint64_t newSize = *InterpretPinValue<uint64_t>(newVal);
					if (newSize == 0)
						return;
					if (SampleBuffer.Info.Buffer.Size == newSize)
						return;
					SampleBuffer.Info.Buffer.Size = newSize;
					Buffers.clear();
					for (size_t i = 0; i < QueueSize; i++)
						Buffers.emplace_back(SampleBuffer);
				});
			AddPinValueWatcher(NSN_Alignment, [this](nos::Buffer const& newVal, std::optional<nos::Buffer> oldVal)
				{
					uint64_t newAlignment = *InterpretPinValue<uint64_t>(newVal);
					if (SampleBuffer.Info.Buffer.Alignment == newAlignment)
						return;
					SampleBuffer.Info.Buffer.Alignment = newAlignment;
					Buffers.clear();
					for (size_t i = 0; i < QueueSize; i++)
						Buffers.emplace_back(SampleBuffer);
				});
			AddPinValueWatcher(NSN_ForceHostMemory, [this](nos::Buffer const& newVal, std::optional<nos::Buffer> oldVal)
				{
					bool newForceHostMemory = *InterpretPinValue<bool>(newVal);
					auto& memFlags = SampleBuffer.Info.Buffer.MemoryFlags;
					if (!!(memFlags & NOS_MEMORY_FLAGS_FORCE_HOST_MEMORY) == newForceHostMemory)
						return;
					if (newForceHostMemory)
						memFlags = nosMemoryFlags(memFlags | NOS_MEMORY_FLAGS_FORCE_HOST_MEMORY);
					else
						memFlags = nosMemoryFlags(memFlags & ~NOS_MEMORY_FLAGS_FORCE_HOST_MEMORY);
					Buffers.clear();
					for (size_t i = 0; i < QueueSize; i++)
						Buffers.emplace_back(SampleBuffer);
				});
		}

		nosResult ExecuteNode(nosNodeExecuteParams* params) override
		{
			if (Buffers.size() == 0)
				return NOS_RESULT_FAILED;
			auto execParams = nos::NodeExecuteParams(params);
			auto& nextBuf = Buffers[CurrentIndex];
			CurrentIndex = (CurrentIndex + 1) % QueueSize;
			if (nextBuf.Event)
			{
				util::Stopwatch sw;
				nosGPUEvent* event = nullptr;
				auto res = nosVulkan->GetGPUEvent(nextBuf.Event, &event);
				if (*event)
					nosVulkan->WaitGpuEvent(event, UINT64_MAX);
				nosEngine.WatchLog("UploadBufferProvider Wait", sw.ElapsedString().c_str());
			}

			nosEngine.SetPinValue(execParams[NSN_Buffer].Id, Buffer::From(vkss::ConvertBufferInfo(nextBuf.BufferInfo)));
			nosEngine.SetPinValue(execParams[NSN_GPUEventRef].Id, Buffer::From(nos::sys::vulkan::GPUEventResource(nextBuf.Event)));

			return NOS_RESULT_SUCCESS;
		}
		
		void OnPathStop() override
		{
			for (auto& buf : Buffers)
			{
				if (buf.Event)
				{
					nosGPUEvent* event = nullptr;
					auto res = nosVulkan->GetGPUEvent(buf.Event, &event);
					if (*event)
						nosVulkan->WaitGpuEvent(event, UINT64_MAX);
				}
			}
			CurrentIndex = 0;
		}

		void OnPathCommand(const nosPathCommand* command) override
		{
			switch (command->Event)
			{
			case NOS_RING_SIZE_CHANGE: {
				if (command->RingSize == 0)
				{
					nosEngine.LogW("Buffer provider size cannot be 0");
					return;
				}
				nosEngine.SetPinValue(PinName2Id[NSN_QueueSize], Buffer::From(uint32_t(command->RingSize)));
				break;
			}
			default: return;
			}
		}

	};

	nosResult RegisterUploadBufferProvider(nosNodeFunctions* functions)
	{
		NOS_BIND_NODE_CLASS(NOS_NAME("UploadBufferProvider"), UploadBufferProviderNodeContext, functions)
			return NOS_RESULT_SUCCESS;
	}
}