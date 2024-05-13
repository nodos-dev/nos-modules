#include "RingNodeBase.hpp"

namespace nos::MediaIO
{
struct BufferRingNodeContext : RingNodeBase<nosBufferInfo, true>
{
	static constexpr nosBufferInfo SampleBuffer =
		nosBufferInfo{ .Size = 1,
					  .Alignment = 0,
					  .Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_TRANSFER_DST),
					  .MemoryFlags = nosMemoryFlags(NOS_MEMORY_FLAGS_DOWNLOAD | NOS_MEMORY_FLAGS_HOST_VISIBLE) };
	
	BufferRingNodeContext(nosFbNode const* node) : RingNodeBase(node, SampleBuffer, RingNodeBase::RingType::RING)
	{
	}
	std::string GetName() const override
	{
		return "BoundedTextureQueue";
	}
};

nosResult RegisterBufferRing(nosNodeFunctions* functions)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("BufferRing"), BufferRingNodeContext, functions)
	return NOS_RESULT_SUCCESS;
}

}