#include "RingNodeBase.hpp"

namespace nos::MediaIO
{
	struct BoundedTextureQueueNodeContext : RingNodeBase<nosTextureInfo, false>
	{
		static constexpr nosTextureInfo SampleTexture = nosTextureInfo{ 
			.Width = 1920, 
			.Height = 1080,
			.Format = NOS_FORMAT_R16G16B16A16_SFLOAT,
			.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_SRC | NOS_IMAGE_USAGE_TRANSFER_DST),
		};
		BoundedTextureQueueNodeContext(nosFbNode const* node) : RingNodeBase(node, SampleTexture, RingNodeBase::RingType::COPY_RING, RingNodeBase::OnRestartType::RESET)
		{
		}
		std::string GetName() const override
		{
			return "BoundedTextureQueue";
		}
	};

	nosResult RegisterBoundedTextureQueue(nosNodeFunctions* functions)
	{
		NOS_BIND_NODE_CLASS(NOS_NAME("BoundedTextureQueue"), BoundedTextureQueueNodeContext, functions)
			return NOS_RESULT_SUCCESS;
	}

}