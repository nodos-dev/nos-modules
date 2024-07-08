// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

// Includes
#include <Nodos/PluginHelpers.hpp>
#include <glm/glm.hpp>
#include <Builtins_generated.h>

#include <nosVulkanSubsystem/nosVulkanSubsystem.h>

NOS_INIT_WITH_MIN_REQUIRED_MINOR(4);
NOS_VULKAN_INIT();
NOS_REGISTER_NAME(Input);
NOS_REGISTER_NAME(Output);
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME(Out);
NOS_REGISTER_NAME(Path);
NOS_REGISTER_NAME(sRGB);

namespace nos::mediaio
{

enum Nodes : int
{	// CPU nodes
	Interlace,
	Deinterlace,
	RGB2YCbCr,
	YCbCr2RGB,
	YUVBufferSizeCalculator,
	GammaLUT,
	ColorSpaceMatrix,
	BufferRing,
	BoundedTextureQueue,
	UploadBufferProvider,
	YUY2ToRGBA,
	TextureFormatConverter,
	Count
};

nosResult RegisterInterlace(nosNodeFunctions*);
nosResult RegisterDeinterlace(nosNodeFunctions*);
nosResult RegisterRGB2YCbCr(nosNodeFunctions*);
nosResult RegisterYCbCr2RGB(nosNodeFunctions*);
nosResult RegisterYUVBufferSizeCalculator(nosNodeFunctions*);
nosResult RegisterGammaLUT(nosNodeFunctions*);
nosResult RegisterColorSpaceMatrix(nosNodeFunctions*);
nosResult RegisterBufferRing(nosNodeFunctions*);
nosResult RegisterBoundedTextureQueue(nosNodeFunctions*);
nosResult RegisterUploadBufferProvider(nosNodeFunctions*);
nosResult RegisterYUY2ToRGBA(nosNodeFunctions*);
nosResult RegisterTextureFormatConverter(nosNodeFunctions* fn);

extern "C"
{

NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outSize, nosNodeFunctions** outList)
{
    *outSize = Nodes::Count;
	if (!outList)
		return NOS_RESULT_SUCCESS;

	auto ret = RequestVulkanSubsystem();
	if (ret != NOS_RESULT_SUCCESS)
		return ret;

#define GEN_CASE_NODE(name)				\
	case Nodes::name: {					\
		auto ret = Register##name(node);	\
		if (NOS_RESULT_SUCCESS != ret)		\
			return ret;						\
		break;								\
	}

	for (int i = 0; i < Nodes::Count; ++i)
	{
		auto node = outList[i];
		switch ((Nodes)i) {
			default:
				break;
			GEN_CASE_NODE(Interlace)
			GEN_CASE_NODE(Deinterlace)
			GEN_CASE_NODE(RGB2YCbCr)
			GEN_CASE_NODE(YCbCr2RGB)
			GEN_CASE_NODE(YUVBufferSizeCalculator)
			GEN_CASE_NODE(GammaLUT)
			GEN_CASE_NODE(ColorSpaceMatrix)
			GEN_CASE_NODE(BufferRing)
			GEN_CASE_NODE(BoundedTextureQueue)
			GEN_CASE_NODE(UploadBufferProvider)
			GEN_CASE_NODE(YUY2ToRGBA)
			GEN_CASE_NODE(TextureFormatConverter)
		}
	}
	return NOS_RESULT_SUCCESS;
}
}
}
