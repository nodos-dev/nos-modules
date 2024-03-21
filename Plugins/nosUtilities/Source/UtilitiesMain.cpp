// Copyright Nodos AS. All Rights Reserved.

// Includes
#include <Nodos/PluginHelpers.hpp>
#include <glm/glm.hpp>
#include <Builtins_generated.h>

#include <nosVulkanSubsystem/nosVulkanSubsystem.h>

NOS_INIT();
NOS_VULKAN_INIT();
NOS_REGISTER_NAME(Input);
NOS_REGISTER_NAME(Output);
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME(Out);
NOS_REGISTER_NAME(Path);
NOS_REGISTER_NAME(sRGB);

namespace nos::utilities
{

enum Utilities : int
{	// CPU nodes
	Resize = 0,
	ChannelViewer,
	Merge,
	Time,
	ReadImage,
	WriteImage,
	Interlace,
	Deinterlace,
	RGB2YCbCr,
	GammaLUT,
	Buffer2Texture,
	ColorSpaceMatrix,
	CPUSleep,
	BufferRing,
	Count
};

nosResult RegisterMerge(nosNodeFunctions*);
nosResult RegisterTime(nosNodeFunctions*);
nosResult RegisterReadImage(nosNodeFunctions*);
nosResult RegisterWriteImage(nosNodeFunctions*);
nosResult RegisterChannelViewer(nosNodeFunctions*);
nosResult RegisterResize(nosNodeFunctions*);
nosResult RegisterInterlace(nosNodeFunctions*);
nosResult RegisterDeinterlace(nosNodeFunctions*);
nosResult RegisterRGB2YCbCr(nosNodeFunctions*);
nosResult RegisterGammaLUT(nosNodeFunctions*);
nosResult RegisterBuffer2Texture(nosNodeFunctions*);
nosResult RegisterColorSpaceMatrix(nosNodeFunctions*);
nosResult RegisterCPUSleep(nosNodeFunctions*);
nosResult RegisterBufferRing(nosNodeFunctions*);

extern "C"
{

NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outSize, nosNodeFunctions** outList)
{
    *outSize = Utilities::Count;
	if (!outList)
		return NOS_RESULT_SUCCESS;

	auto ret = RequestVulkanSubsystem();
	if (ret != NOS_RESULT_SUCCESS)
		return ret;

#define GEN_CASE_NODE(name)				\
	case Utilities::name: {					\
		auto ret = Register##name(node);	\
		if (NOS_RESULT_SUCCESS != ret)		\
			return ret;						\
		break;								\
	}

	for (int i = 0; i < Utilities::Count; ++i)
	{
		auto node = outList[i];
		switch ((Utilities)i) {
			default:
				break;
			GEN_CASE_NODE(Merge)
			GEN_CASE_NODE(Time)
			GEN_CASE_NODE(ReadImage)
			GEN_CASE_NODE(WriteImage)
			GEN_CASE_NODE(ChannelViewer)
			GEN_CASE_NODE(Resize)
			GEN_CASE_NODE(Interlace)
			GEN_CASE_NODE(Deinterlace)
			GEN_CASE_NODE(RGB2YCbCr)
			GEN_CASE_NODE(GammaLUT)
			GEN_CASE_NODE(Buffer2Texture)
			GEN_CASE_NODE(ColorSpaceMatrix)
			GEN_CASE_NODE(CPUSleep)
			GEN_CASE_NODE(BufferRing)
		}
	}
	return NOS_RESULT_SUCCESS;
}
}
}
