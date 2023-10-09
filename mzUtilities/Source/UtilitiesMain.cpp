// Copyright MediaZ AS. All Rights Reserved.

// Includes
#include <MediaZ/Helpers.hpp>
#include <glm/glm.hpp>
#include <Builtins_generated.h>

// Shaders
#include "Checkerboard.frag.spv.dat"
#include "Color.frag.spv.dat"
#include "Gradient.frag.spv.dat"
#include "Merge.frag.spv.dat"
#include "Offset.frag.spv.dat"
#include "QuadMerge.frag.spv.dat"
#include "Resize.frag.spv.dat"
#include "SevenSegment.frag.spv.dat"
#include "Swizzle.frag.spv.dat"
#include "TextureSwitcher.frag.spv.dat"


MZ_INIT();

namespace mz::utilities
{

enum Utilities : int
{
	Checkerboard = 0,
	Color,
	Gradient,
	Offset,
	QuadMerge,
	Resize,
	SevenSegment,
	Swizzle,
	TextureSwitcher,
	ChannelViewer,
	Merge,
	Time,
	ReadImage,
	WriteImage,
	Interlace,
	Deinterlace,
	Count
};

void RegisterMerge(mzNodeFunctions*);
void RegisterTime(mzNodeFunctions*);
void RegisterReadImage(mzNodeFunctions*);
void RegisterWriteImage(mzNodeFunctions*);
void RegisterChannelViewer(mzNodeFunctions*);
void RegisterResize(mzNodeFunctions*);
void RegisterInterlace(mzNodeFunctions*);
void RegisterDeinterlace(mzNodeFunctions*);

extern "C"
{

MZAPI_ATTR mzResult MZAPI_CALL mzExportNodeFunctions(size_t* outSize, mzNodeFunctions** outList)
{
    *outSize = Utilities::Count;
	if (!outList)
	{
		return MZ_RESULT_SUCCESS;
	}

#define GEN_CASE_GPU_NODE(name)                                     \
	case Utilities::name: {                                         \
			node->TypeName = MZ_NAME_STATIC("mz.utilities." #name); \
			node->GetShaderSource = [](mzShaderSource* spirv) {     \
				spirv->SpirvBlob = {(void*)(name##_frag_spv),       \
						sizeof(name##_frag_spv)};                   \
				return MZ_RESULT_SUCCESS;                           \
			};                                                      \
			break;                                                  \
	}
#define GEN_CASE_CPU_NODE(name) \
	case Utilities::name: {     \
            Register##name(node);\
			break;              \
	}

	for (int i = 0; i < Utilities::Count; ++i)
	{
		auto node = outList[i];
		switch ((Utilities)i) {
			GEN_CASE_GPU_NODE(Checkerboard)
			GEN_CASE_GPU_NODE(Color)
			GEN_CASE_GPU_NODE(Gradient)
			GEN_CASE_GPU_NODE(Offset)
			GEN_CASE_GPU_NODE(QuadMerge)
			GEN_CASE_GPU_NODE(SevenSegment)
			GEN_CASE_GPU_NODE(Swizzle)
			GEN_CASE_GPU_NODE(TextureSwitcher)
			GEN_CASE_CPU_NODE(Merge)
			GEN_CASE_CPU_NODE(Time)
			GEN_CASE_CPU_NODE(ReadImage)
			GEN_CASE_CPU_NODE(WriteImage)
			GEN_CASE_CPU_NODE(ChannelViewer)
			GEN_CASE_CPU_NODE(Resize)
			GEN_CASE_CPU_NODE(Interlace)
			GEN_CASE_CPU_NODE(Deinterlace)
		};
	}
	return MZ_RESULT_SUCCESS;
}
}
}
