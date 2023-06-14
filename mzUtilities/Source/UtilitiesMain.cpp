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
#include "Distort.frag.spv.dat"
#include "Undistort.frag.spv.dat"
#include "Swizzle.frag.spv.dat"
#include "TextureSwitcher.frag.spv.dat"


MZ_INIT();

namespace mz::utilities
{

enum Utilities
{
	Checkerboard,
	Color,
	Gradient,
	Offset,
	QuadMerge,
	Resize,
	SevenSegment,
	Distort,
	Undistort,
	Swizzle,
	TextureSwitcher,
	ChannelViewer,
	Merge,
	Time,
	ReadImage,
	WriteImage,
	Array,
	Disarray,
	Count
};

void RegisterMerge(mzNodeFunctions*);
void RegisterTime(mzNodeFunctions*);
void RegisterReadImage(mzNodeFunctions*);
void RegisterWriteImage(mzNodeFunctions*);
void RegisterChannelViewer(mzNodeFunctions*);
void RegisterResize(mzNodeFunctions*);
void RegisterArray(mzNodeFunctions* fn);
void RegisterDisarray(mzNodeFunctions* fn);

extern "C"
{

MZAPI_ATTR mzResult MZAPI_CALL mzExportNodeFunctions(size_t* outSize, mzNodeFunctions* funcs)
{
    *outSize = Utilities::Count;
	if (!funcs)
	{
		return MZ_RESULT_SUCCESS;
	}

#define REGISTER_FILTER(name) \
    funcs[Utilities::##name] = mzNodeFunctions{ \
        .TypeName = MZ_NAME_STATIC("mz.utilities." #name), \
        .GetShaderSource = [](mzBuffer* spirv) { \
				*spirv = { (void*)(name##_frag_spv), sizeof (name##_frag_spv) }; \
				return MZ_RESULT_SUCCESS; \
			},\
    };

    REGISTER_FILTER(Checkerboard);
    REGISTER_FILTER(Color);
    REGISTER_FILTER(Gradient);
    REGISTER_FILTER(Offset);
    REGISTER_FILTER(QuadMerge);
    REGISTER_FILTER(SevenSegment);
    REGISTER_FILTER(Distort);
    REGISTER_FILTER(Undistort);
    REGISTER_FILTER(Swizzle);
    REGISTER_FILTER(TextureSwitcher);

    RegisterMerge(&funcs[Utilities::Merge]);
    RegisterTime(&funcs[Utilities::Time]);
    RegisterReadImage(&funcs[Utilities::ReadImage]);
    RegisterWriteImage(&funcs[Utilities::WriteImage]);
    RegisterChannelViewer(&funcs[Utilities::ChannelViewer]);
	RegisterResize(&funcs[Utilities::Resize]);
	RegisterArray(&funcs[Utilities::Array]);
	RegisterDisarray(&funcs[Utilities::Disarray]);
	return MZ_RESULT_SUCCESS;
}
}
}
