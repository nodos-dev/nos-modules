// Copyright Nodos AS. All Rights Reserved.

// Includes
#include <Nodos/Helpers.hpp>
#include <glm/glm.hpp>
#include <Builtins_generated.h>

// Shaders
#include "../Shaders/Checkerboard.frag.spv.dat"
#include "../Shaders/Color.frag.spv.dat"
#include "../Shaders/Gradient.frag.spv.dat"
#include "../Shaders/Merge.frag.spv.dat"
#include "../Shaders/Offset.frag.spv.dat"
#include "../Shaders/QuadMerge.frag.spv.dat"
#include "../Shaders/Resize.frag.spv.dat"
#include "../Shaders/SevenSegment.frag.spv.dat"
#include "../Shaders/Swizzle.frag.spv.dat"
#include "../Shaders/TextureSwitcher.frag.spv.dat"


NOS_INIT();

namespace nos::utilities
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

void RegisterMerge(nosNodeFunctions*);
void RegisterTime(nosNodeFunctions*);
void RegisterReadImage(nosNodeFunctions*);
void RegisterWriteImage(nosNodeFunctions*);
void RegisterChannelViewer(nosNodeFunctions*);
void RegisterResize(nosNodeFunctions*);
void RegisterInterlace(nosNodeFunctions*);
void RegisterDeinterlace(nosNodeFunctions*);

extern "C"
{

NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outSize, nosNodeFunctions** outList)
{
    *outSize = Utilities::Count;
	if (!outList)
	{
		return NOS_RESULT_SUCCESS;
	}

#define GEN_CASE_GPU_NODE(name)                                     \
	case Utilities::name: {                                         \
			node->TypeName = NOS_NAME_STATIC("nos.utilities." #name); \
			node->GetShaderSource = [](nosShaderSource* spirv) {     \
				spirv->SpirvBlob = {(void*)(name##_frag_spv),       \
						sizeof(name##_frag_spv)};                   \
				return NOS_RESULT_SUCCESS;                           \
			};                                                      \
			break;                                                  \
	}
#define GEN_CASE_GPU_NODE_LICENSED(name, featureName, featureMessage)							\
	case Utilities::name: {																		\
			node->TypeName = NOS_NAME_STATIC("nos.utilities." #name);								\
			node->GetShaderSource = [](nosShaderSource* spirv) {									\
				spirv->SpirvBlob = {(void*)(name##_frag_spv),									\
						sizeof(name##_frag_spv)};												\
				return NOS_RESULT_SUCCESS;														\
			};																					\
			node->OnNodeCreated = [](const nosFbNode* node, void** outCtxPtr) {					\
					nosEngine.RegisterFeature(*node->id(), featureName, 1, featureMessage);		\
				};																				\
			node->OnNodeDeleted = [](void* ctx, nosUUID nodeId) {								\
					nosEngine.UnregisterFeature(nodeId, featureName);							\
				};																				\
			break;																				\
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
			default:
			{
				break;
			}
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
	return NOS_RESULT_SUCCESS;
}
}
}
