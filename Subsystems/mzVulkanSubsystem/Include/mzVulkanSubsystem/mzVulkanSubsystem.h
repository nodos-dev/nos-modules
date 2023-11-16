#ifndef MZ_VULKAN_SUBSYSTEM_H_INCLUDED
#define MZ_VULKAN_SUBSYSTEM_H_INCLUDED

#include "MediaZ/Types.h"

typedef enum mzFormat
{
	MZ_FORMAT_NONE = 0,
	MZ_FORMAT_R8_UNORM = 9,
	MZ_FORMAT_R8_UINT = 13,
	MZ_FORMAT_R8_SRGB = 15,
	MZ_FORMAT_R8G8_UNORM = 16,
	MZ_FORMAT_R8G8_UINT = 20,
	MZ_FORMAT_R8G8_SRGB = 22,
	MZ_FORMAT_R8G8B8_UNORM = 23,
	MZ_FORMAT_R8G8B8_SRGB = 29,
	MZ_FORMAT_B8G8R8_UNORM = 30,
	MZ_FORMAT_B8G8R8_UINT = 34,
	MZ_FORMAT_B8G8R8_SRGB = 36,
	MZ_FORMAT_R8G8B8A8_UNORM = 37,
	MZ_FORMAT_R8G8B8A8_UINT = 41,
	MZ_FORMAT_R8G8B8A8_SRGB = 43,
	MZ_FORMAT_B8G8R8A8_UNORM = 44,
	MZ_FORMAT_B8G8R8A8_SRGB = 50,
	MZ_FORMAT_A2R10G10B10_UNORM_PACK32 = 58,
	MZ_FORMAT_A2R10G10B10_SNORM_PACK32 = 59,
	MZ_FORMAT_A2R10G10B10_USCALED_PACK32 = 60,
	MZ_FORMAT_A2R10G10B10_SSCALED_PACK32 = 61,
	MZ_FORMAT_A2R10G10B10_UINT_PACK32 = 62,
	MZ_FORMAT_A2R10G10B10_SINT_PACK32 = 63,
	MZ_FORMAT_R16_UNORM = 70,
	MZ_FORMAT_R16_SNORM = 71,
	MZ_FORMAT_R16_USCALED = 72,
	MZ_FORMAT_R16_SSCALED = 73,
	MZ_FORMAT_R16_UINT = 74,
	MZ_FORMAT_R16_SINT = 75,
	MZ_FORMAT_R16_SFLOAT = 76,
	MZ_FORMAT_R16G16_UNORM = 77,
	MZ_FORMAT_R16G16_SNORM = 78,
	MZ_FORMAT_R16G16_USCALED = 79,
	MZ_FORMAT_R16G16_SSCALED = 80,
	MZ_FORMAT_R16G16_UINT = 81,
	MZ_FORMAT_R16G16_SINT = 82,
	MZ_FORMAT_R16G16_SFLOAT = 83,
	MZ_FORMAT_R16G16B16_UNORM = 84,
	MZ_FORMAT_R16G16B16_SNORM = 85,
	MZ_FORMAT_R16G16B16_USCALED = 86,
	MZ_FORMAT_R16G16B16_SSCALED = 87,
	MZ_FORMAT_R16G16B16_UINT = 88,
	MZ_FORMAT_R16G16B16_SINT = 89,
	MZ_FORMAT_R16G16B16_SFLOAT = 90,
	MZ_FORMAT_R16G16B16A16_UNORM = 91,
	MZ_FORMAT_R16G16B16A16_SNORM = 92,
	MZ_FORMAT_R16G16B16A16_USCALED = 93,
	MZ_FORMAT_R16G16B16A16_SSCALED = 94,
	MZ_FORMAT_R16G16B16A16_UINT = 95,
	MZ_FORMAT_R16G16B16A16_SINT = 96,
	MZ_FORMAT_R16G16B16A16_SFLOAT = 97,
	MZ_FORMAT_R32_UINT = 98,
	MZ_FORMAT_R32_SINT = 99,
	MZ_FORMAT_R32_SFLOAT = 100,
	MZ_FORMAT_R32G32_UINT = 101,
	MZ_FORMAT_R32G32_SINT = 102,
	MZ_FORMAT_R32G32_SFLOAT = 103,
	MZ_FORMAT_R32G32B32_UINT = 104,
	MZ_FORMAT_R32G32B32_SINT = 105,
	MZ_FORMAT_R32G32B32_SFLOAT = 106,
	MZ_FORMAT_R32G32B32A32_UINT = 107,
	MZ_FORMAT_R32G32B32A32_SINT = 108,
	MZ_FORMAT_R32G32B32A32_SFLOAT = 109,
	MZ_FORMAT_B10G11R11_UFLOAT_PACK32 = 122,
	MZ_FORMAT_D16_UNORM = 124,
	MZ_FORMAT_X8_D24_UNORM_PACK32 = 125,
	MZ_FORMAT_D32_SFLOAT = 126,
	MZ_FORMAT_G8B8G8R8_422_UNORM = 1000156000,
	MZ_FORMAT_B8G8R8G8_422_UNORM = 1000156001,
} mzFormat;

typedef enum mzTextureFilter
{
	MZ_TEXTURE_FILTER_NEAREST = 0,
	MZ_TEXTURE_FILTER_LINEAR = 1,
	MZ_TEXTURE_FILTER_CUBIC = 1000015000,
	MZ_TEXTURE_FILTER_MIN = MZ_TEXTURE_FILTER_NEAREST,
	MZ_TEXTURE_FILTER_MAX = MZ_TEXTURE_FILTER_CUBIC
} mzTextureFilter;

typedef enum mzBufferUsage
{
	MZ_BUFFER_USAGE_TRANSFER_SRC = 0x00000001,
	MZ_BUFFER_USAGE_TRANSFER_DST = 0x00000002,
	MZ_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER = 0x00000004,
	MZ_BUFFER_USAGE_STORAGE_TEXEL_BUFFER = 0x00000008,
	MZ_BUFFER_USAGE_UNIFORM_BUFFER = 0x00000010,
	MZ_BUFFER_USAGE_STORAGE_BUFFER = 0x00000020,
	MZ_BUFFER_USAGE_INDEX_BUFFER = 0x00000040,
	MZ_BUFFER_USAGE_VERTEX_BUFFER = 0x00000080,
	MZ_BUFFER_USAGE_DEVICE_MEMORY = 0x00000100,
	MZ_BUFFER_USAGE_NOT_HOST_VISIBLE = 0x00000200,
} mzBufferUsage;

typedef enum mzImageUsage
{
	MZ_IMAGE_USAGE_NONE = 0x00000000,
	MZ_IMAGE_USAGE_TRANSFER_SRC = 0x00000001,
	MZ_IMAGE_USAGE_TRANSFER_DST = 0x00000002,
	MZ_IMAGE_USAGE_SAMPLED = 0x00000004,
	MZ_IMAGE_USAGE_STORAGE = 0x00000008,
	MZ_IMAGE_USAGE_RENDER_TARGET = 0x00000010,
	MZ_IMAGE_USAGE_DEPTH_STENCIL = 0x00000020,
	MZ_IMAGE_USAGE_TRANSIENT = 0x00000040,
	MZ_IMAGE_USAGE_INPUT = 0x00000080,
} mzImageUsage;

typedef enum mzResourceType
{
	MZ_RESOURCE_TYPE_BUFFER = 1,
	MZ_RESOURCE_TYPE_TEXTURE = 2,
} mzResourceType;

typedef enum mzTextureFieldType
{
	MZ_TEXTURE_FIELD_TYPE_UNKNOWN = 0,
	MZ_TEXTURE_FIELD_TYPE_EVEN = 1,
	MZ_TEXTURE_FIELD_TYPE_ODD = 2,
	MZ_TEXTURE_FIELD_TYPE_PROGRESSIVE = 3,
} mzTextureFieldType;

typedef struct mzMemoryInfo
{
	uint32_t Type;
	uint64_t Handle;
	uint64_t PID;
	uint64_t Memory;
	uint64_t Offset;
} mzMemoryInfo;

typedef struct mzTextureInfo
{
	uint32_t Width;
	uint32_t Height;
	mzFormat Format;
	mzTextureFilter Filter;
	mzImageUsage Usage;
	uint64_t Semaphore;
	mzTextureFieldType FieldType;
} mzTextureInfo;

typedef struct mzBufferInfo
{
	uint32_t Size;
	mzBufferUsage Usage;
} mzBufferInfo;

typedef struct mzTextureShareInfo
{
	mzMemoryInfo Memory;
	mzTextureInfo Info;
} mzTextureShareInfo;

typedef struct mzBufferShareInfo
{
	mzMemoryInfo Memory;
	mzBufferInfo Info;
} mzBufferShareInfo;

typedef struct mzResourceInfo
{
	mzResourceType Type;
	union {
		mzTextureInfo Texture;
		mzBufferInfo Buffer;
	};
} mzResourceInfo;

typedef struct mzResourceShareInfo
{
	mzMemoryInfo Memory;
	mzResourceInfo Info;
} mzResourceShareInfo;

typedef void* mzCmd;

typedef struct mzShaderBinding
{
	mzName Name;
	union 
    {
		const struct mzResourceShareInfo* Resource;
		struct
		{
			const void* Data;
			size_t Size;
		};
	};
} mzShaderBinding;

typedef enum mzDepthFunction {
	MZ_DEPTH_FUNCTION_NEVER,
	MZ_DEPTH_FUNCTION_LESS,
	MZ_DEPTH_FUNCTION_EQUAL,
	MZ_DEPTH_FUNCTION_LESS_OR_EQUAL,
	MZ_DEPTH_FUNCTION_GREATER,
	MZ_DEPTH_FUNCTION_NOT_EQUAL,
	MZ_DEPTH_FUNCTION_GREATER_OR_EQUAL,
	MZ_DEPTH_FUNCTION_ALWAYS
} mzDepthFunction;

typedef struct mzVertexData {
	mzResourceShareInfo Buffer;
	uint32_t VertexOffset;
	uint32_t IndexOffset;
	uint32_t IndexCount;
	mzDepthFunction DepthFunc;
	mzBool DepthWrite;
	mzBool DepthTest;
} mzVertexData;

typedef struct mzRunPassParams {
	mzName Key;
	const mzShaderBinding* Bindings;
	uint32_t BindingCount;
	mzResourceShareInfo Output;
	mzVertexData Vertices;
	mzBool Wireframe;
	mzBool Benchmark;
	mzBool DoNotClear;
	mzVec4 ClearCol;
} mzRunPassParams;

typedef struct mzDrawCall {
    const mzShaderBinding* Bindings;
    uint32_t BindingCount;
    mzVertexData Vertices;
} mzDrawCall;

typedef struct mzRunPass2Params {
	mzName Key;
	mzResourceShareInfo Output;
    const mzDrawCall* DrawCalls;
    uint32_t DrawCallCount;
	mzBool Wireframe;
	uint32_t Benchmark;
	mzBool DoNotClear;
	mzVec4 ClearCol;
} mzRunPass2Params;

typedef struct mzRunComputePassParams {
	mzName Key;
	const mzShaderBinding* Bindings;
	uint32_t BindingCount;
	mzResourceShareInfo Output;
	mzVec2u DispatchSize;
	uint32_t Benchmark;
} mzRunComputePassParams;

typedef struct mzVulkanSubsystem
{
	mzResult (MZAPI_CALL *Begin)(mzCmd* outCmd);
	mzResult (MZAPI_CALL *End)(mzCmd cmd);
	mzResult (MZAPI_CALL *WaitEvent)(uint64_t eventHandle);
	mzResult (MZAPI_CALL *Copy)(mzCmd, const mzResourceShareInfo* src, const mzResourceShareInfo* dst, const char* benchmark); // benchmark as string?
	mzResult (MZAPI_CALL *RunPass)(mzCmd, const mzRunPassParams* params);
	mzResult (MZAPI_CALL *RunPass2)(mzCmd, const mzRunPass2Params* params);
	mzResult (MZAPI_CALL *RunComputePass)(mzCmd, const mzRunComputePassParams* params);
	mzResult (MZAPI_CALL *Clear)(mzCmd, const mzResourceShareInfo* texture, mzVec4 color);
	mzResult (MZAPI_CALL *Download)(mzCmd, const mzResourceShareInfo* texture, mzResourceShareInfo* outBuffer);
	mzResult (MZAPI_CALL *Create)(mzResourceShareInfo* inout);
	mzResult (MZAPI_CALL *Destroy)(const mzResourceShareInfo* resource);
	mzResult (MZAPI_CALL *ReloadShaders)(mzName nodeName);
	uint8_t* (MZAPI_CALL *Map)(const mzResourceShareInfo* buffer);
	mzResult (MZAPI_CALL *GetColorTexture)(mzVec4 color, mzResourceShareInfo* out);
	mzResult (MZAPI_CALL *ImageLoad)(void* buf, mzVec2u extent, mzFormat format, mzResourceShareInfo* out);
} mzVulkanSubsystem;

#endif // MZ_VULKAN_SUBSYSTEM_H_INCLUDED