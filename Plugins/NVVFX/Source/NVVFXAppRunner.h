#pragma once
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <string>
#include <iostream>

#include "nvVideoEffects.h"
#include <Windows.h>
#include "Nodos/PluginAPI.h"
#include "Nodos/PluginHelpers.hpp"
#include "CUDAResourceManager.h"

#define CHECK_NVCV_ERROR(nvcv_res)	\
	do{							\
		if (nvcv_res != NVCV_SUCCESS) {	\
			nosEngine.LogE("NVVFX failed with error: %s", NvCV_GetErrorStringFromCode(nvcv_res));	\
			return NOS_RESULT_FAILED; \
		}						\
	}while(0)					\

class NVVFXAppRunner {
public:
	NVVFXAppRunner();
	~NVVFXAppRunner();

	nosResult InitTransferBuffers(NvCVImage* source, NvCVImage* destination);
	nosResult CreateArtifactReductionEffect(std::string modelsDir);
	nosResult CreateSuperResolutionEffect(std::string modelsDir);
	nosResult CreateAIGreenScreenEffect(std::string modelsDir);
	nosResult RunArtifactReduction(NvCVImage* input, NvCVImage* output);
	nosResult RunSuperResolution(NvCVImage* input, NvCVImage* output);
	nosResult RunAIGreenScreenEffect(NvCVImage* input, NvCVImage* output);
	
private:
	
	NvVFX_Handle AR_EffectHandle;
	NvVFX_Handle UpScale_EffectHandle;
	NvVFX_Handle SuperRes_EffectHandle;
	NvVFX_Handle AIGreenScreen_EffectHandle; //AIGS
	NvVFX_StateObjectHandle AIGS_StateObjectHandle;

	NvCVImage InputTransferred = {};
	NvCVImage OutputToBeTransferred = {};
	NvCVImage Temp = {};
	int LastWidth = 0, LastHeight = 0;
	NvCVImage_ComponentType LastComponentType = NvCVImage_ComponentType::NVCV_TYPE_UNKNOWN;
	NvCVImage_PixelFormat LastPixelFormat = NvCVImage_PixelFormat::NVCV_FORMAT_UNKNOWN;
	bool NeedToSet = false;
};



/*
Here is a brief documentation:
retrieved from: https://docs.nvidia.com/deeplearning/maxine/pdf/vfx-sdk-programming-guide.pdf at 29 Dec 2023

**Effects**
 
	AI Green Screen:
		The AI green screen filter segments a video or still image into foreground and background
		regions.
		-> Quality mode, which gives the highest quality result.
			-> Images must be at least 288 pixels high and at least 512 pixels wide.
			-> This mode is the default.
		-> Performance mode, which gives the fastest performance.
			-> Some degradation in quality may be observed.
			-> Images must be at least 288 pixels high and at least 512 pixels wide.
		The filter’s input/output is as follows:
			-> The input should be provided in a GPU buffer in BGR interleaved format, where each pixel
				is a 24-bit unsigned char value, and, therefore, 8-bit per pixel component.
			-> The output of the filter is written to an 8-bit (grayscale) GPU buffer. This buffer should be used as a mask.


	Background Blur Filter:
		-> The input must be a 24-bit BGR input image, and therefore, 8-bit per pixel component.
			The data type is UINT8, and the range of values is [0, 255].
		-> The segmentation mask must be an 8-bit segmentation mask.
			The data type is UINT8, and the range of values is [0, 255].
		-> Output is a 24-bit BGR image, and the data type is UINT8.

	Artifact Reduction Filter:
		Here are the two modes of operation:
		-> Mode 0 removes lesser artifacts, preserves low gradient information better, and is suited
			for higher bitrate videos.
		-> Mode 1 is better suited for lower bitrate videos.

		The filter’s input/output is as follows:
		-> The input should be provided in a GPU buffer in BGR planar format, where each pixel
			component is a 32-bit float value.
		-> The output of the filter is a GPU buffer of the same size as the input image, in BGR planar
			format, where each pixel component is a 32-bit float value.

	Super Resolution Filter:
		-> Mode 0 enhances less and removes more encoding artifacts and is suited for lower-quality
			videos.
		-> Mode 1 enhances more and is suited for higher quality lossless videos.
			The filter’s input/output is as follows:
		-> The input should be provided in a GPU buffer in BGR planar format, where each pixel
			component is a 32-bit float value.
		-> The output of the filter is a GPU buffer in BGR planar format, where each pixel component
			is a 32-bit float value.

	Upscale Filter:
		This is a very fast and light-weight method for upscaling an input video.
		Upscale filter provides a floating-point strength value that ranges between 0.0 and 1.0. This
		signifies an enhancement parameter:
			-> Strength 0 implies no enhancement, only upscaling.
			-> Strength 1 implies the maximum enhancement.
			-> The default value is 0.4.
				The filter’s input/output is as follows:
			-> The input should be provided in a GPU buffer in RGBA chunky/interleaved format, where
				each pixel is 32-bit and therefore, 8-bit per pixel component.
			-> The output of the filter is a GPU buffer in RGBA chunky/interleaved format, where each
				pixel is 32-bit, and, therefore, 8-bit per pixel component.
				Here are some general recommendations:
			-> If a video without encoding artifacts needs a fast resolution increase, use Upscale.
				Effects in the Video Effects SDK
				NVIDIA Video Effects SDK PG-10470-001_v0.7.2   |   5
			-> If a video has no encoding artifacts, to increase the resolution, use SuperRes with mode 1
				for greater enhancement.
			-> If a video has fewer encoding artifacts, to remove artifacts, use ArtifactReduction only with
				mode 0.
			-> If a video has more encoding artifacts, to remove artifacts, use ArtifactReduction only with
				mode 1.
			-> To increase the resolution of a video with encoding artifacts:
			-> For light artifacts, use SuperRes with mode 0.
			-> Otherwise, use ArtifactReduction followed by SuperRes with mode 1.


*/