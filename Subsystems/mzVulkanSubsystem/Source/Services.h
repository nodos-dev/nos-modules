#pragma once

#include "mzVulkanSubsystem/mzVulkanSubsystem.h"

namespace mz::vkss
{
mzResult Initialize();
mzResult Deinitialize();


mzResult Begin (mzCmd* outCmd);
mzResult End (mzCmd cmd);
mzResult WaitEvent (uint64_t eventHandle);
mzResult Copy (mzCmd, const mzResourceShareInfo* src, const mzResourceShareInfo* dst, const char* benchmark); // benchmark as string?
mzResult RunPass (mzCmd, const mzRunPassParams* params);
mzResult RunPass2 (mzCmd, const mzRunPass2Params* params);
mzResult RunComputePass (mzCmd, const mzRunComputePassParams* params);
mzResult Clear (mzCmd, const mzResourceShareInfo* texture, mzVec4 color);
mzResult Download (mzCmd, const mzResourceShareInfo* texture, mzResourceShareInfo* outBuffer);
mzResult Create (mzResourceShareInfo* inout);
mzResult Destroy (const mzResourceShareInfo* resource);
mzResult ReloadShaders (mzName nodeName);
uint8_t* Map (const mzResourceShareInfo* buffer);
mzResult GetColorTexture (mzVec4 color, mzResourceShareInfo* out);
mzResult ImageLoad (void* buf, mzVec2u extent, mzFormat format, mzResourceShareInfo* out);
}
