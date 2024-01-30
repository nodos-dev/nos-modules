#pragma once
#include "nosCUDASubsystem/nosCUDASubsystem.h"
#include "CUDASubsysCommon.h"
namespace nos::cudass 
{
	extern UtilsProxy::ResourceManagerProxy<nosCUDABufferInfo> ResManager;
	extern void* PrimaryContext;
	extern std::unordered_map<int32_t, void*> IDContextMap;
}