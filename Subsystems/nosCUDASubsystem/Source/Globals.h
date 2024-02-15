#pragma once
#include "nosCUDASubsystem/nosCUDASubsystem.h"
#include "CUDASubsysCommon.h"
namespace nos::cudass 
{
	extern UtilsProxy::ResourceManagerProxy<nosCUDABufferInfo> ResManager;
	extern uint32_t CurrentDevice;
	extern void* PrimaryContext;
	extern void* ActiveContext;
	extern std::unordered_map<uint64_t, void*> IDContextMap;
}