// Copyright MediaZ AS. All Rights Reserved.
#include <Nodos/SubsystemAPI.h>
#include <cstring>
#include "nosCUDASubsystem/nosCUDASubsystem.h"
#include "nosVulkanSubsystem/nosVulkanSubsystem.h"
#include "nosAI/nosAI.h"

extern nosVulkanSubsystem* nosVulkan = nullptr;
extern nosCUDASubsystem* nosCUDA = nullptr;

namespace nos::ai
{
	nosResult Bind(nosAISubsystem* subsys) {

	}
}
