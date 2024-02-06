// Copyright MediaZ AS. All Rights Reserved.

#pragma once
#include "nosAI/nosAI.h"

#define CHECK_RESULT(nosRes) \
	do { \
		nosResult result = nosRes; \
		if (result != NOS_RESULT_SUCCESS) { \
			nosEngine.LogE("Failed from %s %d with error %d.",__FILE__, __LINE__, result); \
			return NOS_RESULT_FAILED; \
		} \
	} while (0); \

namespace nos::ai
{
	nosResult Bind(nosAISubsystem* subsys);
}
