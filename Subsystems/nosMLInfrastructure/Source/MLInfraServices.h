// Copyright MediaZ AS. All Rights Reserved.

#pragma once
#include "nosMLInfrastructure/nosMLInfrastructure.h"

namespace nos::ml::infra
{
	void Bind(nosMLInfrastructure* subsys);
	nosResult CreateEmptyTensor(nosTensor* tensorOut, nosTensorCreateInfo createInfo);
	nosResult InitTensor(nosTensor* tensorOut, void* MemoryAddress, nosTensorCreateInfo createInfo);
	nosResult CopyDataToTensor(nosTensor* tensorOut, void* MemoryAddress, uint64_t Size);
	nosResult SliceTensor(nosTensor* tensorIn, uint64_t* outCount, nosTensor* outTensors);

	uint64_t GetRequiredSizeFromCreateInfo(nosTensorCreateInfo createInfo);
	short GetSizeOfElementType(TensorElementType type);
}
