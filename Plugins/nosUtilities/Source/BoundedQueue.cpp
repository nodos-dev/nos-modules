// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#pragma once

#include <Nodos/PluginHelpers.hpp>

// External
#include <glm/glm.hpp> // TODO: Ring no longer needs glm::mat4 colormatrix. Remove this
#include <nosVulkanSubsystem/Helpers.hpp>

#include "Ring.h"
#include "nosUtil/Stopwatch.hpp"

namespace nos::utilities
{

struct BoundedQueueNodeContext : RingNodeBase
{
	BoundedQueueNodeContext(nosFbNode const* node) : RingNodeBase(node, RingNodeBase::OnRestartType::RESET)
	{
	}
	std::string GetName() const override
	{
		return "BoundedQueue";
	}
	
	nosResult CopyFrom(nosCopyInfo* cpy) override {
		ResourceTypeManager::ResourceBase* slot = nullptr;
		nosResourceShareInfo outputResource = {};
		auto beginResult = CommonCopyFrom(cpy, &slot, &outputResource);
		if(beginResult != NOS_RESULT_SUCCESS)
			return beginResult;

		Ring->Manager->SendCopyCmdToGPU(slot, cpy, NodeId, &outputResource);

		cpy->CopyFromOptions.ShouldSetSourceFrameNumber = true;
		cpy->FrameNumber = slot->FrameNumber;

		Ring->EndPop(slot);
		SendScheduleRequest(1);
		return NOS_RESULT_SUCCESS;
	}

	nosResult ExecuteNode(nosNodeExecuteParams* params) override {
		return ExecuteRingNode(params, false, NOS_NAME_STATIC("BoundedQueue"), false);
	}
};

nosResult RegisterBoundedQueue(nosNodeFunctions* functions)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("BoundedQueue"), BoundedQueueNodeContext, functions)
		return NOS_RESULT_SUCCESS;
}


}