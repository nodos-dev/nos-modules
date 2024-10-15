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

struct RingBufferNodeContext : RingNodeBase
{
	RingBufferNodeContext(nosFbNode const* node) : RingNodeBase(node, RingNodeBase::OnRestartType::WAIT_UNTIL_FULL)
	{
	}
	std::string GetName() const override
	{
		return "RingBuffer";
	}

	nosResult CopyFrom(nosCopyInfo* cpy) override {
		ResourceTypeManager::ResourceBase* slot = nullptr;
		nosResourceShareInfo outputResource = {};
		auto beginResult = CommonCopyFrom(cpy, &slot, &outputResource);
		if (beginResult != NOS_RESULT_SUCCESS)
			return beginResult;

		Ring->Manager->WaitForDownloadToEnd(slot, "RingBuffer", NodeName.AsString(), cpy);

		cpy->CopyFromOptions.ShouldSetSourceFrameNumber = true;
		cpy->FrameNumber = slot->FrameNumber;

		LastPopped = slot;
		SendScheduleRequest(1);
		return NOS_RESULT_SUCCESS;
	}

	nosResult ExecuteNode(nosNodeExecuteParams* params) override {
		return ExecuteRingNode(params, true, NOS_NAME_STATIC("RingBuffer"), true);
	}

	void OnEndFrame(nosUUID pinId, bool causedByCancel) override
	{
		if (pinId == PinName2Id[NOS_NAME_STATIC("Output")])
		{
			if (!LastPopped)
				return;
			Ring->EndPop(LastPopped);
			LastPopped = nullptr;
		}
	}
};

nosResult RegisterRingBuffer(nosNodeFunctions* functions)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("RingBuffer"), RingBufferNodeContext, functions)
		return NOS_RESULT_SUCCESS;
}


}