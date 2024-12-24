// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#pragma once

#include <Nodos/PluginHelpers.hpp>

// External
#include <glm/glm.hpp> // TODO: Ring no longer needs glm::mat4 colormatrix. Remove this
#include <nosVulkanSubsystem/Helpers.hpp>

#include "Ring.h"
#include <nosUtil/Stopwatch.hpp>

namespace nos::utilities
{

struct InterlacingBoundedTextureQueue : RingNodeBase
{
	InterlacingBoundedTextureQueue(nosFbNode const* node) : RingNodeBase(node, RingNodeBase::OnRestartType::RESET)
	{
	}

	std::string GetName() const override
	{
		return "InterlacingBoundedTextureQueue";
	}

	void OnPathStart() override
	{
		RingNodeBase::OnPathStart();
		{
			std::unique_lock lock(ArrivedFramesMutex);
			ArrivedFramesQueue = {};
		}
		LastServedFrameNumberBase = 0;
	}

	uint64_t LastServedFrameNumberBase;
	std::mutex ArrivedFramesMutex;
	std::queue<nosTextureFieldType> ArrivedFramesQueue;
	
	nosResult CopyFrom(nosCopyInfo* cpy) override
	{
		nosTextureFieldType currentField;
		{
			std::unique_lock lock(ArrivedFramesMutex);
			if (ArrivedFramesQueue.empty())
				return NOS_RESULT_PENDING;
			currentField = ArrivedFramesQueue.front();
			ArrivedFramesQueue.pop();
		}
		if (currentField == NOS_TEXTURE_FIELD_TYPE_ODD)
		{
			cpy->CopyFromOptions.ShouldSetSourceFrameNumber = true;
			cpy->FrameNumber = 2 * LastServedFrameNumberBase + 1;
			vkss::SetFieldType(cpy->ID, *cpy->PinData, currentField);
			return NOS_RESULT_SUCCESS;
		}

		ResourceInterface::ResourceBase* slot = nullptr;
		auto beginResult = CommonCopyFrom(cpy, &slot);
		if(beginResult != NOS_RESULT_SUCCESS || !slot)
			return beginResult;

		Ring->ResInterface->Copy(slot, cpy, NodeId);

		cpy->CopyFromOptions.ShouldSetSourceFrameNumber = true;
		LastServedFrameNumberBase = slot->FrameNumber;
		cpy->FrameNumber = 2 * LastServedFrameNumberBase;

		Ring->EndPop(slot);
		vkss::SetFieldType(cpy->ID, *cpy->PinData, currentField);
		SendScheduleRequest(1);
		return NOS_RESULT_SUCCESS;
	}

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		auto res = ExecuteRingNode(params, false, NOS_NAME_STATIC("InterlacingBoundedTextureQueue"), false);
		if (res == NOS_RESULT_SUCCESS)
		{
			std::unique_lock lock(ArrivedFramesMutex);
			ArrivedFramesQueue.push(NOS_TEXTURE_FIELD_TYPE_EVEN);
			ArrivedFramesQueue.push(NOS_TEXTURE_FIELD_TYPE_ODD);
		}
		return res;
	}
};

nosResult RegisterInterlacingBoundedTextureQueue(nosNodeFunctions* functions)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("InterlacingBoundedTextureQueue"), InterlacingBoundedTextureQueue, functions)
	return NOS_RESULT_SUCCESS;
}


}