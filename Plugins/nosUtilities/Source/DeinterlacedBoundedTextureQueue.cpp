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

struct DeinterlacedBoundedTextureQueueNode : RingNodeBase
{
	DeinterlacedBoundedTextureQueueNode(nosFbNode const* node) : RingNodeBase(node, RingNodeBase::OnRestartType::RESET)
	{
	}

	std::string GetName() const override
	{
		return "DeinterlacedBoundedTextureQueue";
	}

	void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer value) override
	{
		if (pinName == NOS_NAME("ShouldInterlace"))
		{
			auto& shouldInterlace = *InterpretPinValue<bool>(value);
			if (ShouldInterlace != shouldInterlace)
			{
				ShouldInterlace = shouldInterlace;
				nosEngine.RecompilePath(NodeId);
			}
		}
	}

	void OnPathStart() override
	{
		RingNodeBase::OnPathStart();
		{
			std::unique_lock lock(ArrivedFramesMutex);
			ArrivedFramesQueue = {};
			FrameArrivedCond.notify_all();
		}
		LastServedFrameNumberBase = 0;
	}

	void OnPathStop() override
	{
		RingNodeBase::OnPathStop();
		FrameArrivedCond.notify_all();
	}

	uint64_t LastServedFrameNumberBase;
	std::mutex ArrivedFramesMutex;
	std::queue<nosTextureFieldType> ArrivedFramesQueue;
	std::condition_variable FrameArrivedCond;
	std::atomic_bool ShouldInterlace = false;
	
	nosResult CopyFrom(nosCopyInfo* cpy) override
	{
		if (!Ring)
			return NOS_RESULT_FAILED;
		nosTextureFieldType currentField;
		{
			std::unique_lock lock(ArrivedFramesMutex);
			FrameArrivedCond.wait_for(lock, std::chrono::milliseconds(1000), [this] { return !ArrivedFramesQueue.empty() || this->Ring->Exit; });
			if (ArrivedFramesQueue.empty())
				return NOS_RESULT_FAILED;
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
		if (beginResult != NOS_RESULT_SUCCESS || !slot)
			return beginResult;

		Ring->ResInterface->Copy(slot, cpy, NodeId);

		cpy->CopyFromOptions.ShouldSetSourceFrameNumber = true;
		LastServedFrameNumberBase = slot->FrameNumber;
		cpy->FrameNumber = (ShouldInterlace + 1) * LastServedFrameNumberBase;

		Ring->EndPop(slot);
		if (currentField != NOS_TEXTURE_FIELD_TYPE_UNKNOWN)
			vkss::SetFieldType(cpy->ID, *cpy->PinData, currentField);
		SendScheduleRequest(1);
		return NOS_RESULT_SUCCESS;
	}

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		auto res = ExecuteRingNode(params, false, NOS_NAME_STATIC("DeinterlacedBoundedTextureQueue"), false);
		if (res == NOS_RESULT_SUCCESS)
		{
			std::unique_lock lock(ArrivedFramesMutex);
			if (ShouldInterlace)
			{
				ArrivedFramesQueue.push(NOS_TEXTURE_FIELD_TYPE_EVEN);
				ArrivedFramesQueue.push(NOS_TEXTURE_FIELD_TYPE_ODD);
			}
			else
			{
				ArrivedFramesQueue.push(NOS_TEXTURE_FIELD_TYPE_UNKNOWN);
			}
			FrameArrivedCond.notify_one();
		}
		return res;
	}
};

nosResult RegisterDeinterlacedBoundedTextureQueue(nosNodeFunctions* functions)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("DeinterlacedBoundedTextureQueue"), DeinterlacedBoundedTextureQueueNode, functions)
	return NOS_RESULT_SUCCESS;
}


}