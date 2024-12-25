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

struct DeinterlacedBufferRingNode : RingNodeBase
{
	DeinterlacedBufferRingNode(nosFbNode const* node) : RingNodeBase(node, RingNodeBase::OnRestartType::WAIT_UNTIL_FULL)
	{
	}
	~DeinterlacedBufferRingNode()
	{
		NOS_SOFT_CHECK(LastPopped == nullptr);
	}

	ResourceInterface::ResourceBase* LastPopped = nullptr;

	std::string GetName() const override
	{
		return "DeinterlacedBufferRing";
	}
	
	void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer value) override
	{
		if (pinName == NOS_NAME("ShouldDeinterlace"))
		{
			auto& shouldDeinterlace = *InterpretPinValue<bool>(value);
			if (ShouldDeinterlace != shouldDeinterlace)
			{
				ShouldDeinterlace = shouldDeinterlace;
				nosEngine.RecompilePath(NodeId);
			}
		}
	}

	nosResult CopyFrom(nosCopyInfo* cpy) override {
		NOS_SOFT_CHECK(LastPopped == nullptr);
		ResourceInterface::ResourceBase* slot = nullptr;
		auto beginResult = CommonCopyFrom(cpy, &slot);
		if (beginResult != NOS_RESULT_SUCCESS || !slot)
			return beginResult;

		Ring->ResInterface->WaitForDownloadToEnd(slot, "DeinterlacedBufferRing", NodeName.AsString(), cpy);

		cpy->CopyFromOptions.ShouldSetSourceFrameNumber = true;
		cpy->FrameNumber = slot->FrameNumber;

		LastPopped = slot;
		SendScheduleRequest(1);
		return NOS_RESULT_SUCCESS;
	}

	nosResult ExecuteNode(nosNodeExecuteParams* params) override {
		nosResult res = NOS_RESULT_SUCCESS;
		auto pins = NodeExecuteParams(params);
		auto fieldType = *pins.GetPinData<nosTextureFieldType>(NOS_NAME("FieldType"));
		if (ShouldDeinterlace)
		{
			if (LastFieldType == NOS_TEXTURE_FIELD_TYPE_UNKNOWN)
				LastFieldType = fieldType;
			else if (fieldType == LastFieldType)
			{
				nosEngine.LogW("Field mismatch. Waiting for a new frame.");
				SendScheduleRequest(0);
				return NOS_RESULT_FAILED;
			}
			LastFieldType = fieldType;
		}
		if (!ShouldDeinterlace || fieldType == NOS_TEXTURE_FIELD_TYPE_ODD)
			res = ExecuteRingNode(params, true, NOS_NAME_STATIC("DeinterlacedBufferRing"), true);
		else
			SendScheduleRequest(1);
		return res;
	}

	std::atomic_bool ShouldDeinterlace = false;
	nosTextureFieldType LastFieldType = NOS_TEXTURE_FIELD_TYPE_UNKNOWN;

	void OnPathStart() override
	{
		RingNodeBase::OnPathStart();
		LastFieldType = NOS_TEXTURE_FIELD_TYPE_UNKNOWN;
	}

	void OnEndFrame(nosUUID pinId, nosEndFrameCause cause) override
	{
		RingNodeBase::OnEndFrame(pinId, cause);
		if (pinId == PinName2Id[NOS_NAME_STATIC("Output")])
		{
			if (!LastPopped)
				return;
			Ring->EndPop(LastPopped);
			LastPopped = nullptr;
		}
	}
};

nosResult RegisterDeinterlacedBufferRing(nosNodeFunctions* functions)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("DeinterlacedBufferRing"), DeinterlacedBufferRingNode, functions)
		return NOS_RESULT_SUCCESS;
}


}