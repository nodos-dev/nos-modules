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
	static constexpr nosTextureInfo SampleTexture = nosTextureInfo{
		.Width = 1920,
		.Height = 1080,
		.Format = NOS_FORMAT_R16G16B16A16_SFLOAT,
		.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_SRC | NOS_IMAGE_USAGE_TRANSFER_DST),
	};
	BoundedQueueNodeContext(nosFbNode const* node) : RingNodeBase(node, RingNodeBase::OnRestartType::RESET)
	{
	}
	std::string GetName() const override
	{
		return "BoundedQueue";
	}
	
	nosResult CopyFrom(nosCopyInfo* cpy) override {
		TRing::Resource* slot = nullptr;
		nosResourceShareInfo outputResource = {};
		auto beginResult = CopyFromBegin(cpy, &slot, &outputResource);
		if(beginResult != NOS_RESULT_SUCCESS)
			return beginResult;

		nosCmd cmd;
		nosCmdBeginParams beginParams = { NOS_NAME("BoundedQueue"), NodeId, &cmd };
		nosVulkan->Begin2(&beginParams);
		nosVulkan->Copy(cmd, &slot->Res, &outputResource, 0);
		nosCmdEndParams end{ .ForceSubmit = NOS_TRUE, .OutGPUEventHandle = &slot->Params.WaitEvent };
		nosVulkan->End(cmd, &end);

		switch (type)
		{
		case ResourceType::Buffer:
		{
			nosTextureFieldType outFieldType = slot->Res.Info.Buffer.FieldType;
			auto outputBufferDesc = *static_cast<sys::vulkan::Buffer*>(cpy->PinData->Data);
			outputBufferDesc.mutate_field_type((sys::vulkan::FieldType)outFieldType);
			nosEngine.SetPinValue(cpy->ID, nos::Buffer::From(outputBufferDesc));
			break;
		}
		case ResourceType::Texture:
		{
			nosTextureFieldType outFieldType = slot->Res.Info.Texture.FieldType;
			auto outputTextureDesc = static_cast<sys::vulkan::Texture*>(cpy->PinData->Data);
			auto output = vkss::DeserializeTextureInfo(outputTextureDesc);
			output.Info.Texture.FieldType = slot->Res.Info.Texture.FieldType;
			sys::vulkan::TTexture texDef = vkss::ConvertTextureInfo(output);
			texDef.unscaled = true;
			nosEngine.SetPinValue(cpy->ID, Buffer::From(texDef));
			break;
		}
		case ResourceType::Generic:
		default:
			return NOS_RESULT_FAILED;
		}

		cpy->CopyFromOptions.ShouldSetSourceFrameNumber = true;
		cpy->FrameNumber = slot->FrameNumber;

		Ring->EndPop(slot);
		SendScheduleRequest(1);
		return NOS_RESULT_SUCCESS;
	}

	nosResult ExecuteNode(nosNodeExecuteParams* params) override {
		return ExecuteRingNode(params, false, NOS_NAME_STATIC("BoundedQueue"), false);
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

nosResult RegisterBoundedQueue(nosNodeFunctions* functions)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("BoundedQueue"), BoundedQueueNodeContext, functions)
		return NOS_RESULT_SUCCESS;
}


}