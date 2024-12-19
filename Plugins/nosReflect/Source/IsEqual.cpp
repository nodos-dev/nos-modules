// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include "TypeCommon.h"

namespace nos::reflect
{
struct IsEqualNode : NodeContext
{
	std::optional<nos::TypeInfo> Type = std::nullopt;

	nos::Buffer Value;
	
	IsEqualNode(const nosFbNode* inNode) : NodeContext(inNode)
	{
		for (auto pin : *inNode->pins())
		{
			if(pin->name()->string_view() == NSN_A)
			{
				if (pin->type_name()->string_view() != NSN_VOID)
					Type = nos::TypeInfo(nos::Name(pin->type_name()->string_view()));
			}
		}
	}
	
	void OnPinUpdated(nosPinUpdate const* update) override
	{
		if (Type || update->UpdatedField != NOS_PIN_FIELD_TYPE_NAME)
			return;
		if (update->PinName == NSN_A && update->TypeName != NSN_VOID)
			Type = nos::TypeInfo(update->TypeName);
	}
	
	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		if (!Type)
			return NOS_RESULT_FAILED;

		auto& type = *Type;

		auto pins = NodeExecuteParams(params);
		auto& A = pins[NSN_A];
		auto& B = pins[NSN_B];
		void* aPtr, *bPtr;
		if (!type->ByteSize && type->BaseType == NOS_BASE_TYPE_STRUCT)
		{
			aPtr = (void*)flatbuffers::GetRoot<flatbuffers::Table>(A.Data->Data);
			bPtr = (void*)flatbuffers::GetRoot<flatbuffers::Table>(B.Data->Data);
		}
		else
		{
			aPtr = A.Data->Data;
			bPtr = B.Data->Data;
		}
		bool isEqual = AreFlatBuffersEqual(type, aPtr, bPtr);
		nosEngine.SetPinValueByName(NodeId, NSN_IsEqual, nosBuffer{.Data = &isEqual, .Size = sizeof(bool)});
		return NOS_RESULT_SUCCESS;
	}
};

nosResult RegisterIsEqual(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NSN_IsEqual, IsEqualNode, fn);
	return NOS_RESULT_SUCCESS;
}

} // namespace nos