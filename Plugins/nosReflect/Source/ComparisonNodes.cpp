// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include "TypeCommon.h"

namespace nos::reflect
{
template<CompareResult TCompareResult>
struct ComparisonNode : NodeContext
{
	std::optional<nos::TypeInfo> Type = std::nullopt;

	nos::Buffer Value;
	
	ComparisonNode(nosFbNodePtr inNode) : NodeContext(inNode)
	{
		for (auto pin : *inNode->pins())
		{
			if(pin->name()->string_view() == NSN_A)
			{
				if (pin->type_name()->string_view() != NSN_TypeNameGeneric)
					Type = nos::TypeInfo(nos::Name(pin->type_name()->string_view()));
			}
		}
	}
	
	void OnPinUpdated(nosPinUpdate const* update) override
	{
		if (Type || update->UpdatedField != NOS_PIN_FIELD_TYPE_NAME)
			return;
		if (update->PinName == NSN_A && update->TypeName != NSN_TypeNameGeneric)
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
		bool yes = CompareFlatBuffers<TCompareResult>(type, aPtr, bPtr);
		nos::Name pinName;
		if constexpr (TCompareResult == CompareResult::Equal)
			pinName = NSN_IsEqual;
		else if constexpr (TCompareResult == CompareResult::Less)
			pinName = NSN_IsLess;
		else if constexpr (TCompareResult == CompareResult::Greater)
			pinName = NSN_IsGreater;
		nosEngine.SetPinValueByName(NodeId, pinName, nosBuffer{.Data = &yes, .Size = sizeof(bool)});
		return NOS_RESULT_SUCCESS;
	}
};

nosResult RegisterIsEqual(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NSN_IsEqual, ComparisonNode<CompareResult::Equal>, fn);
	return NOS_RESULT_SUCCESS;
}

nosResult RegisterLessThan(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NSN_LessThan, ComparisonNode<CompareResult::Less>, fn);
	return NOS_RESULT_SUCCESS;
}

nosResult RegisterGreaterThan(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NSN_GreaterThan, ComparisonNode<CompareResult::Greater>, fn);
	return NOS_RESULT_SUCCESS;
}

} // namespace nos