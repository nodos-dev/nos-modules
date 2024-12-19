// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include "TypeCommon.h"

namespace nos::reflect
{
NOS_REGISTER_NAME(IndexOf)
NOS_REGISTER_NAME(InputArray)

extern nos::Name NSN_Index;
extern nos::Name NSN_Value;

struct IndexOfNode : NodeContext
{
	std::optional<nos::TypeInfo> Type = std::nullopt;

	nos::Buffer Value;
	
	IndexOfNode(const nosFbNode* inNode) : NodeContext(inNode)
	{
		for (auto pin : *inNode->pins())
		{
			if(pin->name()->string_view() == NSN_InputArray)
			{
				if (pin->type_name()->string_view() != NSN_VOID)
				{
					Type = nos::TypeInfo(nos::Name(pin->type_name()->string_view()));
				}
			}
			else if (pin->name()->string_view() == NSN_Index)
			{
				if (flatbuffers::IsFieldPresent(pin, fb::Pin::VT_DATA))
				{
					Value = nos::Buffer(pin->data()->Data(), pin->data()->size());
				}
			}
		}
	}

	nosResult OnResolvePinDataTypes(nosResolvePinDataTypesParams* params) override
	{
		if (params->InstigatorPinName == NSN_InputArray)
		{
			nos::TypeInfo info(params->IncomingTypeName);
			if (info->BaseType != NOS_BASE_TYPE_ARRAY)
			{
				strcpy(params->OutErrorMessage, "InputArray pin must be an array type");
				return NOS_RESULT_FAILED;
			}
			
			nos::Name elementName = info->ElementType->TypeName;

			for (size_t i = 0; i < params->PinCount; i++)
			{
				auto& pinInfo = params->Pins[i];
				if (pinInfo.Name == NSN_Value)
				{
					pinInfo.OutResolvedTypeName = elementName;
					break;
				}
			}

			return NOS_RESULT_SUCCESS;
		}
		else if (params->InstigatorPinName == NSN_Value)
		{
			nos::TypeInfo info(params->IncomingTypeName);
			if (info->BaseType == NOS_BASE_TYPE_ARRAY)
			{
				strcpy(params->OutErrorMessage, "Value pin must not be an array type");
				return NOS_RESULT_FAILED;
			}
			nos::Name arrayName = nos::Name("[" + nos::Name(info.TypeName).AsString() + "]");
			for (size_t i = 0; i < params->PinCount; i++)
			{
				auto& pinInfo = params->Pins[i];
				if (pinInfo.Name == NSN_InputArray)
				{
					pinInfo.OutResolvedTypeName = arrayName;
					break;
				}
			}

			return NOS_RESULT_SUCCESS;
		}
		return NOS_RESULT_FAILED;
	}
	
	void OnPinUpdated(nosPinUpdate const* update) override
	{
		if (Type || update->UpdatedField != NOS_PIN_FIELD_TYPE_NAME)
			return;
		if (update->PinName == NSN_Value)
		{
			Type = nos::TypeInfo(update->TypeName);
		}
		else if (update->PinName == NSN_InputArray)
		{
			auto newTypeName = nos::Name(update->TypeName);
			auto typeInfo = nos::TypeInfo(newTypeName);
			if (typeInfo->BaseType == NOS_BASE_TYPE_ARRAY)
			{
				newTypeName = typeInfo->ElementType->TypeName;
			}
			Type = nos::TypeInfo(newTypeName);
		}
	}

	void ClearOutputState()
	{
		SetPinOrphanState(NSN_Index, fb::PinOrphanStateType::ACTIVE, "");
		ClearNodeStatusMessages();
	}

	int FoundIndex = -2;
	
	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		if (!Type)
			return NOS_RESULT_FAILED;

		auto& type = *Type;

		auto pins = NodeExecuteParams(params);
		auto indexPinId = *GetPinId(NSN_Index);

		auto vec = (flatbuffers::Vector<uint8_t>*)(pins[NSN_InputArray].Data->Data);
		void* value = pins[NSN_Value].Data->Data;
		if (!type->ByteSize)
			value = (void*)flatbuffers::GetRoot<flatbuffers::Table>(value);
		int index =	-1;
		if (type->ByteSize)
		{
			for (size_t i = 0; i < vec->size(); ++i)
			{
				if (memcmp(vec->data() + i * type->ByteSize, value, type->ByteSize) == 0)
				{
					index = i;
					break;
				}
			}
		}
		else
		{
			auto vecOfTables = (flatbuffers::Vector<flatbuffers::Offset<flatbuffers::Table>>*)(vec);
			for (size_t i = 0; i < vecOfTables->size(); ++i)
			{
				auto elem = vecOfTables->Get(i);
				if (IsEqualTable(type, elem, (flatbuffers::Table*)value))
				{
					index = i;
					break;
				}
			}
		}
		if (index != FoundIndex)
		{
			if (index == -1)
			{
				SetNodeStatusMessage("No such value found in the array", fb::NodeStatusMessageType::FAILURE);
				SetPinOrphanState(NSN_Index, fb::PinOrphanStateType::PASSIVE, "No such value found in the array");
			}
			else
			{
				ClearOutputState();
			}
			FoundIndex = index;
		}	
		nosEngine.SetPinValue(indexPinId, nos::Buffer::From(FoundIndex));
		return NOS_RESULT_SUCCESS;
	}
};

nosResult RegisterIndexOf(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NSN_IndexOf, IndexOfNode, fn);
	return NOS_RESULT_SUCCESS;
}

} // namespace nos