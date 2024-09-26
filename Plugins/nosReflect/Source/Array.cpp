// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include "TypeCommon.h"

namespace nos::reflect
{
NOS_REGISTER_NAME(Array)

struct ArrayNode : NodeContext
{
	std::optional<nos::TypeInfo> Type = std::nullopt;
	size_t PinCount = 0;
	ArrayNode(const nosFbNode* inNode) : NodeContext(inNode)
	{
		for (auto& pin : Pins | std::views::values)
		{
			if (pin.ShowAs == fb::ShowAs::INPUT_PIN)
			{
				PinCount++;
			}
			if (pin.ShowAs == fb::ShowAs::OUTPUT_PIN)
			{
				nos::TypeInfo arrayType = nos::TypeInfo(pin.TypeName);
				if (arrayType.TypeName != NSN_VOID)
				{
					auto elementTypeName = arrayType.TypeName;
					Type = nos::TypeInfo(elementTypeName);
				}
			}
		}
		LoadPins();
		UpdateOutputVectorSize();
	}

	void OnNodeUpdated(const nosFbNode* inNode) override
	{
		auto oldPinCount = PinCount;
		PinCount = GetInputs().size();
		if (oldPinCount == PinCount)
			return;
		std::vector<const void*> values;
		for (auto* pin : *inNode->pins())
			if (pin->show_as() == fb::ShowAs::INPUT_PIN)
				values.push_back((void*)pin->data()->data());
		SendOutputArray(values);
	}

	nosResult OnResolvePinDataTypes(nosResolvePinDataTypesParams* params) override
	{
		auto setResolvedTypeNames = [params](nosName elementName) {
			for (size_t i = 0; i < params->PinCount; i++)
			{
				auto& pinInfo = params->Pins[i];
				std::string pinName = nosEngine.GetString(pinInfo.Name);
				if (pinName.find("Input") != std::string::npos)
				{
					pinInfo.OutResolvedTypeName = elementName;
				}
				if (pinName == "Output")
					pinInfo.OutResolvedTypeName = nos::Name('[' + nos::Name(elementName).AsString() + ']');
			}
		};
		nos::TypeInfo info(params->IncomingTypeName);
		if (params->InstigatorPinName == NSN_Output)
		{
			if (info->BaseType != NOS_BASE_TYPE_ARRAY)
			{
				strcpy(params->OutErrorMessage, "Output pin must be an array type");
				return NOS_RESULT_FAILED;
			}
			nos::Name elementName = info->ElementType->TypeName;
			Type = nos::TypeInfo(elementName);
			setResolvedTypeNames(elementName);
		}

		for (auto& pin : GetInputs())
		{
			if (pin->ShowAs == nos::fb::ShowAs::INPUT_PIN && pin->Id == params->InstigatorPinId)
			{
				if (info->BaseType == NOS_BASE_TYPE_ARRAY)
				{
					strcpy(params->OutErrorMessage, "Input pin must not be an array type");
					return NOS_RESULT_FAILED;
				}
				Type = nos::TypeInfo(params->IncomingTypeName);
				setResolvedTypeNames(params->IncomingTypeName);
				break;
			}
		}
		UpdateOutputVectorSize();
		return NOS_RESULT_SUCCESS;
	}

	void OnPinUpdated(nosPinUpdate const* update) override
	{
		if (Type || update->UpdatedField != NOS_PIN_FIELD_TYPE_NAME)
			return;
		
		auto newTypeName = Name(update->TypeName);
		auto typeInfo = nos::TypeInfo(newTypeName);
		if (typeInfo->BaseType != NOS_BASE_TYPE_ARRAY)
		{
			newTypeName = nos::Name('[' + nos::Name(update->TypeName).AsString() + ']');
		}
		Type = nos::TypeInfo(newTypeName);
	}

	std::vector<const NodePin*> GetInputs()
	{
		std::vector<const NodePin*> inputs;
		size_t i = 0;
		while (true)
		{
			auto pin = GetPin(nos::Name("Input " + std::to_string(i)));
			if (!pin)
				break;
			inputs.push_back(pin);
			i++;
		}
		return inputs;
	}

	bool SendOutputArray(std::vector<const void*> const& values)
	{
		auto outPin = GetPin(NSN_Output);
		if (!outPin || !Type)
			return false;

		auto outval = GenerateVector(*Type, values);

		auto vec = (flatbuffers::Vector<flatbuffers::Offset<flatbuffers::Table>>*)(outval.data());
		assert(GetInputs().size() == vec->size());
		nosEngine.SetPinValue(outPin->Id, {outval.data(), outval.size()});
		return true;
	}

	void LoadPins()
	{
		flatbuffers::FlatBufferBuilder fbb;
		std::vector<flatbuffers::Offset<PartialPinUpdate>> updates;

		for (auto& [id, p] : Pins)
			if (p.IsOrphan)
				updates.push_back(CreatePartialPinUpdate(fbb, &p.Id, 0, fb::CreateOrphanState(fbb, false)));

		if (!updates.empty())
		{
			HandleEvent(CreateAppEvent(
				fbb, CreatePartialNodeUpdateDirect(fbb, &NodeId, ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, &updates)));
		}
	}

	void UpdateOutputVectorSize()
	{
		if (!Type)
			return;

		auto typeName = Name(Type->TypeName).AsString();
		auto outputType = "[" + typeName + "]";

		nosBuffer value;
		std::vector<u8> data;

		if (NOS_RESULT_SUCCESS == nosEngine.GetDefaultValueOfType(Type->TypeName, &value))
		{
			data = std::vector<u8>{(u8*)value.Data, (u8*)value.Data + value.Size};
		}

		std::vector<const void*> datas;
		for (unsigned int i = 0; i < GetInputs().size(); i++) {
			datas.push_back(data.data());
		}
		SendOutputArray(datas);
	}

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		if (!Type)
			return NOS_RESULT_FAILED;
		std::vector<const void*> values;
		for (size_t i = 0; i < params->PinCount; ++i)
		{
			if (params->Pins[i].Name == NSN_Output)
				continue;
			values.push_back(params->Pins[i].Data->Data);
		}
		return SendOutputArray(values) ? NOS_RESULT_SUCCESS : NOS_RESULT_FAILED;
	}

	void OnMenuRequested(const nosContextMenuRequest* request) override
	{
		auto inputs = GetInputs();

		flatbuffers::FlatBufferBuilder fbb;
		std::vector<flatbuffers::Offset<nos::ContextMenuItem>> fields;
		std::string add = "Add Input " + std::to_string(inputs.size());
		fields.push_back(nos::CreateContextMenuItemDirect(fbb, add.c_str(), 1));
		if (inputs.size() > 0)
		{
			std::string remove = "Remove Input " + std::to_string(inputs.size() - 1);
			fields.push_back(nos::CreateContextMenuItemDirect(fbb, remove.c_str(), 2));
		}
		HandleEvent(CreateAppEvent(
			fbb,
			nos::app::CreateAppContextMenuUpdateDirect(fbb, &NodeId, request->pos(), request->instigator(), &fields)));
	}

	void SendAddElementRequest() {
		auto inputs = GetInputs();
		flatbuffers::FlatBufferBuilder fbb;

		nosBuffer value;
		std::vector<u8> data;
		nos::Name typeName = Type ? Name(Type->TypeName) : NSN_VOID;
		if (NOS_RESULT_SUCCESS == nosEngine.GetDefaultValueOfType(typeName, &value))
			data = std::vector<u8>{ (u8*)value.Data, (u8*)value.Data + value.Size };

		auto outputType = "[" + typeName.AsString() + "]";
		auto name = "Input " + std::to_string(inputs.size());
		nosUUID id = nosEngine.GenerateID();

		std::vector pins = {
			nos::fb::CreatePinDirect(fbb,
									 &id,
									 name.c_str(),
									 typeName.AsCStr(),
									 nos::fb::ShowAs::INPUT_PIN,
									 nos::fb::CanShowAs::INPUT_PIN_OR_PROPERTY,
									 0,
									 0,
									 &data),
		};
		HandleEvent(
			CreateAppEvent(fbb, CreatePartialNodeUpdateDirect(fbb, &NodeId, ClearFlags::NONE, 0, &pins)));
		UpdateOutputVectorSize();
	}

	void SendRemoveElementRequest() {
		auto inputs = GetInputs();
		if (inputs.size() == 0) {
			nosEngine.LogE("You can't remove element from an empty array");
			return;
		}
		flatbuffers::FlatBufferBuilder fbb;

		std::vector<fb::UUID> id = { inputs.back()->Id };
		HandleEvent(
			CreateAppEvent(fbb, CreatePartialNodeUpdateDirect(fbb, &NodeId, ClearFlags::NONE, &id)));
	}

	void OnMenuCommand(nosUUID itemID, uint32_t cmd) override
	{
		switch (cmd)
		{
		case 1: // Add Field
		{
			SendAddElementRequest();
		}
		break;
		case 2: // Remove Field
		{
			SendRemoveElementRequest();
		}
		break;
		}
	}

	static nosResult GetFunctions(size_t* outCount, nosName* outFunctionNames, nosPfnNodeFunctionExecute* outFunctions) {
		*outCount = 2;
		if (!outFunctionNames || !outFunctions)
			return NOS_RESULT_SUCCESS;

		outFunctionNames[0] = NOS_NAME_STATIC("Add Element");
		outFunctions[0] = [](void* ctx, nosFunctionExecuteParams* params)
			{
				auto nodeCtx = (ArrayNode*)ctx;
				nodeCtx->SendAddElementRequest();
				return NOS_RESULT_SUCCESS;
			};

		outFunctionNames[1] = NOS_NAME_STATIC("Remove Element");
		outFunctions[1] = [](void* ctx, nosFunctionExecuteParams* params)
			{
				auto nodeCtx = (ArrayNode*)ctx;
				nodeCtx->SendRemoveElementRequest();
				return NOS_RESULT_SUCCESS;
			};

		return NOS_RESULT_SUCCESS;
	}
};

nosResult RegisterArray(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NSN_Array, ArrayNode, fn);
	return NOS_RESULT_SUCCESS;
}

} // namespace nos::engine