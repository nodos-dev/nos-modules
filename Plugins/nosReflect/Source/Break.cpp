// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include "TypeCommon.h"

namespace nos::reflect
{
NOS_REGISTER_NAME(Break)

struct BreakNode : NodeContext
{
	std::optional<nosTypeInfo> Type = std::nullopt;

	BreakNode(const nosFbNode* node) : NodeContext(node)
	{
		for (auto* pin : *node->pins())
		{
			if (pin->show_as() != fb::ShowAs::INPUT_PIN)
				continue;
            auto typeName = nos::Name(pin->type_name()->string_view());
			if (NSN_VOID == typeName)
				break;
			Type = nosTypeInfo{};
            nosEngine.GetTypeInfo(typeName, &*Type);
			LoadPins(false);
            break;
        }
	}

    void OnPinUpdated(const nosPinUpdate* update) override
	{
		if (Type)
			return;
		if (update->UpdatedField == NOS_PIN_FIELD_TYPE_NAME)
		{
			if (update->PinName != NSN_Input)
				return;
			Type = nosTypeInfo{};
			nosEngine.GetTypeInfo(update->TypeName, &*Type);
			LoadPins(true);
		}
	}

    nosResult OnResolvePinDataTypes(nosResolvePinDataTypesParams* params) override
	{
		nosTypeInfo info = {};
		nosEngine.GetTypeInfo(params->IncomingTypeName, &info);
		switch (info.BaseType)
		{
		case NOS_BASE_TYPE_ARRAY:
		case NOS_BASE_TYPE_STRUCT: return NOS_RESULT_SUCCESS;
		default: return NOS_RESULT_FAILED;
		}
	};

    void LoadPins(bool setDisplayName)
    {
        flatbuffers::FlatBufferBuilder fbb;
		std::vector<flatbuffers::Offset<PartialPinUpdate>> pinsToUpdate;
		std::vector<::flatbuffers::Offset<nos::fb::Pin>> pinsToCreate;
        std::vector<fb::UUID> pinsToDelete;

		std::unordered_set<nos::Name> acceptedPins;
	
		if (auto* in = GetPin(NSN_Input))
		{
			pinsToUpdate.push_back(CreatePartialPinUpdate(fbb, &in->Id, 0, fb::CreateOrphanState(fbb, false)));
			acceptedPins.insert(in->Name);
		}

        if (Type->BaseType == NOS_BASE_TYPE_STRUCT)
		{
			for (int i = 0; i < Type->FieldCount; ++i)
			{
				auto& field = Type->Fields[i];
				acceptedPins.insert(field.Name);
				if (auto pin = GetPin(field.Name))
				{
					if (!pin->IsOrphan && pin->TypeName == field.Type->TypeName)
						continue;
					pinsToUpdate.push_back(CreatePartialPinUpdateDirect(fbb,
																		&pin->Id,
																		0,
																		nos::fb::CreateOrphanStateDirect(fbb, false),
																		nos::Name(field.Type->TypeName).AsCStr()));
				}
				else
				{
					nosUUID newPinId;
					nosEngine.GenerateID(&newPinId);

					std::vector<uint8_t> data =
						std::vector((const uint8_t*)Type->DefaultVals[i].Data,
									(const uint8_t*)Type->DefaultVals[i].Data + Type->DefaultVals[i].Size);

					pinsToCreate.push_back(fb::CreatePinDirect(fbb,
															   &newPinId,
															   nos::Name(field.Name).AsCStr(),
															   nos::Name(field.Type->TypeName).AsCStr(),
															   fb::ShowAs::OUTPUT_PIN,
															   fb::CanShowAs::OUTPUT_PIN_OR_PROPERTY,
															   0,
															   0,
															   &data,
															   0,
															   0,
															   0,
															   &data));
				}
			}
		}
		auto type = "Break " + nos::Name(Type->TypeName).AsString();
		for (auto& [id, pin] : Pins)
			if (!acceptedPins.contains(pin.Name))
				pinsToDelete.push_back(id);

        HandleEvent(CreateAppEvent(
			fbb, CreatePartialNodeUpdateDirect(fbb, &NodeId, ClearFlags::NONE, &pinsToDelete, &pinsToCreate, 0, 0, 0, 0, 0, &pinsToUpdate, 0, 0, 0, setDisplayName ? type.c_str() : 0)));
    }

	std::unordered_map<nosUUID, nos::Buffer> LastServedPinValues;

	void SetPinValueCached(const nosUUID& pinId, nosBuffer value)
	{
		auto it = LastServedPinValues.find(pinId);
		if (it != LastServedPinValues.end() && it->second == value)
			return;
        nosEngine.SetPinValueDirect(pinId, value);
		LastServedPinValues[pinId] = value;
	}
	
    void SetOutputValues(const nosBuffer* buf)
    {
        if(!buf)
			return;

		auto data = (const u8*)buf->Data;
        
        switch (Type->BaseType)
        {
        //case NOS_BASE_TYPE_ARRAY:
        //{
        //    const flatbuffers::Vector<u8>* vec = (flatbuffers::Vector<u8>*)(data);
        //    const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::Table>>* vect = (const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::Table>>*)vec;
        //    SetOutputs(vec->size());
        //    for (int i = 0; i < vec->size(); ++i)
        //    {
        //        auto pin = pins.find(&PinInfo::name, nos::Name("Output " + std::to_string(i)));
        //        if(!pin) 
        //            continue;
        //        auto pinId  = pin->id;
        //        if (info.ElementType->ByteSize)
        //        {
        //            auto data = vec->data() + i * info.ElementType->ByteSize;
        //            SetPinValueCached(pinId, { (void*)data, info.ElementType->ByteSize });
        //        }
        //        else
        //        {
        //        	//TODO: Strings
        //            flatbuffers::FlatBufferBuilder fbb;
        //            fbb.Finish(flatbuffers::Offset<flatbuffers::Table>(CopyTable(fbb, info.ElementType, vect->Get(i))));
        //            nos::Buffer buf = fbb.Release();
        //            SetPinValueCached(pinId, { (void*)buf.Data(), buf.Size() });
        //        }
        //    }
        //    break;
        //}
        case NOS_BASE_TYPE_STRUCT:
        {
            auto root = Type->ByteSize ? (flatbuffers::Table*)data : flatbuffers::GetRoot<flatbuffers::Table>(data);
            for (int i = 0; i < Type->FieldCount; ++i)
            {
				auto& field = Type->Fields[i];
				auto pin = GetPin(field.Name);
				if (!pin)
					continue;
				if (!Type->ByteSize && !root->CheckField(field.Offset)) 
				{
					if (Type->DefaultVals[i].Size)
						SetPinValueCached(pin->Id, Type->DefaultVals[i]);
					continue;
				}
				
				if (field.Type->ByteSize)
				{
					auto data = !Type->ByteSize ? root->GetStruct<u8*>(field.Offset) : ((u8*)root + field.Offset);
					SetPinValueCached(pin->Id, { (void*)data, field.Type->ByteSize });
				}
                else
                {
                	nos::Buffer buf;
                    if (field.Type->BaseType == NOS_BASE_TYPE_STRING)
                    {
	                    auto str = root->GetPointer<const ::flatbuffers::String *>(field.Offset)->string_view();
						buf = Buffer(str.data(), str.size() + 1);
                    }
                	else
                	{
						flatbuffers::FlatBufferBuilder fbb;
                		fbb.Finish(flatbuffers::Offset<flatbuffers::Table>(CopyTable(fbb, field.Type, root->GetPointer<flatbuffers::Table*>(field.Offset))));
						buf = fbb.Release();
                	}
					SetPinValueCached(pin->Id, buf);
                }
            }
        }
        }
    }

    nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		if(!Type)
			return NOS_RESULT_SUCCESS;
		auto pins = NodeExecuteArgs(args);
		SetOutputValues(pins[NSN_Input].Data);
		return NOS_RESULT_SUCCESS;
	}

    void OnPathCommand(const nosPathCommand* command) override
	{
		switch (command->Event)
		{
		case NOS_ARRAY_RESIZE:
			if (auto pin = GetPin(NSN_Input))
			{
				//SetOutputs(command->ArrayPathCommandArgs.ArraySize);
			}
			break;
		}
	}
};

nosResult RegisterBreak(nosNodeFunctions* fn)
{ 
    NOS_BIND_NODE_CLASS(NSN_Break, BreakNode, fn);
	return NOS_RESULT_SUCCESS;
}

} // namespace nos