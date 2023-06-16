#include "TypeCommon.h"

namespace mz::utilities
{

MZ_REGISTER_NAME_SPACED(Disarray, "mz.utilities.Disarray")
MZ_REGISTER_NAME(Input);
MZ_REGISTER_NAME_SPACED(VOID, "mz.fb.Void");

struct Disarray
{
    fb::UUID id;
    std::vector<fb::UUID> outputs;
    fb::UUID input;
    mzName type = MZN_VOID;

    Disarray(const mzFbNode* inNode)
    {
        id = *inNode->id();

        for (auto pin : *inNode->pins())
        {
            if (pin->show_as() == fb::ShowAs::OUTPUT_PIN)
            {
                outputs.push_back(*pin->id());
            }
            else
            {
                type = mzEngine.GetName(pin->type_name()->c_str());
                input = *pin->id();
            }
        }
    }
    
    void SetOutputs(const mzTypeInfo* structdef)
    {
        if(structdef->BaseType != MZ_BASE_TYPE_STRUCT) 
            return;
        if(outputs.size() == structdef->FieldCount)
            return;

        flatbuffers::FlatBufferBuilder fbb;
        std::vector<::flatbuffers::Offset<mz::fb::Pin>> pins;
        std::vector<fb::UUID> deleted = std::move(outputs);

        for(int i = 0; i < structdef->FieldCount; ++i)
        {
            auto& field = structdef->Fields[i];
            mzUUID id;
            mzEngine.GenerateID(&id);
            outputs.push_back(id);
            pins.push_back(fb::CreatePinDirect(fbb, &id, 
                            mzEngine.GetString(field.Name), 
                            mzEngine.GetString(field.Type->TypeName), 
                            fb::ShowAs::OUTPUT_PIN, fb::CanShowAs::OUTPUT_PIN_ONLY));
        }
        mzEngine.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &id, ClearFlags::NONE, &deleted, &pins)));
    }

    void SetOutputs(u32 sz)
    {
        const u32 curr = outputs.size();
        if(curr == sz) return;

        std::string arrayType = mzEngine.GetString(type);
        if(!(arrayType.starts_with('[') && arrayType.ends_with(']')))
            return;
        
        std::string typeName = std::string(arrayType.begin() + 1, arrayType.end() - 1);
        flatbuffers::FlatBufferBuilder fbb;
        std::vector<::flatbuffers::Offset<mz::fb::Pin>> pins;
        std::vector<fb::UUID> deleted;
        for (u32 i = curr; i < sz; ++i)
        {
            mzUUID id;
            mzEngine.GenerateID(&id);
            outputs.push_back(id);
            pins.push_back(fb::CreatePinDirect(fbb, &id, ("Output " + std::to_string(i)).c_str(), typeName.c_str(), fb::ShowAs::OUTPUT_PIN, fb::CanShowAs::OUTPUT_PIN_ONLY));
        }
        for (u32 i = sz; i < curr; ++i)
        {
            deleted.push_back(outputs.back());
            outputs.pop_back();
        }
        mzEngine.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &id, ClearFlags::NONE, &deleted, &pins)));
    }

    void Reset()
    {
        type = MZN_VOID;
        flatbuffers::FlatBufferBuilder fbb;
        std::vector<::flatbuffers::Offset<mz::fb::Pin>> pins = {
            fb::CreatePinDirect(fbb, &input, "Input",  "mz.fb.Void", fb::ShowAs::INPUT_PIN, fb::CanShowAs::INPUT_PIN_OR_PROPERTY),
        };
        mzEngine.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &id, ClearFlags::ANY, 0, &pins)));
    }
    
};

void RegisterDisarray(mzNodeFunctions* fn)
{
	fn->TypeName = MZN_Disarray;

	fn->OnNodeCreated = [](const mzFbNode* node, void** outCtxPtr) 
    {
        *outCtxPtr = new Disarray(node);
	};

	fn->OnNodeDeleted = [](void* ctx, mzUUID nodeId) 
    {
        delete (Disarray*)ctx;
	};

    fn->OnNodeUpdated = [](void* ctx, const mzFbNode* node)
    {
        *((Disarray*)ctx) = Disarray(node);
    };

	fn->ExecuteNode = [](void* ctx, const mzNodeExecuteArgs* args) -> mzResult 
    {
        auto c = (Disarray*)ctx;
        if(MZN_VOID == c->type)
            return MZ_RESULT_SUCCESS;
        auto pins = NodeExecuteArgs(args);

        mzTypeInfo info = {};
        mzEngine.GetTypeInfo(c->type, &info);
        auto input = pins[MZN_Input].Buf.Data;
        switch(info.BaseType)
        {
            case MZ_BASE_TYPE_ARRAY:
                {
                    auto vec  = flatbuffers::GetRoot<flatbuffers::Vector<u8>>(input);
                    auto vect = flatbuffers::GetRoot<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::Table>>>(input);
                    c->SetOutputs(vec->size());
                    if(!input) break;
                    for(int i = 0; i < vec->size(); ++i)
                    {
                        if(info.ElementType->ByteSize)
                        {
                            auto data = vec->data() + i * info.ElementType->ByteSize;
                            mzEngine.SetPinValue(c->outputs[i], { (void*)data, info.ElementType->ByteSize });
                        }
                        else
                        {
                            flatbuffers::FlatBufferBuilder fbb;
                            fbb.Finish(flatbuffers::Offset<flatbuffers::Table>(CopyTable(fbb, info.ElementType, vect->Get(i))));
                            mz::Buffer buf = fbb.Release();
                            mzEngine.SetPinValue(c->outputs[i], {(void*)buf.data(), buf.size()});
                        }
                    }
                }
                break;
            case MZ_BASE_TYPE_STRUCT:
                c->SetOutputs(&info);
                if(!input) break;
                {
                    auto root = flatbuffers::GetRoot<flatbuffers::Table>(input);
                    for (int i = 0; i < info.FieldCount; ++i)
                    {
                        auto& field = info.Fields[i];
                        if(!field.Type->ByteSize)
                        {
                            flatbuffers::FlatBufferBuilder fbb;
                            fbb.Finish(flatbuffers::Offset<flatbuffers::Table>(CopyTable(fbb, field.Type, root->GetPointer<flatbuffers::Table*>(field.Offset))));
                            mz::Buffer buf = fbb.Release();
                            mzEngine.SetPinValue(c->outputs[i], {(void*)buf.data(), buf.size()});
                        }
                        else
                        {
                            auto data = root->GetStruct<u8*>(field.Offset);
                            mzEngine.SetPinValue(c->outputs[i], { (void*)data, field.Type->ByteSize });
                        }
                    }
                }
                break;
            default:
                c->Reset();
        }
        return MZ_RESULT_SUCCESS;
	};

    fn->OnPinValueChanged = [](void* ctx, mzName pinName, mzBuffer* value)
    {
        auto c = (Disarray*)ctx;
        if(pinName != MZN_Input || MZN_VOID == c->type)
            return;

        mzTypeInfo info = {};
        mzEngine.GetTypeInfo(c->type, &info);
        switch(info.BaseType)
        {
            case MZ_BASE_TYPE_ARRAY:
                c->SetOutputs(value->Data ? flatbuffers::GetRoot<flatbuffers::Vector<u8>>(value->Data)->size() : 0);
                break;
            case MZ_BASE_TYPE_STRUCT:
                c->SetOutputs(&info);
                break;
            default:
                c->Reset();
        }
    };

    fn->OnPinConnected = [](void* ctx, mzName pinName, mzUUID connector)
    {
        auto c = (Disarray*)ctx;

        if (MZN_Input != pinName)
            return;
        
        mzTypeInfo info = {};
        mzEngine.GetPinType(connector, &info);
        if (info.BaseType != MZ_BASE_TYPE_ARRAY && info.BaseType != MZ_BASE_TYPE_STRUCT)
        {
            c->Reset();
            return;
        }

        if (MZN_VOID == c->type)
        {
            c->type = info.TypeName;
            flatbuffers::FlatBufferBuilder fbb;
            std::vector<::flatbuffers::Offset<mz::PartialPinUpdate>> updates = {
                CreatePartialPinUpdateDirect(fbb, &c->input, 0, mz::Action::NOP, mz::Action::NOP, mz::Action::NOP, mzEngine.GetString(info.TypeName))
            };
            mzEngine.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &c->id, ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, &updates)));
        }
    };
}

} // namespace mz