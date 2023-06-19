#include "TypeCommon.h"

namespace mz::utilities
{

MZ_REGISTER_NAME_SPACED(Indexer, "mz.utilities.Indexer")
MZ_REGISTER_NAME(Input);
MZ_REGISTER_NAME(Output);
MZ_REGISTER_NAME(Index);
MZ_REGISTER_NAME_SPACED(VOID, "mz.fb.Void");

struct Indexer
{
    fb::UUID id;
    fb::UUID output;
    fb::UUID input;
    mzName type = MZN_VOID;
    
    u32 Index = 0;
    u32 ArraySize = 0;

    Indexer(const mzFbNode* inNode)
    {
        id = *inNode->id();
        u32 mbyArraySize = 0;

        for (auto pin : *inNode->pins())
        {
            if (pin->show_as() == fb::ShowAs::OUTPUT_PIN)
            {
                if(flatbuffers::IsFieldPresent(pin, fb::Pin::VT_DATA))
                    mbyArraySize = flatbuffers::GetRoot<flatbuffers::Vector<u8>>(pin->data()->Data())->size();
                output = *pin->id();
            }
            else if(pin->name()->str() != "Index")
            {
                type = mzEngine.GetName(pin->type_name()->c_str());
                input = *pin->id();
            }
            else
            {
                if(flatbuffers::IsFieldPresent(pin, fb::Pin::VT_DATA))
                    Index = *(u32*)pin->data()->Data();
            }
        }
        if(type != MZN_VOID)
            ArraySize = mbyArraySize;
    }

    void SetType(mzName type)
    {
        this->type = type;
        std::string arrayTypeName = mzEngine.GetString(type);
        std::string typeName = std::string(arrayTypeName.begin() + 1, arrayTypeName.end() - 1);

        flatbuffers::FlatBufferBuilder fbb;
        std::vector<::flatbuffers::Offset<mz::PartialPinUpdate>> updates = {
            CreatePartialPinUpdateDirect(fbb, &input,  0, mz::Action::NOP, mz::Action::NOP, mz::Action::NOP, arrayTypeName.c_str()),
            CreatePartialPinUpdateDirect(fbb, &output, 0, mz::Action::NOP, mz::Action::NOP, mz::Action::NOP, typeName.c_str())
        };
        mzEngine.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &id, ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, &updates)));
    }

    void Reset()
    {
        type = MZN_VOID;
        flatbuffers::FlatBufferBuilder fbb;
        std::vector<::flatbuffers::Offset<mz::fb::Pin>> pins = {
            fb::CreatePinDirect(fbb, &input,  "Input",  "mz.fb.Void", fb::ShowAs::INPUT_PIN, fb::CanShowAs::INPUT_PIN_OR_PROPERTY),
            fb::CreatePinDirect(fbb, &output, "Output", "mz.fb.Void", fb::ShowAs::OUTPUT_PIN, fb::CanShowAs::OUTPUT_PIN_ONLY),
        };
        mzEngine.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &id, ClearFlags::NONE, 0, &pins)));
    }

    void Orphan(bool c)
    {
        flatbuffers::FlatBufferBuilder fbb;
        std::vector<::flatbuffers::Offset<mz::PartialPinUpdate>> updates = {
            CreatePartialPinUpdateDirect(fbb, &output, 0, c ? Action::SET : Action::RESET, mz::Action::NOP, mz::Action::NOP)
        };
        mzEngine.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &id, ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, &updates)));
    }
};

void RegisterIndexer(mzNodeFunctions* fn)
{
	fn->TypeName = MZN_Indexer;

	fn->OnNodeCreated = [](const mzFbNode* node, void** outCtxPtr) 
    {
        *outCtxPtr = new Indexer(node);
	};

	fn->OnNodeDeleted = [](void* ctx, mzUUID nodeId) 
    {
        delete (Indexer*)ctx;
	};

    fn->OnNodeUpdated = [](void* ctx, const mzFbNode* node)
    {
        // *((Indexer*)ctx) = Indexer(node);
    };

	fn->ExecuteNode = [](void* ctx, const mzNodeExecuteArgs* args) -> mzResult 
    {
        auto c = (Indexer*)ctx;
        if(MZN_VOID == c->type)
            return MZ_RESULT_SUCCESS;

        auto pins = NodeExecuteArgs(args);
        auto vec = flatbuffers::GetRoot<flatbuffers::Vector<u8>>(pins[MZN_Input].Buf.Data);
        c->Index = *(u32*)pins[MZN_Index].Buf.Data;
        c->ArraySize = vec->size();
        if(c->Index < c->ArraySize)
        {
            mzTypeInfo info = {};
            mzEngine.GetTypeInfo(c->type, &info);
            auto ID = pins[MZN_Output].ID;
            if (info.ElementType->ByteSize)
            {
                auto data = vec->data() + c->Index * info.ElementType->ByteSize;
                mzEngine.SetPinValue(ID, { (void*)data, info.ElementType->ByteSize });
            }
            else
            {
                flatbuffers::FlatBufferBuilder fbb;
                auto vect = flatbuffers::GetRoot<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::Table>>>(pins[MZN_Input].Buf.Data);
                fbb.Finish(flatbuffers::Offset<flatbuffers::Table>(CopyTable(fbb, info.ElementType, vect->Get(c->Index))));
                mz::Buffer buf = fbb.Release();
                mzEngine.SetPinValue(ID, { (void*)buf.data(), buf.size() });
            }
        }
        return MZ_RESULT_SUCCESS;
	};

    fn->OnPinValueChanged = [](void* ctx, mzName pinName, mzBuffer* value)
    {
        auto c = (Indexer*)ctx;
        
        if(pinName == MZN_Index)
        {
            c->Index = *(u32*)value->Data;
            c->Orphan(c->Index >= c->ArraySize);
            return;
        }

        if(MZN_VOID != c->type)
        {
            if (pinName == MZN_Input)
            {
                c->ArraySize = flatbuffers::GetRoot<flatbuffers::Vector<u8>>(value->Data)->size();
                c->Orphan(c->Index >= c->ArraySize);
            }
            return;
        }

        if(pinName != MZN_Input)
            return;
        
        mzTypeInfo info = {};
        mzEngine.GetTypeInfo(c->type, &info);
        if (info.BaseType != MZ_BASE_TYPE_ARRAY)
        {
            c->Reset();
            return;
        }
    };

    fn->OnPinConnected = [](void* ctx, mzName pinName, mzUUID connector)
    {
        auto c = (Indexer*)ctx;

        if (MZN_Input != pinName)
            return;
        
        mzTypeInfo info = {};
        mzEngine.GetPinType(connector, &info);
        if (info.BaseType != MZ_BASE_TYPE_ARRAY)
        {
            c->Reset();
            return;
        }

        if (MZN_VOID == c->type)
        {
            c->SetType(info.TypeName);
        }

    };
}

} // namespace mz