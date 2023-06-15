#include <MediaZ/Helpers.hpp>

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
    std::optional<mzName> type;

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

    void SetOutputs(u32 sz)
    {
        const u32 curr = outputs.size();
        if(curr == sz) return;

        std::string typeName = mzEngine.GetString(*type) + 1;
        typeName.pop_back();
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
        type = {};
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
        auto pins = NodeExecuteArgs(args);
        auto vec = flatbuffers::GetRoot<flatbuffers::Vector<u8>>(pins[MZN_Input].Buf.Data);
        c->SetOutputs(vec->size());
        return MZ_RESULT_SUCCESS;
	};


    fn->OnPinValueChanged = [](void* ctx, mzName pinName, mzBuffer* value)
    {

        auto c = (Disarray*)ctx;
        if(pinName != MZN_Input || !c->type)
            return;

        mzTypeInfo info = {};
        mzEngine.GetTypeInfo(*c->type, &info);
        if (info.BaseType != MZ_BASE_TYPE_ARRAY)
        {
            c->Reset();
            return;
        }
        auto vec = flatbuffers::GetRoot<flatbuffers::Vector<u8>>(value->Data);
        c->SetOutputs(vec->size());
    };

    fn->OnPinConnected = [](void* ctx, mzName pinName, mzUUID connector)
    {
        auto c = (Disarray*)ctx;

        if (MZN_Input != pinName)
            return;
        
        mzTypeInfo info = {};
        mzEngine.GetPinType(connector, &info);
        if (info.BaseType != MZ_BASE_TYPE_ARRAY)
        {
            c->Reset();
            return;
        }
        if (!c->type)
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