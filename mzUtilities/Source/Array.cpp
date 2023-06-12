#include <MediaZ/Helpers.hpp>

namespace mz::utilities
{

MZ_REGISTER_NAME_SPACED(Array, "mz.utilities.Array")

struct ArrayCtx
{
    fb::UUID id;
    std::vector<fb::UUID> inputs;
    std::optional<fb::UUID> output;
    std::optional<mzName> type;

    ArrayCtx(const mzFbNode* inNode)
    {
        id = *inNode->id();
        static mzName VOID = mzEngine.GetName("mz.fb.Void");
        for (auto pin : *inNode->pins())
        {
            if (pin->show_as() == fb::ShowAs::INPUT_PIN ||
                pin->show_as() == fb::ShowAs::PROPERTY)
            {
                auto ty = mzEngine.GetName(pin->type_name()->c_str());
                if (!type)
                {
                    if (ty != VOID)
                        type = ty;
                }
                else
                    assert(*type == ty);
                inputs.push_back(*pin->id());
            }
            else
            {
                assert(!output);
                output = *pin->id();
            }
        }
    }

};

void RegisterArray(mzNodeFunctions* fn)
{
	fn->TypeName = MZN_Array;
    
	fn->OnNodeCreated = [](const mzFbNode* node, void** outCtxPtr) 
    {
        *outCtxPtr = new ArrayCtx(node);
	};

    fn->OnNodeUpdated = [](void* ctx, const mzFbNode* node) 
    {
        *((ArrayCtx*)ctx) = ArrayCtx(node);
	};

    fn->OnPinConnected = [](void* ctx, mzName pinName, mzUUID connector)
    {
        auto c = (ArrayCtx*)ctx;
        if (c->type)
            return;

        mzTypeInfo info = {};
        mzEngine.GetPinType(connector, &info);
        auto typeName = mzEngine.GetString(info.TypeName);
        auto outputType = "[" + std::string(typeName) + "]";

        flatbuffers::FlatBufferBuilder fbb;

        mzUUID id0, id1;
        mzEngine.GenerateID(&id0);
        mzEngine.GenerateID(&id1);
        std::vector<::flatbuffers::Offset<mz::fb::Pin>> pins = {
            fb::CreatePinDirect(fbb, &id0, "Input 0", typeName, fb::ShowAs::INPUT_PIN, fb::CanShowAs::INPUT_PIN_OR_PROPERTY),
            fb::CreatePinDirect(fbb, &id1, "Output",  outputType.c_str(), fb::ShowAs::OUTPUT_PIN, fb::CanShowAs::OUTPUT_PIN_ONLY),
        };
        mzEngine.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &c->id, ClearFlags::ANY, 0, &pins)));
    };
    
	fn->OnNodeDeleted = [](void* ctx, mzUUID nodeId) 
    {
        delete (ArrayCtx*)ctx;
	};

	fn->ExecuteNode = [](void* ctx, const mzNodeExecuteArgs* args) -> mzResult 
    {
        return MZ_RESULT_SUCCESS;
	};

    fn->OnMenuRequested = [](void* ctx, const mzContextMenuRequest* request)
    {
        auto c = (ArrayCtx*)ctx;
        if(!c->type) return;
        flatbuffers::FlatBufferBuilder fbb;
        std::vector<flatbuffers::Offset<mz::ContextMenuItem>> fields;
        std::string add = "Add Input " + std::to_string(c->inputs.size());
        fields.push_back(mz::CreateContextMenuItemDirect(fbb, add.c_str(), 1));
        if (!c->inputs.empty())
        {
            std::string remove = "Remove Input " + std::to_string(c->inputs.size() - 1);
            fields.push_back(mz::CreateContextMenuItemDirect(fbb, remove.c_str(), 2));
        }
        mzEngine.HandleEvent(CreateAppEvent(fbb, mz::CreateContextMenuUpdateDirect(fbb, &c->id, request->pos(), request->instigator(), &fields)));
    };

    fn->OnMenuCommand = [](void* ctx, uint32_t cmd)
    {
        auto c = (ArrayCtx*)ctx;
        if(!c->type)
            return;

        const u32 action = cmd & 3;
   
        flatbuffers::FlatBufferBuilder fbb;
        switch (action)
        {
        case 2: // Remove Field
        {
            std::vector<fb::UUID> id = { c->inputs.back() };
            c->inputs.pop_back();
            mzEngine.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &c->id, ClearFlags::NONE, &id)));
        }
        break;
        case 1: // Add Field
        {
            auto typeName = mzEngine.GetString(*c->type);
            auto outputType = "[" + std::string(typeName) + "]";

            mzBuffer value;
            std::vector<u8> data;
            if (MZ_RESULT_SUCCESS == mzEngine.GetDefaultValueOfType(*c->type, &value))
            {
                data = std::vector<u8>{ (u8*)value.Data, (u8*)value.Data + value.Size};
            }

            auto slot = c->inputs.size();
            mzUUID id;
            mzEngine.GenerateID(&id);
            c->inputs.push_back(id);
            std::vector<flatbuffers::Offset<mz::fb::Pin>> pins = {
                    mz::fb::CreatePinDirect(fbb, &id, ("Input " + std::to_string(slot)).c_str(), typeName, mz::fb::ShowAs::INPUT_PIN, mz::fb::CanShowAs::INPUT_PIN_OR_PROPERTY, 0, 0, &data),
            };
            mzEngine.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &c->id, ClearFlags::NONE, 0, &pins)));
        }
        break;
        }
    };
}

} // namespace mz