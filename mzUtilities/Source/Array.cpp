#include "TypeCommon.h"

namespace mz::utilities
{

MZ_REGISTER_NAME_SPACED(Array, "mz.utilities.Array")
MZ_REGISTER_NAME(Output);

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

        mzBuffer value;
        std::vector<u8> data;
        std::vector<u8> outData;

        if (MZ_RESULT_SUCCESS == mzEngine.GetDefaultValueOfType(info.TypeName, &value))
        {
            data = std::vector<u8>{ (u8*)value.Data, (u8*)value.Data + value.Size };
        }

        if (MZ_RESULT_SUCCESS == mzEngine.GetDefaultValueOfType(mzEngine.GetName(outputType.c_str()), &value))
        {
            outData = std::vector<u8>{ (u8*)value.Data, (u8*)value.Data + value.Size };
        }

        flatbuffers::FlatBufferBuilder fbb;

        mzUUID id;
        mzEngine.GenerateID(&id);

        std::vector<::flatbuffers::Offset<mz::PartialPinUpdate>> updates = {
            CreatePartialPinUpdateDirect(fbb, &c->inputs.front(), 0, mz::Action::NOP, mz::Action::NOP, mz::Action::NOP, typeName)
        };

        std::vector<::flatbuffers::Offset<mz::fb::Pin>> pins = {
            fb::CreatePinDirect(fbb, &id, "Output",  outputType.c_str(), fb::ShowAs::OUTPUT_PIN, fb::CanShowAs::OUTPUT_PIN_ONLY, 0, 0, &outData),
        };

        mzEngine.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &c->id, ClearFlags::NONE, 0, &pins, 0, 0, 0, 0, 0, &updates)));
    };
    
	fn->OnNodeDeleted = [](void* ctx, mzUUID nodeId) 
    {
        delete (ArrayCtx*)ctx;
	};

	fn->ExecuteNode = [](void* ctx, const mzNodeExecuteArgs* args) -> mzResult 
    {
        auto c = (ArrayCtx*)ctx;
        if(!c->type) return MZ_RESULT_SUCCESS;
        mzTypeInfo info = {};
        mzEngine.GetTypeInfo(*c->type, &info);
        //auto pins = NodeExecuteArgs(args);
        //pins.erase(MZN_Output);
        flatbuffers::FlatBufferBuilder fbb;

        flatbuffers::uoffset_t offset = 0;
        if (info.ByteSize)
        {
            fbb.StartVector(args->PinCount-1, info.ByteSize, 1);
            for (int i = 0; i < args->PinCount; ++i)
            {
                if(args->PinNames[i] == MZN_Output) continue;
                if (!args->PinValues[i].Size)
                {
                    std::vector<u8> zero(info.ByteSize);
                    fbb.PushBytes(zero.data(), info.ByteSize);
                }
                else
                {
                    assert(info.ByteSize == args->PinValues[i].Size);
                    fbb.PushBytes((u8*)args->PinValues[i].Data, info.ByteSize);
                }
            }
            offset = fbb.EndVector(args->PinCount - 1);
        }
        else
        {
			switch (info.BaseType) 
            {
			case MZ_BASE_TYPE_STRING: {
				std::vector<flatbuffers::Offset<flatbuffers::String>> elements;
                for (int i = 0; i < args->PinCount; ++i) 
                {
                    if(args->PinNames[i] == MZN_Output) continue;
					elements.push_back(fbb.CreateString((const char*)args->PinValues[i].Data));
				}
                offset = fbb.CreateVector(elements).o;
				break;
			}
			case MZ_BASE_TYPE_STRUCT: {
                std::vector<flatbuffers::Offset<const flatbuffers::Table*>> elements;
                for (int i = 0; i < args->PinCount; ++i)
                {
                    if (args->PinNames[i] == MZN_Output) continue;
                    elements.push_back(CopyTable(fbb, &info, flatbuffers::GetRoot<flatbuffers::Table>(args->PinValues[i].Data)));
                }
                offset = fbb.CreateVector(elements).o;
                break;
			}
			default: {                    // Scalars and structs.
				assert(0);
			}
			}
        }
        
        fbb.Finish(flatbuffers::Offset<flatbuffers::Vector<uint8_t>>(offset));
        mz::Buffer buf = fbb.Release();
        mzEngine.SetPinValue(*c->output, {(void*)buf.data(),buf.size()});
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