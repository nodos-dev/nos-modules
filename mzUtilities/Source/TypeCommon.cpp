#include "TypeCommon.h"

namespace mz::utilities
{

void CopyInline(flatbuffers::FlatBufferBuilder& fbb, decltype(mzTypeInfo::Fields) fielddef,
    const flatbuffers::Table* table, size_t align, size_t size) {
    fbb.Align(align);
    fbb.PushBytes(table->GetStruct<const uint8_t*>(fielddef->Offset), size);
    fbb.TrackField(fielddef->Offset, fbb.GetSize());
}

flatbuffers::uoffset_t CopyTable(
	flatbuffers::FlatBufferBuilder& fbb,
	const mzTypeInfo* type,
	const flatbuffers::Table* table)
{
	// Before we can construct the table, we have to first generate any
	// subobjects, and collect their offsets.
	std::vector<flatbuffers::uoffset_t> offsets;

    for(int i = 0; i < type->FieldCount; ++i)
	{
        auto field = &type->Fields[i];
		// Skip if field is not present in the source.
		if (!table->CheckField(field->Offset)) continue;
		flatbuffers::uoffset_t offset = 0;
		switch (field->Type->BaseType) {
		case MZ_BASE_TYPE_STRING: {
			offset = fbb.CreateString(table->GetPointer<const flatbuffers::String*>(field->Offset)).o;
			break;
		}
		case MZ_BASE_TYPE_STRUCT: {
			if (!field->Type->ByteSize) {
				offset = CopyTable(fbb, field->Type, table->GetPointer<flatbuffers::Table*>(field->Offset));
			}
			break;
		}
		//case MZ_BASE_TYPE_UNION: {
		//	offset = CopyTable2(fbb, GetUnionType(objectdef, field, table), table->GetPointer<flatbuffers::Table*>(field->Offset));
		//	break;
		//}
		case MZ_BASE_TYPE_ARRAY: {
			auto vec =
				table->GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::Table>> *>(field->Offset);
			auto element_base_type = field->Type->ElementType->BaseType;
			// auto elemobjectdef = element_base_type == MZ_BASE_TYPE_STRUCT ? field->Type->struct_def : 0;
			
			switch (element_base_type) {
			case MZ_BASE_TYPE_STRING: {
				std::vector<flatbuffers::Offset<const flatbuffers::String*>> elements(vec->size());
				auto vec_s = reinterpret_cast<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(vec);
				for (flatbuffers::uoffset_t i = 0; i < vec_s->size(); i++) {
					elements[i] = fbb.CreateString(vec_s->Get(i)).o;
				}
				offset = fbb.CreateVector(elements).o;
				break;
			}
			case MZ_BASE_TYPE_STRUCT: {
				if (!field->Type->ElementType->ByteSize) {
					std::vector<flatbuffers::Offset<const flatbuffers::Table*>> elements(vec->size());
					for (flatbuffers::uoffset_t i = 0; i < vec->size(); i++) {
						elements[i] = CopyTable(fbb, field->Type->ElementType, vec->Get(i));
					}
					offset = fbb.CreateVector(elements).o;
					break;
				}
			}
								FLATBUFFERS_FALLTHROUGH();  // fall thru
			default: {                    // Scalars and structs.
				fbb.StartVector(vec->size(), field->Type->ByteSize, field->Type->Alignment);
				fbb.PushBytes(vec->Data(), field->Type->ByteSize * vec->size());
				offset = fbb.EndVector(vec->size());
				break;
			}
			}
			break;
		}
		default:  // Scalars.
			break;
		}
		if (offset) { offsets.push_back(offset); }
	}
	// Now we can build the actual table from either offsets or scalar data.
	auto start = type->ByteSize ? fbb.StartStruct(type->Alignment)
		: fbb.StartTable();
	size_t offset_idx = 0;

    for (int i = 0; i < type->FieldCount; ++i)
    {
        auto field = &type->Fields[i];
		if (!table->CheckField(field->Offset)) continue;
		auto base_type = field->Type->BaseType;
		switch (base_type) {
		case MZ_BASE_TYPE_STRUCT: {
			if (field->Type->ByteSize) {
				CopyInline(fbb, field, table, field->Type->Alignment, field->Type->ByteSize);
				break;
			}
		}
		// case MZ_BASE_TYPE_UNION:
		case MZ_BASE_TYPE_STRING:
		case MZ_BASE_TYPE_ARRAY:
			fbb.AddOffset(field->Offset, flatbuffers::Offset<void>(offsets[offset_idx++]));
			break;
		default: {  // Scalars.
            CopyInline(fbb, field, table, field->Type->Alignment, field->Type->ByteSize);
			break;
		}
		}
	}
	FLATBUFFERS_ASSERT(offset_idx == offsets.size());
	if (type->ByteSize) {
		fbb.ClearOffsets();
		return fbb.EndStruct();
	}
	else {
		return fbb.EndTable(start);
	}
}

flatbuffers::uoffset_t GenerateOffset(
    flatbuffers::FlatBufferBuilder& fbb,
    const mzTypeInfo* type,
    const void* data)
{
    if(type->ByteSize) 
        return 0;
    switch (type->BaseType)
    {
    case MZ_BASE_TYPE_STRUCT:
        return CopyTable(fbb, type, flatbuffers::GetRoot<flatbuffers::Table>(data));
    case MZ_BASE_TYPE_STRING:
        return fbb.CreateString((const flatbuffers::String*)data).o;
    case MZ_BASE_TYPE_ARRAY: {
        auto vec = (flatbuffers::Vector<void*>*)(data);
        if(type->ElementType->ByteSize)
        {
            fbb.StartVector(vec->size(), type->ElementType->ByteSize, 1);
            fbb.PushBytes(vec->Data(), type->ElementType->ByteSize * vec->size());
            return fbb.EndVector(vec->size());
        }
        std::vector<flatbuffers::uoffset_t> elements(vec->size());
        for (int i = 0; i < vec->size(); i++) {
            elements[i] = GenerateOffset(fbb, type->ElementType, vec->Get(i));
        }
        return fbb.CreateVector(elements).o;
    }
    }
    return 0;
}
}