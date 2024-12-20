// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include "TypeCommon.h"
namespace flatbuffers
{

EnumVal* EnumDef::ReverseLookup(int64_t enum_idx,
	bool skip_union_default) const {
	auto skip_first = static_cast<int>(is_union && skip_union_default);
	for (auto it = Vals().begin() + skip_first; it != Vals().end(); ++it) {
		if ((*it)->GetAsInt64() == enum_idx) { return *it; }
	}
	return nullptr;
}
}
namespace nos::reflect
{
void CopyInline(
	flatbuffers::FlatBufferBuilder& fbb, uint16_t offset, const uint8_t* data, size_t align, size_t size)
{
	fbb.Align(align);
	fbb.PushBytes(data, size);
	fbb.TrackField(offset, fbb.GetSize());
}

void CopyInline2(flatbuffers::FlatBufferBuilder& fbb, const flatbuffers::FieldDef* fielddef,
	const flatbuffers::Table* table, size_t align, size_t size) {
	fbb.Align(align);
	fbb.PushBytes(table->GetStruct<const uint8_t*>(fielddef->value.offset), size);
	fbb.TrackField(fielddef->value.offset, fbb.GetSize());
}

const flatbuffers::StructDef* GetUnionType(
	const flatbuffers::StructDef* parent,
	const flatbuffers::FieldDef* unionfield, 
	const flatbuffers::Table* table) {
	auto enumdef = unionfield->value.type.enum_def;
	// TODO: this is clumsy and slow, but no other way to find it?
	auto type_field = parent->fields.Lookup(unionfield->name + "_type");
	
	FLATBUFFERS_ASSERT(type_field);
	auto union_type = table->GetField<uint8_t>(type_field->value.offset, 0);
	auto enumval = enumdef->ReverseLookup(union_type);
	return enumval->union_type.struct_def;
}

flatbuffers::uoffset_t CopyTable2(
	flatbuffers::FlatBufferBuilder& fbb,
	const flatbuffers::StructDef* objectdef,
	const flatbuffers::Table* table)
{
	// Before we can construct the table, we have to first generate any
	// subobjects, and collect their offsets.
	std::vector<flatbuffers::uoffset_t> offsets;

	for (auto field : objectdef->fields.vec)
	{
		// Skip if field is not present in the source.
		if (!table->CheckField(field->value.offset)) continue;
		flatbuffers::uoffset_t offset = 0;
		switch (field->value.type.base_type) {
		case flatbuffers::BASE_TYPE_STRING: {
			offset = fbb.CreateString(table->GetPointer<const flatbuffers::String*>(field->value.offset)).o;
			break;
		}
		case flatbuffers::BASE_TYPE_STRUCT: {
			if (!field->value.type.struct_def->fixed) {
				offset = CopyTable2(fbb, field->value.type.struct_def, table->GetPointer<flatbuffers::Table*>(field->value.offset));
			}
			break;
		}
		case flatbuffers::BASE_TYPE_UNION: {
			offset = CopyTable2(fbb, GetUnionType(objectdef, field, table), table->GetPointer<flatbuffers::Table*>(field->value.offset));
			break;
		}
		case flatbuffers::BASE_TYPE_VECTOR: {
			auto vec =
				table->GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::Table>> *>(field->value.offset);
			auto element_base_type = field->value.type.element;
			auto elemobjectdef = element_base_type == flatbuffers::BASE_TYPE_STRUCT ? field->value.type.struct_def : 0;
			
			switch (element_base_type) {
			case flatbuffers::BASE_TYPE_STRING: {
				std::vector<flatbuffers::Offset<const flatbuffers::String*>> elements(vec->size());
				auto vec_s = reinterpret_cast<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(vec);
				for (flatbuffers::uoffset_t i = 0; i < vec_s->size(); i++) {
					elements[i] = fbb.CreateString(vec_s->Get(i)).o;
				}
				offset = fbb.CreateVector(elements).o;
				break;
			}
			case flatbuffers::BASE_TYPE_STRUCT: {
				if (!elemobjectdef->fixed) {
					std::vector<flatbuffers::Offset<const flatbuffers::Table*>> elements(vec->size());
					for (flatbuffers::uoffset_t i = 0; i < vec->size(); i++) {
						elements[i] = CopyTable2(fbb, elemobjectdef, vec->Get(i));
					}
					offset = fbb.CreateVector(elements).o;
					break;
				}
			}
								FLATBUFFERS_FALLTHROUGH();  // fall thru
			default: {                    // Scalars and structs.
				auto element_size = SizeOf(element_base_type);
				auto element_alignment = element_size; // For primitive elements
				if (elemobjectdef && elemobjectdef->fixed)
					element_size = elemobjectdef->bytesize;
				fbb.StartVector(vec->size(), element_size, element_alignment);
				fbb.PushBytes(vec->Data(), element_size * vec->size());
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
	auto start = objectdef->fixed ? fbb.StartStruct(objectdef->minalign)
		: fbb.StartTable();
	size_t offset_idx = 0;

	for (auto field : objectdef->fields.vec)
	{
		if (!table->CheckField(field->value.offset)) continue;
		auto base_type = field->value.type.base_type;
		switch (base_type) {
		case flatbuffers::BASE_TYPE_STRUCT: {
			if (field->value.type.struct_def->fixed) {
				CopyInline2(fbb, field, table, field->value.type.struct_def->minalign, field->value.type.struct_def->bytesize);
				break;
			}
		}
							FLATBUFFERS_FALLTHROUGH();  // fall thru
		case flatbuffers::BASE_TYPE_UNION:
		case flatbuffers::BASE_TYPE_STRING:
		case flatbuffers::BASE_TYPE_VECTOR:
			fbb.AddOffset(field->value.offset, flatbuffers::Offset<void>(offsets[offset_idx++]));
			break;
		default: {  // Scalars.
			auto size = SizeOf(base_type);
			CopyInline2(fbb, field, table, size, size);
			break;
		}
		}
	}
	FLATBUFFERS_ASSERT(offset_idx == offsets.size());
	if (objectdef->fixed) {
		fbb.ClearOffsets();
		return fbb.EndStruct();
	}
	else {
		return fbb.EndTable(start);
	}
}

void CopyInline(flatbuffers::FlatBufferBuilder& fbb, decltype(nosTypeInfo::Fields) fielddef,
	const flatbuffers::Table* table, size_t align, size_t size) {
	fbb.Align(align);
	fbb.PushBytes(table->GetStruct<const uint8_t*>(fielddef->Offset), size);
	fbb.TrackField(fielddef->Offset, fbb.GetSize());
}

flatbuffers::uoffset_t CopyTable(
	flatbuffers::FlatBufferBuilder& fbb,
	const nosTypeInfo* type,
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
		case NOS_BASE_TYPE_STRING: {
			offset = fbb.CreateString(table->GetPointer<const flatbuffers::String*>(field->Offset)).o;
			break;
		}
		case NOS_BASE_TYPE_STRUCT: {
			if (!field->Type->ByteSize) {
				offset = CopyTable(fbb, field->Type, table->GetPointer<flatbuffers::Table*>(field->Offset));
			}
			break;
		}
		//case NOS_BASE_TYPE_UNION: {
		//	offset = CopyTable2(fbb, GetUnionType(objectdef, field, table), table->GetPointer<flatbuffers::Table*>(field->Offset));
		//	break;
		//}
		case NOS_BASE_TYPE_ARRAY: {
			auto vec =
				table->GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::Table>> *>(field->Offset);
			auto element_base_type = field->Type->ElementType->BaseType;
			// auto elemobjectdef = element_base_type == NOS_BASE_TYPE_STRUCT ? field->Type->struct_def : 0;
			
			switch (element_base_type) {
			case NOS_BASE_TYPE_STRING: {
				std::vector<flatbuffers::Offset<const flatbuffers::String*>> elements(vec->size());
				auto vec_s = reinterpret_cast<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(vec);
				for (flatbuffers::uoffset_t i = 0; i < vec_s->size(); i++) {
					elements[i] = fbb.CreateString(vec_s->Get(i)).o;
				}
				offset = fbb.CreateVector(elements).o;
				break;
			}
			case NOS_BASE_TYPE_STRUCT: {
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
		case NOS_BASE_TYPE_STRUCT: {
			if (field->Type->ByteSize) {
				CopyInline(fbb, field, table, field->Type->Alignment, field->Type->ByteSize);
				break;
			}
		}
		// case NOS_BASE_TYPE_UNION:
		case NOS_BASE_TYPE_STRING:
		case NOS_BASE_TYPE_ARRAY:
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

bool AreFlatBuffersEqual(const nosTypeInfo* type,
		                 void* first,
		                 void* second)
{
	switch (type->BaseType)
	{
	case NOS_BASE_TYPE_STRUCT:
		{
			if (type->ByteSize)
			{
				return memcmp(first, second, type->ByteSize) == 0;
			}
			// This is a table.
			auto firstTable = static_cast<flatbuffers::Table*>(first);
			auto secondTable = static_cast<flatbuffers::Table*>(second);
			for (int i = 0; i < type->FieldCount; ++i)
			{
				auto field = &type->Fields[i];
				auto field1Exists = firstTable->CheckField(field->Offset);
				auto field2Exists = secondTable->CheckField(field->Offset);
				if ((field1Exists ^ field2Exists) && (field->Type->BaseType == NOS_BASE_TYPE_STRING || field->Type->BaseType == NOS_BASE_TYPE_ARRAY))
				{
					// If one is present but the other is not, check the size is 0 for the present one.
					// TODO: This might not be desired behaviour in some cases.
					auto presentVec = field1Exists ? firstTable->GetPointer<flatbuffers::Vector<uint8_t>*>(field->Offset) 
						: (field2Exists ? secondTable->GetPointer<flatbuffers::Vector<uint8_t>*>(field->Offset) : nullptr);
					if (!presentVec)
						continue;
					if (presentVec->size() != 0)
						return false;
					continue;
				}
				if (field1Exists != field2Exists)
					return false;
				// Skip if fields are not present in the source.
				if (!field1Exists)
					continue;
				void* field1, *field2;
				if (field->Type->ByteSize)
				{
					field1 = firstTable->GetStruct<void*>(field->Offset);
					field2 = secondTable->GetStruct<void*>(field->Offset);
				}
				else
				{
					if (field->Type->BaseType == NOS_BASE_TYPE_STRING)
					{
						auto str1 = firstTable->GetPointer<const flatbuffers::String*>(field->Offset);
						auto str2 = secondTable->GetPointer<const flatbuffers::String*>(field->Offset);
						field1 = (void*)str1->c_str();
						field2 = (void*)str2->c_str();
					}
					else
					{
						field1 = firstTable->GetPointer<flatbuffers::Table*>(field->Offset);
						field2 = secondTable->GetPointer<flatbuffers::Table*>(field->Offset);	
					}
					
				}
				if (!AreFlatBuffersEqual(field->Type, field1, field2))
					return false;
			}
			return true;
		}
	case NOS_BASE_TYPE_STRING:
		{
			auto firstString = static_cast<const char*>(first);
			auto secondString = static_cast<const char*>(second);
			return strcmp(firstString, secondString) == 0;
		}
	case NOS_BASE_TYPE_ARRAY:
		{
			auto vec1 = static_cast<const flatbuffers::Vector<uint8_t>*>(first);
			auto vec2 = static_cast<const flatbuffers::Vector<uint8_t>*>(second);
			if (vec1->size() != vec2->size())
				return false;
			auto elementType = type->ElementType;
			if (elementType->ByteSize)
			{
				for (size_t i = 0; i < vec1->size(); ++i)
				{
					if (memcmp(vec1->data() + i * type->ByteSize, vec2->data() + i * type->ByteSize, type->ByteSize) != 0)
						return false;
				}
			}
			else
			{
				if (elementType->BaseType == NOS_BASE_TYPE_STRING)
				{
					auto vec1Vectors = reinterpret_cast<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>*>(vec1);
					auto vec2Vectors = reinterpret_cast<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>*>(vec2);
					for (size_t i = 0; i < vec1Vectors->size(); ++i)
					{
						auto elem1 = vec1Vectors->Get(i);
						auto elem2 = vec2Vectors->Get(i);
						if (!AreFlatBuffersEqual(elementType, (void*)elem1->c_str(), (void*)elem2->c_str()))
							return false;
					}
				}
				else
				{
					auto vec1Tables = reinterpret_cast<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::Table>>*>(vec1);
					auto vec2Tables = reinterpret_cast<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::Table>>*>(vec2);
					for (size_t i = 0; i < vec1Tables->size(); ++i)
					{
						auto elem1 = vec1Tables->Get(i);
						auto elem2 = vec2Tables->Get(i);
						if (!AreFlatBuffersEqual(elementType, (void*)elem1, (void*)elem2))
							return false;
					}
				}
			}
			return true;
		}
	case NOS_BASE_TYPE_UNION:
		{
			nosEngine.LogE("Union comparison not implemented yet.");
			break;
		}
	default: // Scalars
		{
			if (type->ByteSize)
			{
				return memcmp(first, second, type->ByteSize) == 0;
			}
			break;
		}
	}
	return true;
}

flatbuffers::uoffset_t CopyArgs(
	flatbuffers::FlatBufferBuilder& fbb,
	const nosTypeInfo* type,
	NodeExecuteParams& table)
{
	if(NOS_BASE_TYPE_STRUCT != type->BaseType)
		return 0;
	// Before we can construct the table, we have to first generate any
	// subobjects, and collect their offsets.
	std::vector<flatbuffers::uoffset_t> offsets;

	for (int i = 0; i < type->FieldCount; ++i)
	{
		auto field = &type->Fields[type->FieldCount-i-1];
		// Skip if field is not present in the source.
		if (!table.contains(field->Name)) continue;

		auto data = table[field->Name].Data->Data;

		flatbuffers::uoffset_t offset = 0;
		switch (field->Type->BaseType) {
		case NOS_BASE_TYPE_STRING: {
			offset = fbb.CreateString((const char*)data).o;
			break;
		}
		case NOS_BASE_TYPE_STRUCT: {
			if (!field->Type->ByteSize) {
				offset = CopyTable(fbb, field->Type, flatbuffers::GetRoot<flatbuffers::Table>(data));
			}
			break;
		}
								//case NOS_BASE_TYPE_UNION: {
								//	offset = CopyTable2(fbb, GetUnionType(objectdef, field, table), table->GetPointer<flatbuffers::Table*>(field->Offset));
								//	break;
								//}
		case NOS_BASE_TYPE_ARRAY: {
			auto vec = (flatbuffers::Vector<flatbuffers::Offset<flatbuffers::Table>>*)(data);
			auto element_base_type = field->Type->ElementType->BaseType;
			// auto elemobjectdef = element_base_type == NOS_BASE_TYPE_STRUCT ? field->Type->struct_def : 0;

			switch (element_base_type) {
			case NOS_BASE_TYPE_STRING: {
				std::vector<flatbuffers::Offset<const flatbuffers::String*>> elements(vec->size());
				auto vec_s = reinterpret_cast<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(vec);
				for (flatbuffers::uoffset_t i = 0; i < vec_s->size(); i++) {
					elements[i] = fbb.CreateString(vec_s->Get(i)).o;
				}
				offset = fbb.CreateVector(elements).o;
				break;
			}
			case NOS_BASE_TYPE_STRUCT: {
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
		auto field = &type->Fields[type->FieldCount-i-1];
		if (!table.contains(field->Name)) continue;

		auto data = table[field->Name].Data->Data;
		auto base_type = field->Type->BaseType;
		switch (base_type) {
		case NOS_BASE_TYPE_STRUCT: {
			if (field->Type->ByteSize) {
				fbb.Align(field->Type->Alignment);
				fbb.PushBytes((uint8_t*)data, field->Type->ByteSize);
				fbb.TrackField(field->Offset, fbb.GetSize());
				break;
			}
		}
								// case NOS_BASE_TYPE_UNION:
		case NOS_BASE_TYPE_STRING:
		case NOS_BASE_TYPE_ARRAY:
			fbb.AddOffset(field->Offset, flatbuffers::Offset<void>(offsets[offset_idx++]));
			break;
		default: {  // Scalars.
			fbb.Align(field->Type->Alignment);
			fbb.PushBytes((uint8_t*)data, field->Type->ByteSize);
			fbb.TrackField(field->Offset, fbb.GetSize());
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
	const nosTypeInfo* type,
	const void* data)
{
	if(type->ByteSize) 
		return 0;
	switch (type->BaseType)
	{
	case NOS_BASE_TYPE_STRUCT:
		return CopyTable(fbb, type, flatbuffers::GetRoot<flatbuffers::Table>(data));
	case NOS_BASE_TYPE_STRING:
		return fbb.CreateString((const flatbuffers::String*)data).o;
	case NOS_BASE_TYPE_ARRAY: {
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

std::vector<uint8_t> GenerateBuffer(
	const nosTypeInfo* type,
	const void* data)
{
	if (type->ByteSize)
	{
		if (data) return std::vector<uint8_t>{(uint8_t*)data, (uint8_t*)data + type->ByteSize};
		return std::vector<uint8_t>(type->ByteSize);
	}
	if(!data) return {};
	flatbuffers::FlatBufferBuilder fbb;
	fbb.Finish(flatbuffers::Offset<uint8_t>(GenerateOffset(fbb, type, data)));
	return nos::Buffer(fbb.Release());
}

flatbuffers::uoffset_t GenerateVector(
	flatbuffers::FlatBufferBuilder& fbb, 
	const nosTypeInfo* type, 
	std::vector<const void*> values)
{
	flatbuffers::uoffset_t offset = 0;
	if (type->ByteSize)
	{
		fbb.StartVector(values.size(), type->ByteSize, 1);
		for (uint32_t i = values.size(); i != 0; --i)
			fbb.PushBytes((uint8_t*)values[i-1], type->ByteSize);
		offset = fbb.EndVector(values.size());
	}
	else
	{
		switch (type->BaseType)
		{
		case NOS_BASE_TYPE_STRING: {
			std::vector<flatbuffers::Offset<flatbuffers::String>> elements;
			for (int i = 0; i < values.size(); ++i)
				elements.push_back(fbb.CreateString((const char*)values[i]));
			offset = fbb.CreateVector(elements).o;
			break;
		}	
		case NOS_BASE_TYPE_STRUCT: {
			std::vector<flatbuffers::Offset<const flatbuffers::Table*>> elements;
			for (int i = 0; i < values.size(); ++i)
				elements.push_back(CopyTable(fbb, type, flatbuffers::GetRoot<flatbuffers::Table>(values[i])));
			offset = fbb.CreateVector(elements).o;
			break;
		}
		case NOS_BASE_TYPE_ARRAY:
		default: {                    // Scalars and structs.
			assert(0);
		}
		}
	}
	return offset;
}


std::vector<uint8_t> GenerateVector(const nosTypeInfo* type, std::vector<const void*> inputs)
{
	flatbuffers::FlatBufferBuilder fbb;
	fbb.Finish(flatbuffers::Offset<flatbuffers::Vector<uint8_t>>(GenerateVector(fbb, type, std::move(inputs))));
	auto buf = fbb.Release();
	return std::vector<uint8_t>{flatbuffers::GetMutableRoot<uint8_t>(buf.data()), buf.data()+buf.size()};
}

} // namespace nos::engine