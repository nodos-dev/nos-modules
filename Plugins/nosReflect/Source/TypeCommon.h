/*
 * Copyright MediaZ Teknoloji A.S. All Rights Reserved.
 */

#pragma once

#include <Nodos/PluginHelpers.hpp>

#include "Names.h"

namespace nos::reflect
{
void CopyInline(flatbuffers::FlatBufferBuilder& fbb, decltype(nosTypeInfo::Fields) fielddef,
	const flatbuffers::Table* table, size_t align, size_t size);

void CopyInline(
	flatbuffers::FlatBufferBuilder& fbb, uint16_t offset, const uint8_t* data, size_t align, size_t size);

void CopyInline2(flatbuffers::FlatBufferBuilder& fbb, const flatbuffers::FieldDef* fielddef,
	const flatbuffers::Table* table, size_t align, size_t size);

const flatbuffers::StructDef* GetUnionType(
	const flatbuffers::StructDef* parent,
	const flatbuffers::FieldDef* unionfield, 
	const flatbuffers::Table* table);

flatbuffers::uoffset_t CopyTable2(
	flatbuffers::FlatBufferBuilder& fbb,
	const flatbuffers::StructDef* objectdef,
	const flatbuffers::Table* table);

flatbuffers::uoffset_t CopyTable(
	flatbuffers::FlatBufferBuilder& fbb,
	const nosTypeInfo* type,
	const flatbuffers::Table* table);

flatbuffers::uoffset_t CopyArgs(
	flatbuffers::FlatBufferBuilder& fbb,
	const nosTypeInfo* type,
	NodeExecuteParams& table);

flatbuffers::uoffset_t GenerateOffset(
	flatbuffers::FlatBufferBuilder& fbb,
	const nosTypeInfo* type,
	const void* data);

std::vector<uint8_t> GenerateBuffer(
	const nosTypeInfo* type,
	const void* data);

std::vector<uint8_t> GenerateVector(
	const nosTypeInfo* type, 
	std::vector<const void*> inputs);

bool AreFlatBuffersEqual(const nosTypeInfo* type,
						 void* first,
						 void* second);

enum class CompareResult
{
	Equal,
	Greater,
	Less
};

template <CompareResult TExpected, typename T>
bool CompareFloatingPoint(T const& first, T const& second)
{
	if (first == second)
		return TExpected == CompareResult::Equal;
	if (first > second)
		return TExpected == CompareResult::Greater;
	return TExpected == CompareResult::Less;
}
	
template <CompareResult TExpected>
bool CompareMemory(const nosTypeInfo* type, const void* first, const void* second)
{
	auto& size = type->ByteSize;
	if constexpr (TExpected == CompareResult::Equal)
		return memcmp(first, second, size) == 0;

	// Recursively compare the fields.
	switch (type->BaseType)
	{
	case NOS_BASE_TYPE_STRUCT:
		{
			for (int i = 0; i < type->FieldCount; ++i)
			{
				auto field = &type->Fields[i];
				if (!CompareMemory<TExpected>(field->Type,
						reinterpret_cast<const char*>(first) + field->Offset,
						reinterpret_cast<const char*>(second) + field->Offset))
					return false;
			}
			return true;
		}
	case NOS_BASE_TYPE_FLOAT:
		{
			if (size == 4)
			{
				auto& firstFloat = *static_cast<const float*>(first);
				auto& secondFloat = *static_cast<const float*>(second);
				return CompareFloatingPoint<TExpected>(firstFloat, secondFloat);
			}
			auto& firstDouble = *static_cast<const double*>(first);
			auto& secondDouble = *static_cast<const double*>(second);
			return CompareFloatingPoint<TExpected>(firstDouble, secondDouble);
		}
	default:
		{
			auto res = memcmp(first, second, size);
			if (res == 0)
				return false;
			if (res > 0)
				return TExpected == CompareResult::Greater;
			return TExpected == CompareResult::Less;
		}
	}
}

CompareResult CompareStrings(const char* first, const char* second);

template <CompareResult TExpected>
bool CompareFlatBuffers(const nosTypeInfo* type,
	                 	void* first,
	                 	void* second)
{
	switch (type->BaseType)
	{
	case NOS_BASE_TYPE_STRUCT:
		{
			if (type->ByteSize)
			{
				return CompareMemory<TExpected>(type, first, second);
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
					{
						if (field1Exists)
							return CompareResult::Greater == TExpected;
						return CompareResult::Less == TExpected;
					}
					continue;
				}
				if (field1Exists && !field2Exists)
					return CompareResult::Greater == TExpected;
				if (!field1Exists && field2Exists)
					return CompareResult::Less == TExpected;
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
				if (!CompareFlatBuffers<TExpected>(field->Type, field1, field2))
					return false;
			}
			return true;
		}
	case NOS_BASE_TYPE_STRING:
		{
			auto firstString = static_cast<const char*>(first);
			auto secondString = static_cast<const char*>(second);
			return CompareStrings(firstString, secondString) == TExpected;
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
					if (!CompareMemory<TExpected>(elementType, vec1->data() + i * type->ByteSize, vec2->data() + i * type->ByteSize))
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
						if (!CompareFlatBuffers<TExpected>(elementType, (void*)elem1->c_str(), (void*)elem2->c_str()))
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
						if (!CompareFlatBuffers<TExpected>(elementType, (void*)elem1, (void*)elem2))
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
				return CompareMemory<TExpected>(type, first, second);
			}
			break;
		}
	}
	return true;
}
}