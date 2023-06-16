#pragma once

#include <MediaZ/Helpers.hpp>

namespace mz::utilities
{

flatbuffers::uoffset_t GenerateOffset(
    flatbuffers::FlatBufferBuilder& fbb,
    const mzTypeInfo* type,
    const void* data);

void CopyInline(flatbuffers::FlatBufferBuilder& fbb, decltype(mzTypeInfo::Fields) fielddef,
    const flatbuffers::Table* table, size_t align, size_t size);

flatbuffers::uoffset_t CopyTable(
	flatbuffers::FlatBufferBuilder& fbb,
	const mzTypeInfo* type,
	const flatbuffers::Table* table);

flatbuffers::uoffset_t GenerateOffset(
    flatbuffers::FlatBufferBuilder& fbb,
    const mzTypeInfo* type,
    const void* data);

}