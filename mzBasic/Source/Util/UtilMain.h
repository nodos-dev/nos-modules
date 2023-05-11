#pragma once

#include <MediaZ/Plugin.h>

namespace mz
{

void RegisterUtil(NodeActionsMap& functions, std::set<flatbuffers::Type const*> const& types);

}