// Copyright MediaZ AS. All Rights Reserved.

#include "Track.h"

namespace mz
{

void RegisterFreeDNode(NodeActionsMap& functions);
void RegisterXyncNode(NodeActionsMap& functions);
void RegisterStypeNode(NodeActionsMap& functions);
void RegisterMoSysNode(NodeActionsMap& functions);
void RegisterController(NodeActionsMap& functions);

void RegisterTrack(NodeActionsMap& functions)
{
    RegisterFreeDNode(functions);
    RegisterXyncNode(functions);
    RegisterStypeNode(functions);
    RegisterMoSysNode(functions);
    RegisterController(functions);
}

} // namespace mz