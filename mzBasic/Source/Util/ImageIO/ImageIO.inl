// Copyright MediaZ AS. All Rights Reserved.

namespace mz
{

void RegisterReadImageNode(NodeActionsMap&);
void RegisterWriteImageNode(NodeActionsMap&);

void  RegisterImageIO(NodeActionsMap& functions)
{
    RegisterReadImageNode(functions);
    RegisterWriteImageNode(functions);
}

} // namespace mz