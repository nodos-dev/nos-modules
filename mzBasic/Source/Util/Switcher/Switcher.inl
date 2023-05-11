// Copyright MediaZ AS. All Rights Reserved.

#include "Swizzle.frag.spv.dat"
#include "TextureSwitcher.frag.spv.dat"

namespace mz
{

void RegisterSwitcher(NodeActionsMap& functions)
{
    functions["mz.Swizzle"] = NodeActions{
        .ShaderSource = [] { return ShaderSrc<sizeof(Swizzle_frag_spv)>(Swizzle_frag_spv); }
    };

    functions["mz.TextureSwitcher"] = NodeActions{
        .ShaderSource = [] { return ShaderSrc<sizeof(TextureSwitcher_frag_spv)>(TextureSwitcher_frag_spv); }
    };
}

} // namespace mz
