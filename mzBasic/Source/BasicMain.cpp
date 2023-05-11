#include <MediaZ/Plugin.h>

#include "AJA/AJAMain.h"
#include "Filters/FiltersMain.h"
#include "Math/MathMain.h"
#include "Util/UtilMain.h"
#include "VirtualStudio/VirtualStudioMain.h"

namespace mz
{
EngineNodeServices GServices;

extern "C"
{

void mzPluginSDK_API Register(NodeActionsMap& functions, EngineNodeServices services, std::set<flatbuffers::Type const*> const& types)
{
    // TODO: Breakup mzBasic into multiple plugins
    GServices = services;
    RegisterAJA(functions);
    RegisterFilters(functions);
    RegisterMath(functions);
    RegisterUtil(functions, types);
    RegisterTrack(functions);
    RegisterRealityKeyer(functions);
    RegisterCyclorama(functions);
    RegisterDistortion(functions);
}

}

}