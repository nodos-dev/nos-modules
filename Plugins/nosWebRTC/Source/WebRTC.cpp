#include <MediaZ/PluginAPI.h>
#include <Builtins_generated.h>
#include <MediaZ/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>

MZ_INIT();
void RegisterWebRTCPlayer(mzNodeFunctions* outFunctions);
void RegisterWebRTCStreamer(mzNodeFunctions* outFunctions);
void RegisterWebRTCSignalingServer(mzNodeFunctions* outFunctions);
extern "C"
{
	MZAPI_ATTR mzResult MZAPI_CALL mzExportNodeFunctions(size_t* outCount, mzNodeFunctions** outFunctions) {
		*outCount = (size_t)(3);
		if (!outFunctions)
			return MZ_RESULT_SUCCESS;

		RegisterWebRTCStreamer(outFunctions[0]);
		RegisterWebRTCPlayer(outFunctions[1]);
		RegisterWebRTCSignalingServer(outFunctions[2]);

		return MZ_RESULT_SUCCESS;
	}
}