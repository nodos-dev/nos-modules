#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>

NOS_INIT();
void RegisterWebRTCPlayer(nosNodeFunctions* outFunctions);
void RegisterWebRTCStreamer(nosNodeFunctions* outFunctions);
void RegisterWebRTCSignalingServer(nosNodeFunctions* outFunctions);
extern "C"
{
	NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outCount, nosNodeFunctions** outFunctions) {
		*outCount = (size_t)(3);
		if (!outFunctions)
			return NOS_RESULT_SUCCESS;

		RegisterWebRTCStreamer(outFunctions[0]);
		RegisterWebRTCPlayer(outFunctions[1]);
		RegisterWebRTCSignalingServer(outFunctions[2]);

		return NOS_RESULT_SUCCESS;
	}
}