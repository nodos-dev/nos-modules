#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include "nosTensorSubsystem/nosTensorSubsystem.h"

NOS_INIT();
NOS_REGISTER_NAME(TextureToTensor);
NOS_REGISTER_NAME(TensorToTexture);
NOS_REGISTER_NAME(Layot)
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME(Out);


void RegisterTextureToTensor(nosNodeFunctions* outFunctions);
void RegisterTensorToTexture(nosNodeFunctions* outFunctions);

extern nosTensorSubsystem* nosTensor = nullptr;
extern nosCUDASubsystem* nosCUDA = nullptr;
extern nosVulkanSubsystem* nosVulkan = nullptr;
extern "C"
{
	NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outCount, nosNodeFunctions** outFunctions)
	{
		*outCount = (size_t)(2);
		if (!outFunctions)
			return NOS_RESULT_SUCCESS;

		nosResult returnRes;
		returnRes = nosEngine.RequestSubsystem(NOS_NAME_STATIC(NOS_TENSOR_SUBSYSTEM_NAME), 1, 0, (void**)&nosTensor);
		if (returnRes != NOS_RESULT_SUCCESS)
			return NOS_RESULT_FAILED;

		returnRes = nosEngine.RequestSubsystem(NOS_NAME_STATIC(NOS_VULKAN_SUBSYSTEM_NAME), NOS_VULKAN_SUBSYSTEM_VERSION_MAJOR, NOS_VULKAN_SUBSYSTEM_VERSION_MINOR, (void**)&nosVulkan);
		if (returnRes != NOS_RESULT_SUCCESS)
			return NOS_RESULT_FAILED;

		returnRes = nosEngine.RequestSubsystem(NOS_NAME_STATIC(NOS_CUDA_SUBSYSTEM_NAME), 1, 0, (void**)&nosCUDA);
		if (returnRes != NOS_RESULT_SUCCESS)
			return NOS_RESULT_FAILED;

		RegisterTextureToTensor(outFunctions[0]);
		RegisterTensorToTexture(outFunctions[1]);

		return NOS_RESULT_SUCCESS;
	}
}