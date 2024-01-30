#include "Globals.h"
#include "cuda.h"
namespace nos::cudass
{
	UtilsProxy::ResourceManagerProxy<nosCUDABufferInfo> ResManager;
	void* PrimaryContext;
	std::unordered_map<int32_t, void*> IDContextMap;
}