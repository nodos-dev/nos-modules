#include "Globals.h"
#include "cuda.h"
namespace nos::cudass
{
	UtilsProxy::ResourceManagerProxy<nosCUDABufferInfo> ResManager;
	void* PrimaryContext;
	void* ActiveContext;
	std::unordered_map<uint64_t, void*> IDContextMap;
}