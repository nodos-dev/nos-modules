#include "Globals.h"
#include "cuda.h"

namespace nos::cudass
{
	UtilsProxy::ResourceManagerProxy<nosCUDABufferInfo> ResManager;
	void* PrimaryContext;
}