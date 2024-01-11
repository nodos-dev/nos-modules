#include <Nodos/PluginHelpers.hpp>

#include <nosVulkanSubsystem/Helpers.hpp>

#include "Names.h"

namespace nos::utilities
{
extern nosVulkanSubsystem* nosVulkan;

NOS_REGISTER_NAME(RESIZE_PASS);
NOS_REGISTER_NAME(Method);
NOS_REGISTER_NAME(Size);
NOS_REGISTER_NAME_SPACED(Nos_Utilities_Resize, "nos.utilities.Resize")

static nosResult ExecuteNode(void* ctx, const nosNodeExecuteArgs* args)
{
	auto pins = GetPinValues(args);
	auto inputTex = vkss::DeserializeTextureInfo(pins[NSN_Input]);
	auto method = GetPinValue<uint32_t>(pins, NSN_Method);
		
	auto tex = vkss::DeserializeTextureInfo(pins[NSN_Output]);
	auto size = GetPinValue<nosVec2u>(pins, NSN_Size);
		
	if(size->x != tex.Info.Texture.Width ||
		size->y != tex.Info.Texture.Height)
	{
		auto prevTex = tex;
		prevTex.Memory = {};
		prevTex.Info.Texture.Width = size->x;
		prevTex.Info.Texture.Height = size->y;
		auto texFb = vkss::ConvertTextureInfo(prevTex);
		texFb.unscaled = true;
		auto texFbBuf = nos::Buffer::From(texFb);
		nosEngine.SetPinValue(args->Pins[1].Id, {.Data = texFbBuf.Data(), .Size = texFbBuf.Size()});
	}
    
	std::vector bindings = {vkss::ShaderBinding(NSN_Input, inputTex), vkss::ShaderBinding(NSN_Method, method)};
		
	nosRunPassParams resizeParam {
		.Key = NSN_RESIZE_PASS,
		.Bindings = bindings.data(),
		.BindingCount = 2,
		.Output = tex,
		.Wireframe = 0,
		.Benchmark = 0,
	};

	nosVulkan->RunPass(nullptr, &resizeParam);

	return NOS_RESULT_SUCCESS;
}

nosResult RegisterResize(nosNodeFunctions* out)
{
	out->ClassName = NSN_Nos_Utilities_Resize;
	out->ExecuteNode = ExecuteNode;
	return NOS_RESULT_SUCCESS;
}

} // namespace nos::utilities


