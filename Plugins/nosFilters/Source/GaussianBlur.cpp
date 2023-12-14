#include "GaussianBlur.hpp"
#include "../Shaders/GaussianBlur.frag.spv.dat"

#include <nosVulkanSubsystem/nosVulkanSubsystem.h>
#include <nosVulkanSubsystem/Helpers.hpp>
#include <nosVulkanSubsystem/Types_generated.h>

nosVulkanSubsystem* nosVulkan = 0;

NOS_REGISTER_NAME(Input);
NOS_REGISTER_NAME(Output);
NOS_REGISTER_NAME(Softness);
NOS_REGISTER_NAME(Kernel_Size);
NOS_REGISTER_NAME(Pass_Type);
NOS_REGISTER_NAME(Gaussian_Blur_Pass);
NOS_REGISTER_NAME(Gaussian_Blur_Shader);

namespace nos::utilities
{

struct GaussBlurContext
{
	nosResourceShareInfo IntermediateTexture = {};
	nos::fb::UUID NodeId;
	nosVulkanSubsystem* vk = 0;

	GaussBlurContext(fb::Node const& node)
	{
		NodeId = *node.id();
		IntermediateTexture.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
		IntermediateTexture.Info.Texture.Filter = NOS_TEXTURE_FILTER_LINEAR;
		IntermediateTexture.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_RENDER_TARGET | NOS_IMAGE_USAGE_SAMPLED); 
		nosEngine.RequestSubsystem(NOS_NAME_STATIC("nos.sys.vulkan"), 1, 0, (void**)&vk);
	}

	~GaussBlurContext()
	{
		DestroyResources();
	}
	
	void DestroyResources()
	{
		if (IntermediateTexture.Memory.Handle)
		{
			vk->DestroyResource(&IntermediateTexture);
		}
	}

	void SetupIntermediateTexture(nosResourceShareInfo* outputTexture)
	{
		
		if (IntermediateTexture.Info.Texture.Width == outputTexture->Info.Texture.Width &&
		    IntermediateTexture.Info.Texture.Height == outputTexture->Info.Texture.Height &&
		    IntermediateTexture.Info.Texture.Format == outputTexture->Info.Texture.Format)
			return;

		DestroyResources();

		IntermediateTexture.Info.Texture.Width = outputTexture->Info.Texture.Width;
		IntermediateTexture.Info.Texture.Height = outputTexture->Info.Texture.Height;
		IntermediateTexture.Info.Texture.Format = outputTexture->Info.Texture.Format;
		
		vk->CreateResource(&IntermediateTexture); // TODO Check result
	}

	void Run(const nosNodeExecuteArgs* pins)
	{
		auto values = GetPinValues(pins);

		const nosResourceShareInfo input = nos::vkss::DeserializeTextureInfo(values[NSN_Input]);
		nosResourceShareInfo output = nos::vkss::DeserializeTextureInfo(values[NSN_Output]);
		const f32 softness = *(f32*)values[NSN_Softness];
		const nosVec2 kernelSize = *(nosVec2*)values[NSN_Kernel_Size];
		const nosVec2u passType = nosVec2u(0, 1);

		SetupIntermediateTexture(&output);

		std::vector<nosShaderBinding> bindings = {
			nos::vkss::ShaderBinding(NSN_Input, input),
			nos::vkss::ShaderBinding(NSN_Kernel_Size, kernelSize.x),
			nos::vkss::ShaderBinding(NSN_Pass_Type, passType.x),
			nos::vkss::ShaderBinding(NSN_Softness, softness),
		};
		
		// Horz pass
		nosRunPassParams pass = {
			.Key = NSN_Gaussian_Blur_Pass,
			.Bindings = bindings.data(),
			.BindingCount = (uint32_t)bindings.size(),
			.Output = IntermediateTexture,
			.Wireframe = false,
		};
		vk->RunPass(0, &pass);

		// Vert pass
		bindings[0] = nos::vkss::ShaderBinding(NSN_Input, IntermediateTexture);
		bindings[1] = nos::vkss::ShaderBinding(NSN_Kernel_Size, kernelSize.y);
		bindings[2] = nos::vkss::ShaderBinding(NSN_Pass_Type, passType.y);

		pass.Output = output;
		vk->RunPass(0, &pass);
	}
};

}

void RegisterGaussianBlur(nosNodeFunctions* out)
{
	out->ClassName = NOS_NAME_STATIC("nos.filters.GaussianBlur");
	out->OnNodeCreated = [](const nosFbNode* node, void** outCtxPtr) {
		*outCtxPtr = new nos::utilities::GaussBlurContext(*node);
	};
	out->OnNodeDeleted = [](void* ctx, nosUUID nodeId) {
		delete static_cast<nos::utilities::GaussBlurContext*>(ctx);
	};
	out->ExecuteNode = [](void* ctx, const nosNodeExecuteArgs* args) {
		((nos::utilities::GaussBlurContext*)ctx)->Run(args);
		return NOS_RESULT_SUCCESS;
	};

	auto ret = nosEngine.RequestSubsystem(NOS_NAME_STATIC("nos.sys.vulkan"), 1, 0, (void**)&nosVulkan);
	if (ret != NOS_RESULT_SUCCESS)
		return;
	
	nosShaderInfo shader =  {.Key=NSN_Gaussian_Blur_Shader, .Source = {.SpirvBlob  = {(void*)GaussianBlur_frag_spv, sizeof(GaussianBlur_frag_spv)}}};
	ret = nosVulkan->RegisterShaders(1, &shader);
	if (ret != NOS_RESULT_SUCCESS)
		return;
	nosPassInfo pass = {
		.Key = NSN_Gaussian_Blur_Pass,
		.Shader = NSN_Gaussian_Blur_Shader,
		.Blend = false,
		.MultiSample = 1,
	};
	nosVulkan->RegisterPasses(1, &pass);
}
