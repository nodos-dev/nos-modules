#include "GaussianBlur.hpp"
#include "../Shaders/GaussianBlur.frag.spv.dat"

namespace nos::utilities
{

NOS_REGISTER_NAME(Input);
NOS_REGISTER_NAME(Output);
NOS_REGISTER_NAME(Softness);
NOS_REGISTER_NAME(Kernel_Size);
NOS_REGISTER_NAME(Pass_Type);
NOS_REGISTER_NAME(Gaussian_Blur_Pass);
NOS_REGISTER_NAME(Gaussian_Blur_Shader);

struct GaussBlurContext
{
	nosResourceShareInfo IntermediateTexture = {};
	nos::fb::UUID NodeId;

	GaussBlurContext(fb::Node const& node)
	{
		NodeId = *node.id();
		IntermediateTexture.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
		IntermediateTexture.Info.Texture.Filter = NOS_TEXTURE_FILTER_LINEAR;
		IntermediateTexture.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_RENDER_TARGET | NOS_IMAGE_USAGE_SAMPLED); 
	}

	~GaussBlurContext()
	{
		DestroyResources();
	}


	static nosResult GetShaders(size_t* outCount, nosShaderInfo* outShaders)
	{
		*outCount = 1;
		if (!outShaders)
			return NOS_RESULT_SUCCESS;

		outShaders[0] = {.Key=NSN_Gaussian_Blur_Shader, .Source = {.SpirvBlob  = {(void*)GaussianBlur_frag_spv, sizeof(GaussianBlur_frag_spv)}}};
		return NOS_RESULT_SUCCESS;
	}

	static nosResult GetPasses(size_t* outCount, nosPassInfo* infos)
	{
		*outCount = 1;
		if (!infos)
			return NOS_RESULT_SUCCESS;

		infos->Key    = NSN_Gaussian_Blur_Pass;
		infos->Shader = NSN_Gaussian_Blur_Shader;
		infos->Blend = false;
		infos->MultiSample = 1;

		return NOS_RESULT_SUCCESS;
	}
	
	void DestroyResources()
	{
		if (IntermediateTexture.Memory.Handle)
		{
			nosEngine.Destroy(&IntermediateTexture);
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
		
		nosEngine.Create(&IntermediateTexture); // TODO Check result
	}

	void Run(const nosNodeExecuteArgs* pins)
	{
		auto values = GetPinValues(pins);

		const nosResourceShareInfo input = DeserializeTextureInfo(values[NSN_Input]);
		nosResourceShareInfo output = DeserializeTextureInfo(values[NSN_Output]);
		const f32 softness = *(f32*)values[NSN_Softness];
		const nosVec2 kernelSize = *(nosVec2*)values[NSN_Kernel_Size];
		const nosVec2u passType = nosVec2u(0, 1);

		SetupIntermediateTexture(&output);

		std::vector<nosShaderBinding> bindings = {
			ShaderBinding(NSN_Input, input),
			ShaderBinding(NSN_Kernel_Size, kernelSize.x),
			ShaderBinding(NSN_Pass_Type, passType.x),
			ShaderBinding(NSN_Softness, softness),
		};
		
		// Horz pass
		nosRunPassParams pass = {
			.Key = NSN_Gaussian_Blur_Pass,
			.Bindings = bindings.data(),
			.BindingCount = (uint32_t)bindings.size(),
			.Output = IntermediateTexture,
			.Wireframe = false,
		};
		nosEngine.RunPass(0, &pass);

		// Vert pass
		bindings[0] = ShaderBinding(NSN_Input, IntermediateTexture);
		bindings[1] = ShaderBinding(NSN_Kernel_Size, kernelSize.y);
		bindings[2] = ShaderBinding(NSN_Pass_Type, passType.y);


		pass.Output = output;
		nosEngine.RunPass(0, &pass);

	}
};

}

void RegisterGaussianBlur(nosNodeFunctions* out)
{
	out->TypeName = NOS_NAME_STATIC("nos.filters.GaussianBlur");
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
	out->GetShaders = nos::utilities::GaussBlurContext::GetShaders;
	out->GetPasses = nos::utilities::GaussBlurContext::GetPasses;
}
