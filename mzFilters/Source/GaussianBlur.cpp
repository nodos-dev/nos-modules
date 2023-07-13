#include "GaussianBlur.hpp"
#include "GaussianBlur.frag.spv.dat"

namespace mz::utilities
{

MZ_REGISTER_NAME(Input);
MZ_REGISTER_NAME(Output);
MZ_REGISTER_NAME(Softness);
MZ_REGISTER_NAME(Kernel_Size);
MZ_REGISTER_NAME(Pass_Type);
MZ_REGISTER_NAME(Gaussian_Blur_Pass);
MZ_REGISTER_NAME(Gaussian_Blur_Shader);

struct GaussBlurContext
{
	mzResourceShareInfo IntermediateTexture = {};
	mz::fb::UUID NodeId;

	GaussBlurContext(fb::Node const& node)
	{
		NodeId = *node.id();
		IntermediateTexture.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
		IntermediateTexture.Info.Texture.Filter = MZ_TEXTURE_FILTER_LINEAR;
		IntermediateTexture.Info.Texture.Usage = mzImageUsage(MZ_IMAGE_USAGE_RENDER_TARGET | MZ_IMAGE_USAGE_SAMPLED); 
	}

	~GaussBlurContext()
	{
		DestroyResources();
	}


	static mzResult GetShaders(size_t* outCount, mzShaderInfo* outShaders)
	{
		*outCount = 1;
		if (!outShaders)
			return MZ_RESULT_SUCCESS;

		outShaders[0] = {.Key=MZN_Gaussian_Blur_Shader, .SpirvBlob = {(void*)GaussianBlur_frag_spv, sizeof(GaussianBlur_frag_spv)}};
		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetPasses(size_t* outCount, mzPassInfo* infos)
	{
		*outCount = 1;
		if (!infos)
			return MZ_RESULT_SUCCESS;

		infos->Key    = MZN_Gaussian_Blur_Pass;
		infos->Shader = MZN_Gaussian_Blur_Shader;
		infos->Blend = false;
		infos->MultiSample = 1;

		return MZ_RESULT_SUCCESS;
	}
	
	void DestroyResources()
	{
		if (IntermediateTexture.Memory.Handle)
		{
			mzEngine.Destroy(&IntermediateTexture);
		}
	}

	void SetupIntermediateTexture(mzResourceShareInfo* outputTexture)
	{
		
		if (IntermediateTexture.Info.Texture.Width == outputTexture->Info.Texture.Width &&
		    IntermediateTexture.Info.Texture.Height == outputTexture->Info.Texture.Height &&
		    IntermediateTexture.Info.Texture.Format == outputTexture->Info.Texture.Format)
			return;

		IntermediateTexture.Info.Texture.Width = outputTexture->Info.Texture.Width;
		IntermediateTexture.Info.Texture.Height = outputTexture->Info.Texture.Height;
		IntermediateTexture.Info.Texture.Format = outputTexture->Info.Texture.Format;
		
		mzEngine.Create(&IntermediateTexture); // TODO Check result
	}

	void Run(const mzNodeExecuteArgs* pins)
	{
		auto values = GetPinValues(pins);

		const mzResourceShareInfo input = DeserializeTextureInfo(values[MZN_Input]);
		mzResourceShareInfo output = DeserializeTextureInfo(values[MZN_Output]);
		const f32 softness = *(f32*)values[MZN_Softness];
		const mzVec2 kernelSize = *(mzVec2*)values[MZN_Kernel_Size];
		const mzVec2u passType = mzVec2u(0, 1);

		SetupIntermediateTexture(&output);

		std::vector<mzShaderBinding> bindings = {
			ShaderBinding(MZN_Input, input),
			ShaderBinding(MZN_Kernel_Size, kernelSize.x),
			ShaderBinding(MZN_Pass_Type, passType.x),
			ShaderBinding(MZN_Softness, softness),
		};
		
		// Horz pass
		mzRunPassParams pass = {
			.Key = MZN_Gaussian_Blur_Pass,
			.Bindings = bindings.data(),
			.BindingCount = (uint32_t)bindings.size(),
			.Output = IntermediateTexture,
			.Wireframe = false,
		};
		mzEngine.RunPass(0, &pass);

		// Vert pass
		bindings[0].Resource = &IntermediateTexture;
		bindings[1].FixedSize = &kernelSize.y;
		bindings[2].FixedSize = &passType.y;
		pass.Output = output;
		mzEngine.RunPass(0, &pass);

	}
};

}

void RegisterGaussianBlur(mzNodeFunctions* out)
{
	out->TypeName = MZ_NAME_STATIC("mz.filters.GaussianBlur");
	out->OnNodeCreated = [](const mzFbNode* node, void** outCtxPtr) {
		*outCtxPtr = new mz::utilities::GaussBlurContext(*node);
	};
	out->ExecuteNode = [](void* ctx, const mzNodeExecuteArgs* args) {
		((mz::utilities::GaussBlurContext*)ctx)->Run(args);
		return MZ_RESULT_SUCCESS;
	};
	out->GetShaders = mz::utilities::GaussBlurContext::GetShaders;
	out->GetPasses = mz::utilities::GaussBlurContext::GetPasses;
}
