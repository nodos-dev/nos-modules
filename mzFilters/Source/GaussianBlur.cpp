#include "GaussianBlur.hpp"
#include "GaussianBlur.frag.spv.dat"

namespace mz::utilities
{

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


	static mzResult GetShaders(size_t* outCount, const char** outShaderNames, mzBuffer* infos)
	{
		*outCount = 1;
		if (!infos)
			return MZ_RESULT_SUCCESS;

		outShaderNames[0] = "Gaussian_Blur";
		infos->Data = (void*)GaussianBlur_frag_spv;
		infos->Size = sizeof(GaussianBlur_frag_spv);

		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetPasses(size_t* outCount, mzPassInfo* infos)
	{
		*outCount = 1;
		if (!infos)
			return MZ_RESULT_SUCCESS;

		infos->Key = "Gaussian_Blur_Pass";
		infos->Shader = "Gaussian_Blur";
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

		mzResourceShareInfo outputTexture;

		auto values = GetPinValues(pins);

		const mzResourceShareInfo input  = DeserializeTextureInfo(values["Input"]);
		const mzResourceShareInfo output = DeserializeTextureInfo(values["Output"]);
		const f32 softness = *(f32*)values["Softness"];
		const mzVec2 Kernel_Size = *(mzVec2*)values["Kernel_Size"];
		const mzVec2u Pass_Type = mzVec2u(0, 1);

		SetupIntermediateTexture(&outputTexture);

		std::vector<mzShaderBinding> bindings = {
			ShaderBinding("Input", input),
			ShaderBinding("Kernel_Size", Kernel_Size.x),
			ShaderBinding("Pass_Type", Pass_Type.x),
			ShaderBinding("Softness", softness),
		};
		
		// Horz pass
		mzRunPassParams pass = {
			.PassKey = "Gaussian_Blur_Pass",
			.Bindings = bindings.data(),
			.BindingCount = (u32)bindings.size(),
			.Output = IntermediateTexture,
			.Wireframe = false,
		};
		mzEngine.RunPass(0, &pass);

		// Vert pass
		bindings[0].Resource = &IntermediateTexture;
		bindings[1].FixedSize = &Kernel_Size.y;
		bindings[2].FixedSize = &Pass_Type.y;
		pass.Output = outputTexture;
		mzEngine.RunPass(0, &pass);

	}
};

}

void RegisterGaussianBlur(mzNodeFunctions* out)
{
	out->TypeName = "mz.filters.GaussianBlur";
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
