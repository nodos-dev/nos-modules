#include "GaussianBlur.hpp"
#include "GaussianBlur.frag.spv.dat"

namespace mz::filters
{

struct GaussBlurContext
{
	MzResourceShareInfo IntermediateTexture = {};
	mz::fb::UUID NodeId;

	GaussBlurContext(fb::Node const& node)
	{
		NodeId = *node.id();

		IntermediateTexture.info.texture.filter = MZ_TEXTURE_FILTER_LINEAR;
		IntermediateTexture.info.texture.usage = MzImageUsage(MZ_IMAGE_USAGE_RENDER_TARGET | MZ_IMAGE_USAGE_SAMPLED); 
	}

	~GaussBlurContext()
	{
		DestroyResources();
	}


	static MzResult GetShaders(size_t* outCount, const char** outShaderNames, MzBuffer* infos)
	{
		*outCount = 1;
		if (!infos)
			return MZ_RESULT_SUCCESS;

		outShaderNames[0] = "Gaussian_Blur";
		infos->Data = (void*)GaussianBlur_frag_spv;
		infos->Size = sizeof(GaussianBlur_frag_spv);

		return MZ_RESULT_SUCCESS;
	}

	static MzResult GetPasses(size_t* outCount, MzPassInfo* infos)
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
		if (IntermediateTexture.memory.handle)
		{
			mzEngine.Destroy(&IntermediateTexture);
		}
	}

	void SetupIntermediateTexture(MzResourceShareInfo* outputTexture)
	{
		
		if (IntermediateTexture.info.texture.width == outputTexture->info.texture.width &&
		    IntermediateTexture.info.texture.height == outputTexture->info.texture.height &&
		    IntermediateTexture.info.texture.format == outputTexture->info.texture.format)
			return;

		IntermediateTexture.info.texture.width = outputTexture->info.texture.width;
		IntermediateTexture.info.texture.height = outputTexture->info.texture.height;
		IntermediateTexture.info.texture.format = outputTexture->info.texture.format;
		
		mzEngine.Create(&IntermediateTexture); // TODO Check result
	}

	void Run(const MzNodeExecuteArgs* pins)
	{

		MzResourceShareInfo outputTexture;

		auto values = GetPinValues(pins);

		const MzResourceShareInfo input  = ValAsTex(values["Input"]);
		const MzResourceShareInfo output = ValAsTex(values["Output"]);
		const f32 softness = *(f32*)values["Softness"];
		const MzVec2 Kernel_Size = *(MzVec2*)values["Kernel_Size"];
		const MzVec2u Pass_Type = MzVec2u(0, 1);

		SetupIntermediateTexture(&outputTexture);

		std::vector<MzShaderBinding> bindings = {
			ShaderBinding("Input", input),
			ShaderBinding("Kernel_Size", Kernel_Size.x),
			ShaderBinding("Pass_Type", Pass_Type.x),
			ShaderBinding("Softness", softness),
		};
		
		// Horz pass
		MzRunPassParams pass = {
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

void RegisterGaussianBlur(MzNodeFunctions* out)
{
	out->TypeName = "mz.filters.GaussianBlur";
	out->OnNodeCreated = [](const MzFbNode* node, void** outCtxPtr) {
		*outCtxPtr = new mz::filters::GaussBlurContext(*node);
	};
	out->ExecuteNode = [](void* ctx, const MzNodeExecuteArgs* args) {
		((mz::filters::GaussBlurContext*)ctx)->Run(args);
		return MZ_RESULT_SUCCESS;
	};
	out->GetShaders = mz::filters::GaussBlurContext::GetShaders;
	out->GetPasses = mz::filters::GaussBlurContext::GetPasses;
}
