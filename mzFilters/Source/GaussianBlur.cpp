#include "GaussianBlur.hpp"
#include "GaussianBlur.frag.spv.dat"

#define SHADER_BINDING(name, varName, variable)						\
					MzShaderBinding name = {						\
							.VariableName = varName,				\
							.Value = {								\
								.Data = &variable,					\
								.Size = sizeof(variable)			\
							}										\
					}

#define SHADER_BUFFER_BINDING(name, varName, variable)				\
					MzShaderBinding name = {						\
							.VariableName = varName,				\
							.Value = variable						\
					}

namespace mz::filters
{

struct GaussBlurContext
{
	MzResourceShareInfo IntermediateTexture = {};
	mz::fb::UUID NodeId;

	GaussBlurContext(fb::Node const& node)
	{
		NodeId = *node.id();
		RegisterShaders();
		RegisterPasses();

		IntermediateTexture.info.texture.filter = MZ_TEXTURE_FILTER_LINEAR;
		IntermediateTexture.info.texture.usage = MzImageUsage(MZ_IMAGE_USAGE_RENDER_TARGET | MZ_IMAGE_USAGE_SAMPLED); 
	}

	~GaussBlurContext()
	{
		DestroyResources();
	}

	void RegisterShaders()
	{
		static bool registered = false;
		if (registered)
			return;
		mzEngine.RegisterShader("Gaussian_Blur",
		                        MzBuffer{
			                        .Data = (void*)GaussianBlur_frag_spv,
			                        .Size = sizeof(GaussianBlur_frag_spv)
		                        }); // TODO Check result
		registered = true;
	}

	void RegisterPasses()
	{
		auto key = "Gaussian_Blur_Pass_" + mz::UUID2STR(NodeId);
		MzPassInfo info = {};
		info.Key = key.c_str();
		info.Shader = "Gaussian_Blur";
		mzEngine.RegisterPass(&info); // TODO Check result
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
		// Create AllPinValues
		MzBuffer inputTexture = {};
		float* softnessPinValue = nullptr;
		MzVec2* kernelPinValue = nullptr;
		uint32_t passTypeValue = 0;
		
		MzResourceShareInfo outputTexture;

		// Check and set all of them
		for (size_t i{}; i < pins->PinCount; i++)
		{
			if (strcmp(pins->PinNames[i], "Input") == 0)
			{
				inputTexture = pins->PinValues[i];
			}
			if (strcmp(pins->PinNames[i], "Softness") == 0)
			{
				softnessPinValue = (float*)(pins->PinValues[i].Data);
				*softnessPinValue += 1.0f;
			}
			if (strcmp(pins->PinNames[i], "Kernel_Size") == 0)
			{
				kernelPinValue = (MzVec2*)(pins->PinValues[i].Data);
			}
			if (strcmp(pins->PinNames[i], "Output") == 0)
			{
				outputTexture = mz::ValAsTex(pins->PinValues[i].Data);
			}
		}

		SetupIntermediateTexture(&outputTexture);
		
		std::string key = "Gaussian_Blur_Pass_" + mz::UUID2STR(NodeId);

		auto intermTexData = SerializeResourceInfo(IntermediateTexture);
		auto intermTexBuf = MzBuffer {.Data = intermTexData.data(), .Size = intermTexData.size()};
		
		SHADER_BUFFER_BINDING(inputBinding, "Input", inputTexture);
		SHADER_BUFFER_BINDING(intermediateBinding, "Input", intermTexBuf);
		SHADER_BINDING(softnessBinding, "Softness", *softnessPinValue);
		SHADER_BINDING(kernelSizeXBinding, "Kernel_Size", kernelPinValue->x);
		SHADER_BINDING(kernelSizeYBinding, "Kernel_Size", kernelPinValue->y);
		SHADER_BINDING(passTypeBinding, "Pass_Type", passTypeValue);

		MzShaderBinding bindings[4];
		bindings[0] = inputBinding;
		bindings[1] = softnessBinding;
		bindings[2] = kernelSizeXBinding;
		bindings[3] = passTypeBinding;

		// Horz pass
		MzRunPassParams passParams = {};
		passParams.PassKey = key.c_str();
		passParams.Bindings = bindings;
		passParams.BindingCount = 4;
		passParams.Wireframe = false;
		passParams.Output = IntermediateTexture;

		MzCmd cmd;
		mzEngine.Begin(&cmd);
		
		mzEngine.RunPass(cmd, &passParams);

		passTypeValue = 1;

		// Vert pass
		bindings[0] = intermediateBinding;
		bindings[3] = kernelSizeYBinding;
		passParams.Bindings = bindings;
		passParams.Output = outputTexture;

		mzEngine.RunPass(cmd, &passParams);

		mzEngine.End(cmd);
	}
};

}

void RegisterGaussianBlur(MzNodeFunctions* out)
{
	out->TypeName = "mz.GaussianBlur";
	out->OnNodeCreated = [](const MzFbNode* node, void** outCtxPtr) {
		*outCtxPtr = new mz::filters::GaussBlurContext(*node);
	};
	out->ExecuteNode = [](void* ctx, const MzNodeExecuteArgs* args) {
		((mz::filters::GaussBlurContext*)ctx)->Run(args);
		return MZ_RESULT_SUCCESS;
	};
}
