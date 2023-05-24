#include "GaussianBlur.hpp"

#include "MediaZ/Helpers.hpp"

#include "GaussianBlur.frag.spv.dat"

#define SHADER_BINDING(name, varName, variable, indexValue)			\
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
	mz::fb::TTexture IntermediateTexture;
	mz::fb::UUID NodeId;

	GaussBlurContext(fb::Node const& node)
	{
		NodeId = *node.id();
		// RegisterShaders();
		RegisterPasses();

		IntermediateTexture.filtering = mz::fb::Filtering::LINEAR;
		IntermediateTexture.usage = mz::fb::ImageUsage::RENDER_TARGET | mz::fb::ImageUsage::SAMPLED;
	}

	~GaussBlurContext()
	{
		// DestroyResources();
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
		auto key = "Gaussian_Blur_Pass_" + mz::Uuid2String(NodeId);
		MzPassInfo info = {};
		info.Key = key.c_str();
		info.Shader = "Gaussian_Blur";
		mzEngine.RegisterPass(&info); // TODO Check result
	}

	void DestroyResources()
	{
		// If handle is not null
		if (IntermediateTexture.handle)
		{
			// create a builder
			flatbuffers::FlatBufferBuilder fbb;
			// create a texture via already existed IntermediateTexture;
			const auto offset = mz::fb::CreateTexture(fbb, &IntermediateTexture);
			// serialize it
			fbb.Finish(offset);
			// get detached buffer
			flatbuffers::DetachedBuffer buf = fbb.Release();
			// get the buffer data as fb::Texture
			auto* tex = flatbuffers::GetMutableRoot<mz::fb::Texture>(buf.data());
			// Init a resource to remove it to root
			MzResource res = {
				.Type = MZ_RESOURCE_TYPE_TEXTURE,
				.Texture = tex
			};
			// Destroy it to root
			mzEngine.Destroy(&res);
			res.Texture->UnPackTo(&IntermediateTexture);
		}

		auto key = "Gaussian_Blur_Pass_" + mz::Uuid2String(NodeId);
		mzEngine.UnregisterPass(key.c_str()); // TODO Check result
	}

	void SetupIntermediateTexture(mz::fb::TTexture* outputTexture)
	{
		if (IntermediateTexture.width == outputTexture->width &&
		    IntermediateTexture.height == outputTexture->height &&
		    IntermediateTexture.format == outputTexture->format)
			return;

		IntermediateTexture.width = outputTexture->width;
		IntermediateTexture.height = outputTexture->height;
		IntermediateTexture.format = outputTexture->format;

		flatbuffers::FlatBufferBuilder fbb;
		const auto offset = mz::fb::CreateTexture(fbb, &IntermediateTexture);
		fbb.Finish(offset);
		flatbuffers::DetachedBuffer buf = fbb.Release();
		auto* tex = flatbuffers::GetMutableRoot<mz::fb::Texture>(buf.data());
		MzResource res = {
			.Type = MZ_RESOURCE_TYPE_TEXTURE,
			.Texture = tex
		};
		mzEngine.Create(&res); // TODO Check result
		res.Texture->UnPackTo(&IntermediateTexture);

	}

	void Run(const MzNodeExecuteArgs* pins)
	{
		// Create AllPinValues
		MzBuffer inputTexture = {};
		float* softnessPinValue = nullptr;
		MzVec2* kernelPinValue = nullptr;
		uint32_t passTypeValue = 0;
		
		mz::fb::TTexture outputTexture;

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
				flatbuffers::GetRoot<MzFbTexture>(pins->PinValues[i].Data)->UnPackTo(&outputTexture);
			}
		}

		SetupIntermediateTexture(&outputTexture);

		auto out = mz::Buffer::From(outputTexture);
		auto intermediate = mz::Buffer::From(IntermediateTexture);

		MzBuffer intermediateBuf = {
			.Data = intermediate.data(),
			.Size = intermediate.size()
		};
		std::string key = "Gaussian_Blur_Pass_" + mz::Uuid2String(NodeId);

		SHADER_BUFFER_BINDING(InputBinding, "Input", inputTexture);
		SHADER_BUFFER_BINDING(IntermediateBinding, "Input", intermediateBuf);
		SHADER_BINDING(SoftnessBinding, "Softness", *softnessPinValue);
		SHADER_BINDING(KernelSizeXBinding, "Kernel_Size", kernelPinValue->x);
		SHADER_BINDING(KernelSizeYBinding, "Kernel_Size", kernelPinValue->y);
		SHADER_BINDING(PassTypeBinding, "Pass_Type", passTypeValue);

		MzShaderBinding bindings[4];
		bindings[0] = InputBinding;
		bindings[1] = SoftnessBinding;
		bindings[2] = KernelSizeXBinding;
		bindings[3] = PassTypeBinding;

		// Horz pass
		MzRunPassParams passParams = {};
		passParams.PassKey = key.c_str();
		passParams.Bindings = bindings;
		passParams.BindingCount = 4;
		passParams.Wireframe = false;
		passParams.Output = intermediate.As<MzFbTexture>();
		passParams.Vertices = nullptr;
		
		mzEngine.RunPass(&passParams);

		passTypeValue = 1;

		// Vert pass
		bindings[0] = IntermediateBinding;
		bindings[3] = KernelSizeYBinding;
		passParams.Bindings = bindings;
		passParams.Output = out.As<MzFbTexture>();
		
		mzEngine.RunPass(&passParams);
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
		return true;
	};
}
