#include "GaussianBlur.hpp"

#include "MediaZ/Helpers.hpp"

#include "GaussianBlur.frag.spv.dat"

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

	void Run(MzNodeExecuteArgs* pins)
	{
		float* softnessPinValue;
		MzVec2* kernelPinValue;
		mz::fb::TTexture outputTexture;

		for(size_t i{}; i < pins->PinCount; i++)
		{
			if(strcmp(pins->PinNames[i], "Softness") == 0)
			{
				softnessPinValue =  static_cast<float*>(pins->PinValues[i].Data);
				*softnessPinValue += 1.0f;
			}
			
			if(strcmp(pins->PinNames[i], "Kernel_Size") == 0)
			{
				kernelPinValue = static_cast<MzVec2*>(pins->PinValues[i].Data);
			}
			if(strcmp(pins->PinNames[i], "Output") == 0)
			{
				flatbuffers::GetRoot<MzFbTexture>(pins->PinValues[i].Data)->UnPackTo(&outputTexture);
			}
		}

		SetupIntermediateTexture(&outputTexture);

		MzPassInfo horzPass;
		
	 //    // Pass 1 begin
	 //    app::TRunPass horzPass;
	 //    horzPass.pass = "Gaussian_Blur_Pass_" + UUID2STR(NodeId);
	 //    CopyUniformFromPin(horzPass, pins, "Input");
	 //    AddUniform(horzPass, "Softness", &softness, sizeof(softness));
	 //    AddUniform(horzPass, "Kernel_Size", &horzKernel, sizeof(horzKernel));
	 //    u32 passType = 0; // Horizontal pass
	 //    AddUniform(horzPass, "Pass_Type", &passType, sizeof(passType));
	 //    horzPass.output.reset(&IntermediateTexture);
	 //    // Pass 1 end
	 //    // Pass 2 begin
	 //    app::TRunPass vertPass;
	 //    vertPass.pass = "Gaussian_Blur_Pass_" + UUID2STR(NodeId);
	 //    AddUniform(vertPass, "Input", mz::Buffer::From(IntermediateTexture));
	 //    AddUniform(vertPass, "Softness", &softness, sizeof(softness));
	 //    AddUniform(vertPass, "Kernel_Size", &vertKernel, sizeof(vertKernel));
	 //    passType = 1; // Vertical pass
	 //    AddUniform(vertPass, "Pass_Type", &passType, sizeof(passType));
	 //    vertPass.output.reset(&outputTexture);
	 //    // Pass 2 end
	 //    // Run passes
	 //    GServices.MakeAPICalls(false, horzPass, vertPass);
	 //    vertPass.output.release();
	 //    horzPass.output.release();
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
		// (mz::filters::GaussBlurContext*)ctx->Run(args);
		return true;
	};
}
