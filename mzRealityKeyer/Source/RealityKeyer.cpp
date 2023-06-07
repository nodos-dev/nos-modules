// Copyright MediaZ AS. All Rights Reserved.

#include "MediaZ/Helpers.hpp"

MZ_INIT()

#include "IBKPass1.frag.spv.dat"
#include "IBKPass2.frag.spv.dat"
#include "IBKHorizontalBlur.frag.spv.dat"

namespace mz
{
MZ_REGISTER_NAME(Core_Matte);
MZ_REGISTER_NAME(Unblurred_Core_Matte);
MZ_REGISTER_NAME(Core_Matte_Texture_Size);
MZ_REGISTER_NAME(Output);
MZ_REGISTER_NAME(Hard_Mask);
MZ_REGISTER_NAME(Hard_Mask_Horz_Blur);
MZ_REGISTER_NAME(Input);
MZ_REGISTER_NAME(Clean_Plate);
MZ_REGISTER_NAME(Clean_Plate_Mask);
MZ_REGISTER_NAME(Key_High_Brightness);
MZ_REGISTER_NAME(Core_Matte_Clean_Plate_Gain);
MZ_REGISTER_NAME(Core_Matte_Gamma_1);
MZ_REGISTER_NAME(Core_Matte_Gamma_2);
MZ_REGISTER_NAME(Core_Matte_Red_Weight);
MZ_REGISTER_NAME(Core_Matte_Green_Weight);
MZ_REGISTER_NAME(Core_Matte_Blue_Weight);
MZ_REGISTER_NAME(Core_Matte_Black_Point);
MZ_REGISTER_NAME(Core_Matte_White_Point);
MZ_REGISTER_NAME(Master_Gamma);
MZ_REGISTER_NAME(Master_Exposure);
MZ_REGISTER_NAME(Master_Offset);
MZ_REGISTER_NAME(Master_Saturation);
MZ_REGISTER_NAME(Master_Contrast);
MZ_REGISTER_NAME(Master_Contrast_Center);
MZ_REGISTER_NAME(Gamma);
MZ_REGISTER_NAME(Exposure);
MZ_REGISTER_NAME(Offset);
MZ_REGISTER_NAME(Saturation);
MZ_REGISTER_NAME(Contrast);
MZ_REGISTER_NAME(Contrast_Center);
MZ_REGISTER_NAME(Erode);
MZ_REGISTER_NAME(Softness);
MZ_REGISTER_NAME(Soft_Matte_Red_Weight);
MZ_REGISTER_NAME(Soft_Matte_Blue_Weight);
MZ_REGISTER_NAME(Soft_Matte_Gamma_1);
MZ_REGISTER_NAME(Soft_Matte_Gamma_2);
MZ_REGISTER_NAME(Soft_Matte_Clean_Plate_Gain);
MZ_REGISTER_NAME(softMatte422Filtering);
MZ_REGISTER_NAME(Core_Matte_Blend);
MZ_REGISTER_NAME(edgeSpillReplaceColor);
MZ_REGISTER_NAME(coreSpillReplaceColor);
MZ_REGISTER_NAME(Spill_Matte_Gamma);
MZ_REGISTER_NAME(Spill_Matte_Red_Weight);
MZ_REGISTER_NAME(Spill_Matte_Blue_Weight);
MZ_REGISTER_NAME(Spill_Matte_Gain);
MZ_REGISTER_NAME(Spill_RB_Weight);
MZ_REGISTER_NAME(Spill_Suppress_Weight);
MZ_REGISTER_NAME(spill422Filtering);
MZ_REGISTER_NAME(Screen_Subtract_Edge);
MZ_REGISTER_NAME(Screen_Subtract_Core);
MZ_REGISTER_NAME(Keep_Edge_Luma);
MZ_REGISTER_NAME(Keep_Core_Luma);
MZ_REGISTER_NAME(Final_Matte_Black_Point);
MZ_REGISTER_NAME(Final_Matte_White_Point);
MZ_REGISTER_NAME(Final_Matte_Gamma);
MZ_REGISTER_NAME(Spill_422_Filtering);
MZ_REGISTER_NAME(Core_Spill_Replace_Color);
MZ_REGISTER_NAME(Edge_Spill_Replace_Color);
MZ_REGISTER_NAME(Soft_Matte_422_Filtering);
MZ_REGISTER_NAME(Output_Type);
MZ_REGISTER_NAME(IBK_Pass_1_Shader);
MZ_REGISTER_NAME(IBK_Pass_2_Shader);
MZ_REGISTER_NAME(IBK_Horz_Blur_Shader);
MZ_REGISTER_NAME(IBK_Pass_1_Pass);
MZ_REGISTER_NAME(IBK_Pass_2_Pass);
MZ_REGISTER_NAME(IBK_Horz_Blur_Pass);
MZ_REGISTER_NAME(Blur_Radius);
MZ_REGISTER_NAME(Input_Texture_Size);

struct RealityKeyerContext : public NodeContext
{
	static mzResult GetShaders(size_t* outCount, mzName* outShaderNames, mzBuffer* outSpirvBufs)
	{
		*outCount = 3;
		if (!outShaderNames || !outSpirvBufs)
			return MZ_RESULT_SUCCESS;
		outShaderNames[0] = MZN_IBK_Pass_1_Shader;
		outShaderNames[1] = MZN_IBK_Pass_2_Shader;
		outShaderNames[2] = MZN_IBK_Horz_Blur_Shader;
		outSpirvBufs[0] = mzBuffer{.Data = (void*)IBKPass1_frag_spv, .Size = sizeof(IBKPass1_frag_spv)};
		outSpirvBufs[1] = mzBuffer{.Data = (void*)IBKPass2_frag_spv, .Size = sizeof(IBKPass2_frag_spv)};
		outSpirvBufs[2] = mzBuffer{.Data = (void*)IBKHorizontalBlur_frag_spv, .Size = sizeof(IBKHorizontalBlur_frag_spv)};
		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetPasses(size_t* outCount, mzPassInfo* outMzPassInfos)
	{
		*outCount = 3;
		if (!outMzPassInfos)
			return MZ_RESULT_SUCCESS;
		outMzPassInfos[0] = mzPassInfo{.Key = MZN_IBK_Pass_1_Pass, .Shader = MZN_IBK_Pass_1_Shader};
		outMzPassInfos[1] = mzPassInfo{.Key = MZN_IBK_Pass_2_Pass, .Shader = MZN_IBK_Pass_2_Shader};
		outMzPassInfos[2] = mzPassInfo{.Key = MZN_IBK_Horz_Blur_Pass, .Shader = MZN_IBK_Horz_Blur_Shader};
		return MZ_RESULT_SUCCESS;
	}

	void Run(const mzNodeExecuteArgs* args)
	{
		auto values = GetPinValues(args);
		auto outputTextureInfo = DeserializeTextureInfo(values[MZN_Output]);
		auto hardMaskTextureInfo = DeserializeTextureInfo(values[MZN_Hard_Mask]);
		auto hardMaskHorzBlurTextureInfo = DeserializeTextureInfo(values[MZN_Hard_Mask_Horz_Blur]);
		auto inputTextureInfo = DeserializeTextureInfo(values[MZN_Input]);
		auto cleanPlate = DeserializeTextureInfo(values[MZN_Clean_Plate]);
		auto cleanPlateMask = DeserializeTextureInfo(values[MZN_Clean_Plate_Mask]);

		mzCmd cmd;
		mzEngine.Begin(&cmd);
		// Pass 1 begin
		mzRunPassParams ibkPass1 = {};
		ibkPass1.Key = MZN_IBK_Pass_1_Pass; 
		std::vector ibkPass1Bindings = {
			ShaderBinding(MZN_Input, inputTextureInfo),
			ShaderBinding(MZN_Clean_Plate, cleanPlate),
			ShaderBinding(MZN_Key_High_Brightness, values[MZN_Key_High_Brightness]),
			ShaderBinding(MZN_Core_Matte_Clean_Plate_Gain, values[MZN_Core_Matte_Clean_Plate_Gain]),
			ShaderBinding(MZN_Core_Matte_Gamma_1, values[MZN_Core_Matte_Gamma_1]),
			ShaderBinding(MZN_Core_Matte_Gamma_2, values[MZN_Core_Matte_Gamma_2]),
			ShaderBinding(MZN_Core_Matte_Red_Weight, values[MZN_Core_Matte_Red_Weight]),
			ShaderBinding(MZN_Core_Matte_Green_Weight, values[MZN_Core_Matte_Green_Weight]),
			ShaderBinding(MZN_Core_Matte_Blue_Weight, values[MZN_Core_Matte_Blue_Weight]),
			ShaderBinding(MZN_Core_Matte_Black_Point, values[MZN_Core_Matte_Black_Point]),
			ShaderBinding(MZN_Core_Matte_White_Point, values[MZN_Core_Matte_White_Point]),
		};
		ibkPass1.Bindings = ibkPass1Bindings.data();
		ibkPass1.BindingCount = ibkPass1Bindings.size();
		ibkPass1.Output = hardMaskTextureInfo;
		mzEngine.RunPass(cmd, &ibkPass1);
		// Pass 1 end

		// Horz blur begin
		mzRunPassParams ibkHorzBlurPass = {};
		ibkHorzBlurPass.Key = MZN_IBK_Horz_Blur_Pass;
		float blurRadius = *(float*)values[MZN_Erode] + *(float*)values[MZN_Softness];
		mz::fb::vec2 blurInputSize(hardMaskTextureInfo.Info.Texture.Width, hardMaskTextureInfo.Info.Texture.Height);
		std::vector ibkHorzBlurPassBindings = {
			ShaderBinding(MZN_Input, hardMaskTextureInfo),
			ShaderBinding(MZN_Blur_Radius, blurRadius),
			ShaderBinding(MZN_Input_Texture_Size, blurInputSize),
		};
		ibkHorzBlurPass.Bindings = ibkHorzBlurPassBindings.data();
		ibkHorzBlurPass.BindingCount = ibkHorzBlurPassBindings.size();
		ibkHorzBlurPass.Output = hardMaskHorzBlurTextureInfo;
		mzEngine.RunPass(cmd, &ibkHorzBlurPass);
		// Horz blur end

		// Pass 2 begin
		mzRunPassParams ibkPass2 = {};
		ibkPass2.Key = MZN_IBK_Pass_2_Pass;
		mz::fb::vec2 coreMatteTextureSize(hardMaskTextureInfo.Info.Texture.Width, hardMaskTextureInfo.Info.Texture.Height);
		
		float softMatte422FilteringValue = *static_cast<float*>(values[MZN_Soft_Matte_422_Filtering]);
		mz::fb::vec2 softMatte422Filtering(1.0f - softMatte422FilteringValue, softMatte422FilteringValue * .5f);
		
		mz::fb::vec3 edgeSpillReplaceColor = *static_cast<mz::fb::vec3*>(values[MZN_Edge_Spill_Replace_Color]);
		edgeSpillReplaceColor.mutate_x(pow(2.0, edgeSpillReplaceColor.x()));
		edgeSpillReplaceColor.mutate_y(pow(2.0, edgeSpillReplaceColor.y()));
		edgeSpillReplaceColor.mutate_z(pow(2.0, edgeSpillReplaceColor.z()));

		mz::fb::vec3 coreSpillReplaceColor = *static_cast<mz::fb::vec3*>(values[MZN_Core_Spill_Replace_Color]);
		coreSpillReplaceColor.mutate_x(pow(2.0, coreSpillReplaceColor.x()));
		coreSpillReplaceColor.mutate_y(pow(2.0, coreSpillReplaceColor.y()));
		coreSpillReplaceColor.mutate_z(pow(2.0, coreSpillReplaceColor.z()));

		float spill422FilteringValue = *static_cast<float*>(values[MZN_Spill_422_Filtering]);
		mz::fb::vec2 spill422Filtering(1.0f - spill422FilteringValue, spill422FilteringValue * .5f);

		float masterGamma = *static_cast<float*>(values[MZN_Master_Gamma]);
		float masterExposure = *static_cast<float*>(values[MZN_Master_Exposure]);
		float masterOffset = *static_cast<float*>(values[MZN_Master_Offset]);
		float masterSaturation = *static_cast<float*>(values[MZN_Master_Saturation]);
		float masterContrast = *static_cast<float*>(values[MZN_Master_Contrast]);
		float masterContrastCenter = *static_cast<float*>(values[MZN_Master_Contrast_Center]);
		fb::vec3 gamma = *static_cast<fb::vec3*>(values[MZN_Gamma]);
		fb::vec3 exposure = *static_cast<fb::vec3*>(values[MZN_Exposure]);
		fb::vec3 offset = *static_cast<fb::vec3*>(values[MZN_Offset]);
		fb::vec3 saturation = *static_cast<fb::vec3*>(values[MZN_Saturation]);
		fb::vec3 contrast = *static_cast<fb::vec3*>(values[MZN_Contrast]);
		fb::vec3 contrastCenter = *static_cast<fb::vec3*>(values[MZN_Contrast_Center]);
		
		gamma.mutate_x(gamma.x() * masterGamma);
		gamma.mutate_y(gamma.y() * masterGamma);
		gamma.mutate_z(gamma.z() * masterGamma);

		exposure.mutate_x(exposure.x() * masterExposure);
		exposure.mutate_y(exposure.y() * masterExposure);
		exposure.mutate_z(exposure.z() * masterExposure);
		exposure.mutate_x(pow(2.0, exposure.x()));
		exposure.mutate_y(pow(2.0, exposure.y()));
		exposure.mutate_z(pow(2.0, exposure.z()));

		offset.mutate_x(offset.x() + masterOffset);
		offset.mutate_y(offset.y() + masterOffset);
		offset.mutate_z(offset.z() + masterOffset);

		saturation.mutate_x(saturation.x() * masterSaturation);
		saturation.mutate_y(saturation.y() * masterSaturation);
		saturation.mutate_z(saturation.z() * masterSaturation);

		contrast.mutate_x(contrast.x() * masterContrast);
		contrast.mutate_y(contrast.y() * masterContrast);
		contrast.mutate_z(contrast.z() * masterContrast);

		contrastCenter.mutate_x(contrastCenter.x() + masterContrastCenter);
		contrastCenter.mutate_y(contrastCenter.y() + masterContrastCenter);
		contrastCenter.mutate_z(contrastCenter.z() + masterContrastCenter);

		std::vector ibkPass2Bindings = {
			ShaderBinding(MZN_Input, inputTextureInfo),
			ShaderBinding(MZN_Clean_Plate, cleanPlate),
			ShaderBinding(MZN_Clean_Plate_Mask, cleanPlateMask),
			ShaderBinding(MZN_Core_Matte, hardMaskHorzBlurTextureInfo),
			ShaderBinding(MZN_Unblurred_Core_Matte, hardMaskTextureInfo),
			ShaderBinding(MZN_Core_Matte_Texture_Size, coreMatteTextureSize),
			ShaderBinding(MZN_Erode, values[MZN_Erode]),
			ShaderBinding(MZN_Softness, values[MZN_Softness]),
			ShaderBinding(MZN_Soft_Matte_Red_Weight, values[MZN_Soft_Matte_Red_Weight]),
			ShaderBinding(MZN_Soft_Matte_Blue_Weight, values[MZN_Soft_Matte_Blue_Weight]),
			ShaderBinding(MZN_Soft_Matte_Gamma_1, values[MZN_Soft_Matte_Gamma_1]),
			ShaderBinding(MZN_Soft_Matte_Gamma_2, values[MZN_Soft_Matte_Gamma_2]),
			ShaderBinding(MZN_Soft_Matte_Clean_Plate_Gain, values[MZN_Soft_Matte_Clean_Plate_Gain]),
			ShaderBinding(MZN_Soft_Matte_422_Filtering, softMatte422Filtering),
			ShaderBinding(MZN_Key_High_Brightness, values[MZN_Key_High_Brightness]),
			ShaderBinding(MZN_Core_Matte_Blend, values[MZN_Core_Matte_Blend]),
			ShaderBinding(MZN_Edge_Spill_Replace_Color, edgeSpillReplaceColor),
			ShaderBinding(MZN_Core_Spill_Replace_Color, coreSpillReplaceColor),
			ShaderBinding(MZN_Spill_Matte_Gamma, values[MZN_Spill_Matte_Gamma]),
			ShaderBinding(MZN_Spill_Matte_Red_Weight, values[MZN_Spill_Matte_Red_Weight]),
			ShaderBinding(MZN_Spill_Matte_Blue_Weight, values[MZN_Spill_Matte_Blue_Weight]),
			ShaderBinding(MZN_Spill_Matte_Gain, values[MZN_Spill_Matte_Gain]),
			ShaderBinding(MZN_Spill_RB_Weight, values[MZN_Spill_RB_Weight]),
			ShaderBinding(MZN_Spill_Suppress_Weight, values[MZN_Spill_Suppress_Weight]),
			ShaderBinding(MZN_Spill_422_Filtering, spill422Filtering),
			ShaderBinding(MZN_Screen_Subtract_Edge, values[MZN_Screen_Subtract_Edge]),
			ShaderBinding(MZN_Screen_Subtract_Core, values[MZN_Screen_Subtract_Core]),
			ShaderBinding(MZN_Keep_Edge_Luma, values[MZN_Keep_Edge_Luma]),
			ShaderBinding(MZN_Keep_Core_Luma, values[MZN_Keep_Core_Luma]),
			ShaderBinding(MZN_Final_Matte_Black_Point, values[MZN_Final_Matte_Black_Point]),
			ShaderBinding(MZN_Final_Matte_White_Point, values[MZN_Final_Matte_White_Point]),
			ShaderBinding(MZN_Final_Matte_Gamma, values[MZN_Final_Matte_Gamma]),
			ShaderBinding(MZN_Gamma, gamma),
			ShaderBinding(MZN_Exposure, exposure),
			ShaderBinding(MZN_Offset, offset),
			ShaderBinding(MZN_Saturation, saturation),
			ShaderBinding(MZN_Contrast, contrast),
			ShaderBinding(MZN_Contrast_Center, contrastCenter),
			ShaderBinding(MZN_Output_Type, values[MZN_Output_Type]),
		};
		ibkPass2.BindingCount = ibkPass2Bindings.size();
		ibkPass2.Bindings = ibkPass2Bindings.data();
		ibkPass2.Output = outputTextureInfo;
		mzEngine.RunPass(cmd, &ibkPass2);
		// Pass 2 end

		mzEngine.End(cmd);
	}
};


extern "C"
MZAPI_ATTR mzResult MZAPI_CALL mzExportNodeFunctions(size_t* outCount, mzNodeFunctions* outFunctions)
{
	*outCount = 1;
	if (!outFunctions)
		return MZ_RESULT_SUCCESS;
	auto& keyer = outFunctions[0];
	
	keyer.TypeName = MZ_NAME_STATIC("mz.realitykeyer.RealityKeyer");
	keyer.OnNodeCreated = [](const mzFbNode* node, void** ctx) {
		*ctx = new RealityKeyerContext();
		auto* RealityKeyerCtx = static_cast<RealityKeyerContext*>(*ctx);
		RealityKeyerCtx->Load(*node);
	};
	keyer.GetShaders = RealityKeyerContext::GetShaders,
	keyer.GetPasses = RealityKeyerContext::GetPasses,
	keyer.OnNodeDeleted = [](void* ctx, auto id) { delete static_cast<RealityKeyerContext*>(ctx); };
	keyer.ExecuteNode = [](void* ctx, const mzNodeExecuteArgs* args) {
		auto* RealityKeyerCtx = static_cast<RealityKeyerContext*>(ctx);
		RealityKeyerCtx->Run(args);
		return MZ_RESULT_SUCCESS;
	};
	return MZ_RESULT_SUCCESS;
}

} // namespace mz
