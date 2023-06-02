// Copyright MediaZ AS. All Rights Reserved.

#include "MediaZ/Helpers.hpp"

MZ_INIT()

#include "IBKPass1.frag.spv.dat"
#include "IBKPass2.frag.spv.dat"
#include "IBKHorizontalBlur.frag.spv.dat"

namespace mz
{
MZ_REGISTER_NAME2(Core_Matte);
MZ_REGISTER_NAME2(Unblurred_Core_Matte);
MZ_REGISTER_NAME2(Core_Matte_Texture_Size);
MZ_REGISTER_NAME2(Output);
MZ_REGISTER_NAME2(Hard_Mask);
MZ_REGISTER_NAME2(Hard_Mask_Horz_Blur);
MZ_REGISTER_NAME2(Input);
MZ_REGISTER_NAME2(Clean_Plate);
MZ_REGISTER_NAME2(Clean_Plate_Mask);
MZ_REGISTER_NAME2(Key_High_Brightness);
MZ_REGISTER_NAME2(Core_Matte_Clean_Plate_Gain);
MZ_REGISTER_NAME2(Core_Matte_Gamma_1);
MZ_REGISTER_NAME2(Core_Matte_Gamma_2);
MZ_REGISTER_NAME2(Core_Matte_Red_Weight);
MZ_REGISTER_NAME2(Core_Matte_Green_Weight);
MZ_REGISTER_NAME2(Core_Matte_Blue_Weight);
MZ_REGISTER_NAME2(Core_Matte_Black_Point);
MZ_REGISTER_NAME2(Core_Matte_White_Point);
MZ_REGISTER_NAME2(Master_Gamma);
MZ_REGISTER_NAME2(Master_Exposure);
MZ_REGISTER_NAME2(Master_Offset);
MZ_REGISTER_NAME2(Master_Saturation);
MZ_REGISTER_NAME2(Master_Contrast);
MZ_REGISTER_NAME2(Master_Contrast_Center);
MZ_REGISTER_NAME2(Gamma);
MZ_REGISTER_NAME2(Exposure);
MZ_REGISTER_NAME2(Offset);
MZ_REGISTER_NAME2(Saturation);
MZ_REGISTER_NAME2(Contrast);
MZ_REGISTER_NAME2(Contrast_Center);
MZ_REGISTER_NAME2(Erode);
MZ_REGISTER_NAME2(Softness);
MZ_REGISTER_NAME2(Soft_Matte_Red_Weight);
MZ_REGISTER_NAME2(Soft_Matte_Blue_Weight);
MZ_REGISTER_NAME2(Soft_Matte_Gamma_1);
MZ_REGISTER_NAME2(Soft_Matte_Gamma_2);
MZ_REGISTER_NAME2(Soft_Matte_Clean_Plate_Gain);
MZ_REGISTER_NAME2(softMatte422Filtering);
MZ_REGISTER_NAME2(Core_Matte_Blend);
MZ_REGISTER_NAME2(edgeSpillReplaceColor);
MZ_REGISTER_NAME2(coreSpillReplaceColor);
MZ_REGISTER_NAME2(Spill_Matte_Gamma);
MZ_REGISTER_NAME2(Spill_Matte_Red_Weight);
MZ_REGISTER_NAME2(Spill_Matte_Blue_Weight);
MZ_REGISTER_NAME2(Spill_Matte_Gain);
MZ_REGISTER_NAME2(Spill_RB_Weight);
MZ_REGISTER_NAME2(Spill_Suppress_Weight);
MZ_REGISTER_NAME2(spill422Filtering);
MZ_REGISTER_NAME2(Screen_Subtract_Edge);
MZ_REGISTER_NAME2(Screen_Subtract_Core);
MZ_REGISTER_NAME2(Keep_Edge_Luma);
MZ_REGISTER_NAME2(Keep_Core_Luma);
MZ_REGISTER_NAME2(Final_Matte_Black_Point);
MZ_REGISTER_NAME2(Final_Matte_White_Point);
MZ_REGISTER_NAME2(Final_Matte_Gamma);
MZ_REGISTER_NAME2(Spill_422_Filtering);
MZ_REGISTER_NAME2(Core_Spill_Replace_Color);
MZ_REGISTER_NAME2(Edge_Spill_Replace_Color);
MZ_REGISTER_NAME2(Soft_Matte_422_Filtering);
MZ_REGISTER_NAME2(Output_Type);
MZ_REGISTER_NAME2(IBK_Pass_1_Shader);
MZ_REGISTER_NAME2(IBK_Pass_2_Shader);
MZ_REGISTER_NAME2(IBK_Horz_Blur_Shader);
MZ_REGISTER_NAME2(IBK_Pass_1_Pass);
MZ_REGISTER_NAME2(IBK_Pass_2_Pass);
MZ_REGISTER_NAME2(IBK_Horz_Blur_Pass);
MZ_REGISTER_NAME2(Blur_Radius);
MZ_REGISTER_NAME2(Input_Texture_Size);

struct RealityKeyerContext : public NodeContext
{
	static mzResult GetShaders(size_t* outCount, mzName* outShaderNames, mzBuffer* outSpirvBufs)
	{
		*outCount = 3;
		if (!outShaderNames || !outSpirvBufs)
			return MZ_RESULT_SUCCESS;
		outShaderNames[0] = IBK_Pass_1_Shader_Name;
		outShaderNames[1] = IBK_Pass_2_Shader_Name;
		outShaderNames[2] = IBK_Horz_Blur_Shader_Name;
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
		outMzPassInfos[0] = mzPassInfo{.Key = IBK_Pass_1_Pass_Name, .Shader = IBK_Pass_1_Shader_Name};
		outMzPassInfos[1] = mzPassInfo{.Key = IBK_Pass_2_Pass_Name, .Shader = IBK_Pass_2_Shader_Name};
		outMzPassInfos[2] = mzPassInfo{.Key = IBK_Horz_Blur_Pass_Name, .Shader = IBK_Horz_Blur_Shader_Name};
		return MZ_RESULT_SUCCESS;
	}

	void Run(const mzNodeExecuteArgs* args)
	{
		auto values = GetPinValues(args);
		auto outputTextureInfo = DeserializeTextureInfo(values[Output_Name]);
		auto hardMaskTextureInfo = DeserializeTextureInfo(values[Hard_Mask_Name]);
		auto hardMaskHorzBlurTextureInfo = DeserializeTextureInfo(values[Hard_Mask_Horz_Blur_Name]);
		auto inputTextureInfo = DeserializeTextureInfo(values[Input_Name]);
		auto cleanPlate = DeserializeTextureInfo(values[Clean_Plate_Name]);
		auto cleanPlateMask = DeserializeTextureInfo(values[Clean_Plate_Mask_Name]);

		mzCmd cmd;
		mzEngine.Begin(&cmd);
		// Pass 1 begin
		mzRunPassParams ibkPass1 = {};
		ibkPass1.Key = IBK_Pass_1_Pass_Name; 
		std::vector ibkPass1Bindings = {
			ShaderBinding(Input_Name, inputTextureInfo),
			ShaderBinding(Clean_Plate_Name, cleanPlate),
			ShaderBinding(Key_High_Brightness_Name, values[Key_High_Brightness_Name]),
			ShaderBinding(Core_Matte_Clean_Plate_Gain_Name, values[Core_Matte_Clean_Plate_Gain_Name]),
			ShaderBinding(Core_Matte_Gamma_1_Name, values[Core_Matte_Gamma_1_Name]),
			ShaderBinding(Core_Matte_Gamma_2_Name, values[Core_Matte_Gamma_2_Name]),
			ShaderBinding(Core_Matte_Red_Weight_Name, values[Core_Matte_Red_Weight_Name]),
			ShaderBinding(Core_Matte_Green_Weight_Name, values[Core_Matte_Green_Weight_Name]),
			ShaderBinding(Core_Matte_Blue_Weight_Name, values[Core_Matte_Blue_Weight_Name]),
			ShaderBinding(Core_Matte_Black_Point_Name, values[Core_Matte_Black_Point_Name]),
			ShaderBinding(Core_Matte_White_Point_Name, values[Core_Matte_White_Point_Name]),
		};
		ibkPass1.Bindings = ibkPass1Bindings.data();
		ibkPass1.BindingCount = ibkPass1Bindings.size();
		ibkPass1.Output = hardMaskTextureInfo;
		mzEngine.RunPass(cmd, &ibkPass1);
		// Pass 1 end

		// Horz blur begin
		mzRunPassParams ibkHorzBlurPass = {};
		ibkHorzBlurPass.Key = IBK_Horz_Blur_Pass_Name;
		float blurRadius = *(float*)values[Erode_Name] + *(float*)values[Softness_Name];
		mz::fb::vec2 blurInputSize(hardMaskTextureInfo.Info.Texture.Width, hardMaskTextureInfo.Info.Texture.Height);
		std::vector ibkHorzBlurPassBindings = {
			ShaderBinding(Input_Name, hardMaskTextureInfo),
			ShaderBinding(Blur_Radius_Name, blurRadius),
			ShaderBinding(Input_Texture_Size_Name, blurInputSize),
		};
		ibkHorzBlurPass.Bindings = ibkHorzBlurPassBindings.data();
		ibkHorzBlurPass.BindingCount = ibkHorzBlurPassBindings.size();
		ibkHorzBlurPass.Output = hardMaskHorzBlurTextureInfo;
		mzEngine.RunPass(cmd, &ibkHorzBlurPass);
		// Horz blur end

		// Pass 2 begin
		mzRunPassParams ibkPass2 = {};
		ibkPass2.Key = IBK_Pass_2_Pass_Name;
		mz::fb::vec2 coreMatteTextureSize(hardMaskTextureInfo.Info.Texture.Width, hardMaskTextureInfo.Info.Texture.Height);
		
		float softMatte422FilteringValue = *static_cast<float*>(values[Soft_Matte_422_Filtering_Name]);
		mz::fb::vec2 softMatte422Filtering(1.0f - softMatte422FilteringValue, softMatte422FilteringValue * .5f);
		
		mz::fb::vec3 edgeSpillReplaceColor = *static_cast<mz::fb::vec3*>(values[Edge_Spill_Replace_Color_Name]);
		edgeSpillReplaceColor.mutate_x(pow(2.0, edgeSpillReplaceColor.x()));
		edgeSpillReplaceColor.mutate_y(pow(2.0, edgeSpillReplaceColor.y()));
		edgeSpillReplaceColor.mutate_z(pow(2.0, edgeSpillReplaceColor.z()));

		mz::fb::vec3 coreSpillReplaceColor = *static_cast<mz::fb::vec3*>(values[Core_Spill_Replace_Color_Name]);
		coreSpillReplaceColor.mutate_x(pow(2.0, coreSpillReplaceColor.x()));
		coreSpillReplaceColor.mutate_y(pow(2.0, coreSpillReplaceColor.y()));
		coreSpillReplaceColor.mutate_z(pow(2.0, coreSpillReplaceColor.z()));

		float spill422FilteringValue = *static_cast<float*>(values[Spill_422_Filtering_Name]);
		mz::fb::vec2 spill422Filtering(1.0f - spill422FilteringValue, spill422FilteringValue * .5f);

		float masterGamma = *static_cast<float*>(values[Master_Gamma_Name]);
		float masterExposure = *static_cast<float*>(values[Master_Exposure_Name]);
		float masterOffset = *static_cast<float*>(values[Master_Offset_Name]);
		float masterSaturation = *static_cast<float*>(values[Master_Saturation_Name]);
		float masterContrast = *static_cast<float*>(values[Master_Contrast_Name]);
		float masterContrastCenter = *static_cast<float*>(values[Master_Contrast_Center_Name]);
		fb::vec3 gamma = *static_cast<fb::vec3*>(values[Gamma_Name]);
		fb::vec3 exposure = *static_cast<fb::vec3*>(values[Exposure_Name]);
		fb::vec3 offset = *static_cast<fb::vec3*>(values[Offset_Name]);
		fb::vec3 saturation = *static_cast<fb::vec3*>(values[Saturation_Name]);
		fb::vec3 contrast = *static_cast<fb::vec3*>(values[Contrast_Name]);
		fb::vec3 contrastCenter = *static_cast<fb::vec3*>(values[Contrast_Center_Name]);
		
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
			ShaderBinding(Input_Name, inputTextureInfo),
			ShaderBinding(Clean_Plate_Name, cleanPlate),
			ShaderBinding(Clean_Plate_Mask_Name, cleanPlateMask),
			ShaderBinding(Core_Matte_Name, hardMaskHorzBlurTextureInfo),
			ShaderBinding(Unblurred_Core_Matte_Name, hardMaskTextureInfo),
			ShaderBinding(Core_Matte_Texture_Size_Name, coreMatteTextureSize),
			ShaderBinding(Erode_Name, values[Erode_Name]),
			ShaderBinding(Softness_Name, values[Softness_Name]),
			ShaderBinding(Soft_Matte_Red_Weight_Name, values[Soft_Matte_Red_Weight_Name]),
			ShaderBinding(Soft_Matte_Blue_Weight_Name, values[Soft_Matte_Blue_Weight_Name]),
			ShaderBinding(Soft_Matte_Gamma_1_Name, values[Soft_Matte_Gamma_1_Name]),
			ShaderBinding(Soft_Matte_Gamma_2_Name, values[Soft_Matte_Gamma_2_Name]),
			ShaderBinding(Soft_Matte_Clean_Plate_Gain_Name, values[Soft_Matte_Clean_Plate_Gain_Name]),
			ShaderBinding(Soft_Matte_422_Filtering_Name,softMatte422Filtering),
			ShaderBinding(Key_High_Brightness_Name, values[Key_High_Brightness_Name]),
			ShaderBinding(Core_Matte_Blend_Name, values[Core_Matte_Blend_Name]),
			ShaderBinding(Edge_Spill_Replace_Color_Name,edgeSpillReplaceColor),
			ShaderBinding(Core_Spill_Replace_Color_Name,coreSpillReplaceColor),
			ShaderBinding(Spill_Matte_Gamma_Name, values[Spill_Matte_Gamma_Name]),
			ShaderBinding(Spill_Matte_Red_Weight_Name, values[Spill_Matte_Red_Weight_Name]),
			ShaderBinding(Spill_Matte_Blue_Weight_Name, values[Spill_Matte_Blue_Weight_Name]),
			ShaderBinding(Spill_Matte_Gain_Name, values[Spill_Matte_Gain_Name]),
			ShaderBinding(Spill_RB_Weight_Name, values[Spill_RB_Weight_Name]),
			ShaderBinding(Spill_Suppress_Weight_Name, values[Spill_Suppress_Weight_Name]),
			ShaderBinding(Spill_422_Filtering_Name,spill422Filtering),
			ShaderBinding(Screen_Subtract_Edge_Name, values[Screen_Subtract_Edge_Name]),
			ShaderBinding(Screen_Subtract_Core_Name, values[Screen_Subtract_Core_Name]),
			ShaderBinding(Keep_Edge_Luma_Name, values[Keep_Edge_Luma_Name]),
			ShaderBinding(Keep_Core_Luma_Name, values[Keep_Core_Luma_Name]),
			ShaderBinding(Final_Matte_Black_Point_Name, values[Final_Matte_Black_Point_Name]),
			ShaderBinding(Final_Matte_White_Point_Name, values[Final_Matte_White_Point_Name]),
			ShaderBinding(Final_Matte_Gamma_Name, values[Final_Matte_Gamma_Name]),
			ShaderBinding(Gamma_Name, gamma),
			ShaderBinding(Exposure_Name, exposure),
			ShaderBinding(Offset_Name, offset),
			ShaderBinding(Saturation_Name, saturation),
			ShaderBinding(Contrast_Name, contrast),
			ShaderBinding(Contrast_Center_Name, contrastCenter),
			ShaderBinding(Output_Type_Name, values[Output_Type_Name]),
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
	
	keyer.TypeName = "mz.realitykeyer.RealityKeyer";
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
