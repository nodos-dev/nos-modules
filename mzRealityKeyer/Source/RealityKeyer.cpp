// Copyright MediaZ AS. All Rights Reserved.

#include "MediaZ/Helpers.hpp"

MZ_INIT()

#include "IBKPass1.frag.spv.dat"
#include "IBKPass2.frag.spv.dat"
#include "IBKHorizontalBlur.frag.spv.dat"

namespace mz
{

struct RealityKeyerContext : public NodeContext
{
	static MzResult GetShaders(size_t* outCount, const char** outShaderNames, MzBuffer* outSpirvBufs)
	{
		*outCount = 3;
		if (!outShaderNames || !outSpirvBufs)
			return MZ_RESULT_SUCCESS;
		outShaderNames[0] = "IBK_Pass_1_Shader";
		outShaderNames[1] = "IBK_Pass_2_Shader";
		outShaderNames[2] = "IBK_Horz_Blur_Pass_Shader";
		outSpirvBufs[0] = MzBuffer{.Data = (void*)IBKPass1_frag_spv, .Size = sizeof(IBKPass1_frag_spv)};
		outSpirvBufs[1] = MzBuffer{.Data = (void*)IBKPass2_frag_spv, .Size = sizeof(IBKPass2_frag_spv)};
		outSpirvBufs[2] = MzBuffer{.Data = (void*)IBKHorizontalBlur_frag_spv, .Size = sizeof(IBKHorizontalBlur_frag_spv)};
		return MZ_RESULT_SUCCESS;
	}

	static MzResult GetPasses(size_t* outCount, MzPassInfo* outMzPassInfos)
	{
		*outCount = 3;
		if (!outMzPassInfos)
			return MZ_RESULT_SUCCESS;
		outMzPassInfos[0] = MzPassInfo{.Key = "IBK_Pass_1", .Shader = "IBK_Pass_1_Shader"};
		outMzPassInfos[1] = MzPassInfo{.Key = "IBK_Pass_2", .Shader = "IBK_Pass_2_Shader"};
		outMzPassInfos[2] = MzPassInfo{.Key = "IBK_Horz_Blur_Pass", .Shader = "IBK_Horz_Blur_Pass_Shader"};
		return MZ_RESULT_SUCCESS;
	}

	void Run(const MzNodeExecuteArgs* args)
	{
		auto values = GetPinValues(args);
		auto outputTextureInfo = DeserializeTextureInfo(values["Output"]);
		auto hardMaskTextureInfo = DeserializeTextureInfo(values["Hard_Mask"]);
		auto hardMaskHorzBlurTextureInfo = DeserializeTextureInfo(values["Hard_Mask_Horz_Blur"]);
		auto inputTextureInfo = DeserializeTextureInfo(values["Input"]);
		auto cleanPlate = DeserializeTextureInfo(values["Clean_Plate"]);
		auto cleanPlateMask = DeserializeTextureInfo(values["Clean_Plate_Mask"]);

		MzCmd cmd;
		mzEngine.Begin(&cmd);
		// Pass 1 begin
		MzRunPassParams ibkPass1 = {};
		ibkPass1.PassKey = "IBK_Pass_1"; 
		std::vector ibkPass1Bindings = {
			ShaderBinding("Input", inputTextureInfo),
			ShaderBinding("Clean_Plate", cleanPlate),
			ShaderBinding("Key_High_Brightness", values["Key_High_Brightness"]),
			ShaderBinding("Core_Matte_Clean_Plate_Gain", values["Core_Matte_Clean_Plate_Gain"]),
			ShaderBinding("Core_Matte_Gamma_1", values["Core_Matte_Gamma_1"]),
			ShaderBinding("Core_Matte_Gamma_2", values["Core_Matte_Gamma_2"]),
			ShaderBinding("Core_Matte_Red_Weight", values["Core_Matte_Red_Weight"]),
			ShaderBinding("Core_Matte_Green_Weight", values["Core_Matte_Green_Weight"]),
			ShaderBinding("Core_Matte_Blue_Weight", values["Core_Matte_Blue_Weight"]),
			ShaderBinding("Core_Matte_Black_Point", values["Core_Matte_Black_Point"]),
			ShaderBinding("Core_Matte_White_Point", values["Core_Matte_White_Point"]),
		};
		ibkPass1.Bindings = ibkPass1Bindings.data();
		ibkPass1.BindingCount = ibkPass1Bindings.size();
		ibkPass1.Output = hardMaskTextureInfo;
		mzEngine.RunPass(cmd, &ibkPass1);
		// Pass 1 end

		// Horz blur begin
		MzRunPassParams ibkHorzBlurPass = {};
		ibkHorzBlurPass.PassKey = "IBK_Horz_Blur_Pass";
		float blurRadius = *(float*)values["Erode"] + *(float*)values["Softness"];
		mz::fb::vec2 blurInputSize(hardMaskTextureInfo.info.texture.width, hardMaskTextureInfo.info.texture.height);
		std::vector ibkHorzBlurPassBindings = {
			ShaderBinding("Input", hardMaskTextureInfo),
			ShaderBinding("Blur_Radius", blurRadius),
			ShaderBinding("Input_Texture_Size", blurInputSize),
		};
		ibkHorzBlurPass.Bindings = ibkHorzBlurPassBindings.data();
		ibkHorzBlurPass.BindingCount = ibkHorzBlurPassBindings.size();
		ibkHorzBlurPass.Output = hardMaskHorzBlurTextureInfo;
		mzEngine.RunPass(cmd, &ibkHorzBlurPass);
		// Horz blur end

		// Pass 2 begin
		MzRunPassParams ibkPass2 = {};
		ibkPass2.PassKey = "IBK_Pass_2";
		mz::fb::vec2 coreMatteTextureSize(hardMaskTextureInfo.info.texture.width, hardMaskTextureInfo.info.texture.height);
		
		float softMatte422FilteringValue = *static_cast<float*>(values["Soft_Matte_422_Filtering"]);
		mz::fb::vec2 softMatte422Filtering(1.0f - softMatte422FilteringValue, softMatte422FilteringValue * .5f);
		
		mz::fb::vec3 edgeSpillReplaceColor = *static_cast<mz::fb::vec3*>(values["Edge_Spill_Replace_Color"]);
		edgeSpillReplaceColor.mutate_x(pow(2.0, edgeSpillReplaceColor.x()));
		edgeSpillReplaceColor.mutate_y(pow(2.0, edgeSpillReplaceColor.y()));
		edgeSpillReplaceColor.mutate_z(pow(2.0, edgeSpillReplaceColor.z()));

		mz::fb::vec3 coreSpillReplaceColor = *static_cast<mz::fb::vec3*>(values["Core_Spill_Replace_Color"]);
		coreSpillReplaceColor.mutate_x(pow(2.0, coreSpillReplaceColor.x()));
		coreSpillReplaceColor.mutate_y(pow(2.0, coreSpillReplaceColor.y()));
		coreSpillReplaceColor.mutate_z(pow(2.0, coreSpillReplaceColor.z()));

		float spill422FilteringValue = *static_cast<float*>(values["Spill_422_Filtering"]);
		mz::fb::vec2 spill422Filtering(1.0f - spill422FilteringValue, spill422FilteringValue * .5f);

		float masterGamma = *static_cast<float*>(values["Master_Gamma"]);
		float masterExposure = *static_cast<float*>(values["Master_Exposure"]);
		float masterOffset = *static_cast<float*>(values["Master_Offset"]);
		float masterSaturation = *static_cast<float*>(values["Master_Saturation"]);
		float masterContrast = *static_cast<float*>(values["Master_Contrast"]);
		float masterContrastCenter = *static_cast<float*>(values["Master_Contrast_Center"]);

		fb::vec3 gamma = *static_cast<fb::vec3*>(values["Gamma"]);
		fb::vec3 exposure = *static_cast<fb::vec3*>(values["Exposure"]);
		fb::vec3 offset = *static_cast<fb::vec3*>(values["Offset"]);
		fb::vec3 saturation = *static_cast<fb::vec3*>(values["Saturation"]);
		fb::vec3 contrast = *static_cast<fb::vec3*>(values["Contrast"]);
		fb::vec3 contrastCenter = *static_cast<fb::vec3*>(values["Contrast_Center"]);
		
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
			ShaderBinding("Input", inputTextureInfo),
			ShaderBinding("Clean_Plate", cleanPlate),
			ShaderBinding("Clean_Plate_Mask", cleanPlateMask),
			ShaderBinding("Core_Matte", hardMaskHorzBlurTextureInfo),
			ShaderBinding("Unblurred_Core_Matte", hardMaskTextureInfo),
			ShaderBinding("Core_Matte_Texture_Size", coreMatteTextureSize),
			ShaderBinding("Erode", values["Erode"]),
			ShaderBinding("Softness", values["Softness"]),
			ShaderBinding("Soft_Matte_Red_Weight", values["Soft_Matte_Red_Weight"]),
			ShaderBinding("Soft_Matte_Blue_Weight", values["Soft_Matte_Blue_Weight"]),
			ShaderBinding("Soft_Matte_Gamma_1", values["Soft_Matte_Gamma_1"]),
			ShaderBinding("Soft_Matte_Gamma_2", values["Soft_Matte_Gamma_2"]),
			ShaderBinding("Soft_Matte_Clean_Plate_Gain", values["Soft_Matte_Clean_Plate_Gain"]),
			ShaderBinding("Soft_Matte_422_Filtering", softMatte422Filtering),
			ShaderBinding("Key_High_Brightness", values["Key_High_Brightness"]),
			ShaderBinding("Core_Matte_Blend", values["Core_Matte_Blend"]),
			ShaderBinding("Edge_Spill_Replace_Color", edgeSpillReplaceColor),
			ShaderBinding("Core_Spill_Replace_Color", coreSpillReplaceColor),
			ShaderBinding("Spill_Matte_Gamma", values["Spill_Matte_Gamma"]),
			ShaderBinding("Spill_Matte_Red_Weight", values["Spill_Matte_Red_Weight"]),
			ShaderBinding("Spill_Matte_Blue_Weight", values["Spill_Matte_Blue_Weight"]),
			ShaderBinding("Spill_Matte_Gain", values["Spill_Matte_Gain"]),
			ShaderBinding("Spill_RB_Weight", values["Spill_RB_Weight"]),
			ShaderBinding("Spill_Suppress_Weight", values["Spill_Suppress_Weight"]),
			ShaderBinding("Spill_422_Filtering", spill422Filtering),
			ShaderBinding("Screen_Subtract_Edge", values["Screen_Subtract_Edge"]),
			ShaderBinding("Screen_Subtract_Core", values["Screen_Subtract_Core"]),
			ShaderBinding("Keep_Edge_Luma", values["Keep_Edge_Luma"]),
			ShaderBinding("Keep_Core_Luma", values["Keep_Core_Luma"]),
			ShaderBinding("Final_Matte_Black_Point", values["Final_Matte_Black_Point"]),
			ShaderBinding("Final_Matte_White_Point", values["Final_Matte_White_Point"]),
			ShaderBinding("Final_Matte_Gamma", values["Final_Matte_Gamma"]),
			ShaderBinding("Gamma", gamma),
			ShaderBinding("Exposure", exposure),
			ShaderBinding("Offset", offset),
			ShaderBinding("Saturation", saturation),
			ShaderBinding("Contrast", contrast),
			ShaderBinding("Contrast_Center", contrastCenter),
			ShaderBinding("Output_Type", values["Output_Type"]),
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
MZAPI_ATTR MzResult MZAPI_CALL mzExportNodeFunctions(size_t* outCount, MzNodeFunctions* outFunctions)
{
	*outCount = 1;
	if (!outFunctions)
		return MZ_RESULT_SUCCESS;
	auto& keyer = outFunctions[0];
	
	keyer.TypeName = "mz.realitykeyer.RealityKeyer";
	keyer.OnNodeCreated = [](const MzFbNode* node, void** ctx) {
		*ctx = new RealityKeyerContext();
		auto* RealityKeyerCtx = static_cast<RealityKeyerContext*>(*ctx);
		RealityKeyerCtx->Load(*node);
	};
	keyer.GetShaders = RealityKeyerContext::GetShaders,
	keyer.GetPasses = RealityKeyerContext::GetPasses,
	keyer.OnNodeDeleted = [](void* ctx, auto id) { delete static_cast<RealityKeyerContext*>(ctx); };
	keyer.ExecuteNode = [](void* ctx, const MzNodeExecuteArgs* args) {
		auto* RealityKeyerCtx = static_cast<RealityKeyerContext*>(ctx);
		RealityKeyerCtx->Run(args);
		return MZ_RESULT_SUCCESS;
	};
	return MZ_RESULT_SUCCESS;
}

} // namespace mz
