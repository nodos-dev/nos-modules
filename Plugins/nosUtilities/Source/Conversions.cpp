#include <Nodos/PluginHelpers.hpp>
#include <nosVulkanSubsystem/Helpers.hpp>

#include "Conversion_generated.h"

#include <glm/glm.hpp>
#include <stb_image.h>
#include <stb_image_write.h>

namespace nos::utilities
{
struct RGB2YCbCrNodeContext : NodeContext
{
	RGB2YCbCrNodeContext(const nosFbNode* node) : NodeContext(node)
	{
	}

	static std::set<u32> const& FindDivisors(const u32 N)
	{
		static std::map<u32, std::set<u32>> Map;

		auto it = Map.find(N);
		if(it != Map.end()) 
			return it->second;

		u32 p2 = 0, p3 = 0, p5 = 0;
		std::set<u32> D;
		u32 n = N;
		while(0 == n % 2) n /= 2, p2++;
		while(0 == n % 3) n /= 3, p3++;
		while(0 == n % 5) n /= 5, p5++;
	
		for(u32 i = 0; i <= p2; ++i)
			for(u32 j = 0; j <= p3; ++j)
				for(u32 k = 0; k <= p5; ++k)
					D.insert(pow(2, i) * pow(3, j) * pow(5, k));

		static std::mutex Lock;
		Lock.lock();
		std::set<u32> const& re = (Map[N] = std::move(D));
		Lock.unlock();
		return re;
	}

	nosVec2u GetSuitableDispatchSize(nosVec2u dispatchSize, nosVec2u outSize, uint8_t bitWidth, bool interlaced) const
	{
		constexpr auto BestFit = [](i64 val, i64 res) -> u32 {
			auto d = FindDivisors(res);
			auto it = d.upper_bound(val);
			if (it == d.begin())
				return *it;
			if (it == d.end())
				return res;
			const i64 hi = *it;
			const i64 lo = *--it;
			return u32(abs(val - lo) < abs(val - hi) ? lo : hi);
		};

		const u32 q = 0;//TODO: IsQuad(); ?
		f32 x = glm::clamp<u32>(dispatchSize.x, 1, outSize.x) * (1 + q) * (.25 * bitWidth - 1);
		f32 y = glm::clamp<u32>(dispatchSize.y, 1, outSize.y) * (1. + q) * (1 + uint8_t(interlaced));

		return nosVec2u(BestFit(x + .5, outSize.x >> (bitWidth - 5)),
						 BestFit(y + .5, outSize.y / 9));
	}


	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		nos::NodeExecuteArgs execArgs(args);
		const nosBuffer* inputPinData = execArgs[NOS_NAME_STATIC("Source")].Data;
		const nosBuffer* outputPinData = execArgs[NOS_NAME_STATIC("Output")].Data;
		auto conversionFlags = *InterpretPinValue<YCbCrConversionFlags>(execArgs[NOS_NAME_STATIC("ConversionFlags")].Data->Data);
		auto input = vkss::DeserializeTextureInfo(inputPinData->Data);
		auto& output = *InterpretPinValue<sys::vulkan::Buffer>(execArgs[NOS_NAME_STATIC("Output")].Data->Data);
		int isOutInterlaced = bool(conversionFlags & (YCbCrConversionFlags::OUTPUT_EVEN | YCbCrConversionFlags::OUTPUT_ODD));

		nosVec2u ext = { input.Info.Texture.Width, input.Info.Texture.Height };
		nosVec2u yCbCrExt((bool(conversionFlags & YCbCrConversionFlags::V210)) ? ((ext.x + (48 - ext.x % 48) % 48) / 3) << 1 : ext.x >> 1, 
						  ext.y >> isOutInterlaced);
		uint32_t bufSize = yCbCrExt.x * yCbCrExt.y * 4;
		// auto alignedSize = bufSize + (bufSize % 4096); 
		if (output.size_in_bytes() != bufSize)
		{
			nosResourceShareInfo bufInfo = {
				.Info = {
					.Type = NOS_RESOURCE_TYPE_BUFFER,
					.Buffer = nosBufferInfo{
						.Size = (uint32_t)bufSize,
						.Usage = nosBufferUsage(NOS_BUFFER_USAGE_STORAGE_BUFFER | NOS_BUFFER_USAGE_TRANSFER_DST |
												NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_DEVICE_MEMORY),
					}}};
			auto bufferDesc = vkss::ConvertBufferInfo(bufInfo);
			nosEngine.SetPinValueByName(NodeId, NOS_NAME_STATIC("Output"), Buffer::From(bufferDesc));
		}
		auto* dispatchSize = execArgs.GetPinData<nosVec2u>(NOS_NAME_STATIC("DispatchSize"));
		*dispatchSize = GetSuitableDispatchSize(*dispatchSize, yCbCrExt, bool(conversionFlags & YCbCrConversionFlags::V210) ? 10 : 8, isOutInterlaced);
		return nosVulkan->ExecuteGPUNode(this, args);
	}
};

nosResult RegisterRGB2YCbCr(nosNodeFunctions* funcs)
{
	NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.utilities.RGB2YCbCr"), RGB2YCbCrNodeContext, funcs);
	return NOS_RESULT_SUCCESS;
}

struct GammaLUTNodeContext : NodeContext
{
	static constexpr auto SSBO_SIZE = 10; // Can have a better name.
	GammaLUTNodeContext(const nosFbNode* node) : NodeContext(node)
	{
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		nos::NodeExecuteArgs execArgs(args);
		const nosBuffer* outputPinData = execArgs[NOS_NAME_STATIC("LUT")].Data;
		const auto& curve = *InterpretPinValue<GammaCurve>(execArgs[NOS_NAME_STATIC("GammaCurve")].Data->Data);
		const auto& dir = *InterpretPinValue<GammaConversionType>(execArgs[NOS_NAME_STATIC("Type")].Data->Data);
		if (Curve == curve && Type == dir)
			return NOS_RESULT_SUCCESS;
		if (!OutputBuffer.Memory.Handle)
		{
			OutputBuffer.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
			OutputBuffer.Info.Buffer.Size = (1 << (SSBO_SIZE)) * sizeof(u16);
			OutputBuffer.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_DST | NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_STORAGE_BUFFER | NOS_BUFFER_USAGE_DEVICE_MEMORY);
			auto pinBuf = vkss::ConvertBufferInfo(OutputBuffer);
			nosEngine.SetPinValueByName(NodeId, NOS_NAME_STATIC("LUT"), Buffer::From(pinBuf));
		}
		const auto& output = *InterpretPinValue<sys::vulkan::Buffer>(outputPinData->Data);
		OutputBuffer = vkss::ConvertToResourceInfo(output);
		auto data = GetGammaLUT(dir == GammaConversionType::DECODE, curve, SSBO_SIZE);
		auto* ptr = nosVulkan->Map(&OutputBuffer);
		memcpy(ptr, data.data(), data.size() * sizeof(u16));
		nosEngine.LogI("GammaLUT: Buffer updated");
		Curve = curve;
		Type = dir;
		return NOS_RESULT_SUCCESS;
	}
	
	static auto GetLUTFunction(bool toLinear, GammaCurve curve) -> f64 (*)(f64)
	{
		switch (curve)
		{
		case GammaCurve::REC709:
		default:
			return toLinear ? [](f64 c) -> f64 { return (c < 0.081) ? (c / 4.5) : pow((c + 0.099) / 1.099, 1.0 / 0.45); }
			: [](f64 c) -> f64 { return (c < 0.018) ? (c * 4.5) : (pow(c, 0.45) * 1.099 - 0.099); };
		case GammaCurve::HLG:
			return toLinear
				   ? [](f64 c)
						 -> f64 { return (c < 0.5) ? (c * c / 3) : (exp(c / 0.17883277 - 5.61582460179) + 0.02372241); }
			: [](f64 c) -> f64 {
				return (c < 1. / 12.) ? sqrt(c * 3) : (std::log(c - 0.02372241) * 0.17883277 + 1.00429346);
		};
		case GammaCurve::ST2084:
			return toLinear ? 
					[](f64 c) -> f64 { c = pow(c, 0.01268331); return pow(glm::max(c - 0.8359375f, 0.) / (18.8515625  - 18.6875 * c), 6.27739463); } : 
						[](f64 c) -> f64 { c = pow(c, 0.15930175); return pow((0.8359375 + 18.8515625 * c) / (1 + 18.6875 * c), 78.84375); };
		}
	}

	static std::vector<u16> GetGammaLUT(bool toLinear, GammaCurve curve, u16 bits)
	{
		std::vector<u16> re(1 << bits, 0.f);
		auto fn = GetLUTFunction(toLinear, curve);
		for (u32 i = 0; i < 1 << bits; ++i)
		{
			re[i] = u16(f64((1 << 16) - 1) * fn(f64(i) / f64((1 << bits) - 1)) + 0.5);
		}
		return re;
	}
	
	nosResourceShareInfo OutputBuffer{};
	std::optional<GammaCurve> Curve = std::nullopt;
	std::optional<GammaConversionType> Type = std::nullopt;
};

nosResult RegisterGammaLUT(nosNodeFunctions* funcs)
{
	NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.utilities.GammaLUT"), GammaLUTNodeContext, funcs);
	return NOS_RESULT_SUCCESS;
}

struct Buffer2TextureNodeContext : NodeContext
{
	Buffer2TextureNodeContext(const nosFbNode* node) : NodeContext(node)
	{
	}
	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		nos::NodeExecuteArgs execArgs(args);
		const auto& inputPinData = *InterpretPinValue<sys::vulkan::Buffer>(execArgs[NOS_NAME_STATIC("Input")].Data->Data);
		const nosBuffer* outputPinData = execArgs[NOS_NAME_STATIC("Output")].Data;
		const auto& output = *InterpretPinValue<sys::vulkan::Texture>(outputPinData->Data);
		const auto& size = *InterpretPinValue<fb::vec2u>(execArgs[NOS_NAME_STATIC("Size")].Data->Data);
		const auto& format = *InterpretPinValue<sys::vulkan::Format>(execArgs[NOS_NAME_STATIC("Format")].Data->Data);
		if (size.x() != output.width() ||
			size.y() != output.height() ||
			format != output.format())
		{
			nosResourceShareInfo tex{.Info = {
				.Type = NOS_RESOURCE_TYPE_TEXTURE,
				.Texture = {
					.Width = size.x(),
					.Height = size.y(),
					.Format = nosFormat(format)
				}
			}};
			sys::vulkan::TTexture texDef = vkss::ConvertTextureInfo(tex);
			nosEngine.SetPinValueByName(NodeId, NOS_NAME_STATIC("Output"), Buffer::From(texDef));
		}
		nosResourceShareInfo out = vkss::DeserializeTextureInfo(outputPinData->Data);
		nosResourceShareInfo in = vkss::ConvertToResourceInfo(inputPinData);

		if (!in.Memory.Handle || !out.Memory.Handle)
			return NOS_RESULT_SUCCESS;

		nosCmd cmd;
		nosVulkan->Begin("Buffer2Texture Copy", &cmd);
		nosVulkan->Copy(cmd, &in, &out, 0);
		nosGPUEvent event;
		nosCmdEndParams params{
.ForceSubmit = true,
.OutGPUEventHandle = &event
		};
		nosVulkan->End(cmd, &params);
		nosVulkan->WaitGpuEvent(&event, UINT_MAX);
		return NOS_RESULT_SUCCESS;
	}
};

nosResult RegisterBuffer2Texture(nosNodeFunctions* funcs)
{
	NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.utilities.Buffer2Texture"), Buffer2TextureNodeContext, funcs);
	return NOS_RESULT_SUCCESS;
}



struct ColorSpaceMatrixNodeContext : NodeContext
{
	static std::array<f64, 2> GetCoeffs(ColorSpace colorSpace)
	{
		switch (colorSpace)
		{
		case ColorSpace::REC601:
			return { .299, .114 };
		case ColorSpace::REC2020:
			return { .2627, .0593 };
		case ColorSpace::REC709:
		default:
			return { .2126, .0722 };
		}
	}

	template<class T>
	static glm::mat<4, 4, T> GetMatrix(ColorSpace colorSpace, u32 bitWidth, bool narrowRange)
	{
		// https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#MODEL_CONVERSION
		const auto [R, B] = GetCoeffs(colorSpace);
		const T G = T(1) - R - B; // Colorspace

		/*
		* https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#QUANTIZATION_NARROW
			Dequantization:
				n = Bit Width {8, 10, 12}
				Although unnoticable, quantization scales differs between bit widths
				This is merely mathematical perfection the error terms is less than 0.001
		*/

		const T QuantizationScalar = T(1 << (bitWidth - 8)) / T((1 << bitWidth) - 1);
		const T Y = narrowRange ? 219 * QuantizationScalar : 1;
		const T C = narrowRange ? 224 * QuantizationScalar : 1;
		const T YT = narrowRange ? 16 * QuantizationScalar : 0;
		const T CT = 128 * QuantizationScalar;
		const T CB = .5 * C / (B - 1);
		const T CR = .5 * C / (R - 1);

		const auto V0 = glm::vec<3, T>(R, G, B);
		const auto V1 = V0 - glm::vec<3, T>(0, 0, 1);
		const auto V2 = V0 - glm::vec<3, T>(1, 0, 0);

		return glm::transpose(glm::mat<4, 4, T>(
			glm::vec<4, T>(Y * V0, YT),
			glm::vec<4, T>(CB * V1, CT),
			glm::vec<4, T>(CR * V2, CT),
			glm::vec<4, T>(0, 0, 0, 1)));
	}

	ColorSpaceMatrixNodeContext(const nosFbNode* node) : NodeContext(node)
	{

	}
	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		nos::NodeExecuteArgs execArgs(args);
		const auto& colorSpace = *InterpretPinValue<ColorSpace>(execArgs[NOS_NAME_STATIC("ColorSpace")].Data->Data);
		auto conversionFlags = *InterpretPinValue<YCbCrConversionFlags>(execArgs[NOS_NAME_STATIC("ConversionFlags")].Data->Data);
		auto narrowRange = *InterpretPinValue<bool>(execArgs[NOS_NAME_STATIC("NarrowRange")].Data->Data);
		glm::mat4 matrix = GetMatrix<f64>(colorSpace, bool(conversionFlags & YCbCrConversionFlags::V210) ? 10 : 8, narrowRange);
		nosEngine.SetPinValueByName(NodeId, NOS_NAME_STATIC("Output"), Buffer::From(matrix));
		return NOS_RESULT_SUCCESS;
	}
};

nosResult RegisterColorSpaceMatrix(nosNodeFunctions* funcs)
{
	NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.utilities.ColorSpaceMatrix"), ColorSpaceMatrixNodeContext, funcs);
	return NOS_RESULT_SUCCESS;
}

}
