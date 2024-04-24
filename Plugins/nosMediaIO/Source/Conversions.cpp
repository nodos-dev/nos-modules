#include <Nodos/PluginHelpers.hpp>
#include <nosVulkanSubsystem/Helpers.hpp>

#include "Conversion_generated.h"

#include <glm/glm.hpp>
#include <stb_image.h>
#include <stb_image_write.h>

namespace nos::MediaIO
{

static std::set<u32> const& FindDivisors(const u32 N)
{
	static std::map<u32, std::set<u32>> Map;

	auto it = Map.find(N);
	if(it != Map.end()) 
		return it->second;

	u32 p2 = 0, p3 = 0, p5 = 0;
	std::set<u32> D;

	if (N == 0)
		return D;

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

nosVec2u GetSuitableDispatchSize(nosVec2u dispatchSize, nosVec2u outSize, uint8_t bitWidth, bool interlaced)
{
	constexpr auto BestFit = [](i64 val, i64 res) -> u32 {
		if (res == 0)
			return val;
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

struct RGB2YCbCrNodeContext : NodeContext
{
	RGB2YCbCrNodeContext(const nosFbNode* node) : NodeContext(node)
	{
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		nos::NodeExecuteArgs execArgs(args);
		const nosBuffer* inputPinData = execArgs[NOS_NAME_STATIC("Source")].Data;
		const nosBuffer* outputPinData = execArgs[NOS_NAME_STATIC("Output")].Data;
		auto interlacedFlags = *InterpretPinValue<InterlacedFlags>(execArgs[NOS_NAME_STATIC("InterlacedFlags")].Data->Data);
		auto fmt = *InterpretPinValue<YCbCrPixelFormat>(execArgs[NOS_NAME("PixelFormat")].Data->Data);
		auto input = vkss::DeserializeTextureInfo(inputPinData->Data);
		auto& output = *InterpretPinValue<sys::vulkan::Buffer>(execArgs[NOS_NAME_STATIC("Output")].Data->Data);
		int isOutInterlaced = bool(interlacedFlags & (InterlacedFlags::OUTPUT_EVEN | InterlacedFlags::OUTPUT_ODD));

		nosVec2u ext = { input.Info.Texture.Width, input.Info.Texture.Height };
		nosVec2u yCbCrExt((fmt == YCbCrPixelFormat::V210) ? ((ext.x + (48 - ext.x % 48) % 48) / 3) << 1 : ext.x >> 1, 
						  ext.y >> isOutInterlaced);
		uint32_t bufSize = yCbCrExt.x * yCbCrExt.y * 4;
		// auto alignedSize = bufSize + (bufSize % 4096); 
		constexpr auto outMemoryFlags = NOS_MEMORY_FLAGS_DEVICE_MEMORY;
		if (output.size_in_bytes() != bufSize || output.memory_flags() != (nos::sys::vulkan::MemoryFlags)(outMemoryFlags))
		{
			nosResourceShareInfo bufInfo = {
				.Info = {
					.Type = NOS_RESOURCE_TYPE_BUFFER,
					.Buffer = nosBufferInfo{
						.Size = (uint32_t)bufSize,
						.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_STORAGE_BUFFER),
						.MemoryFlags = outMemoryFlags,
					}}};
			auto bufferDesc = vkss::ConvertBufferInfo(bufInfo);
			nosEngine.SetPinValueByName(NodeId, NOS_NAME_STATIC("Output"), Buffer::From(bufferDesc));
		}
		auto* dispatchSize = execArgs.GetPinData<nosVec2u>(NOS_NAME_STATIC("DispatchSize"));
		*dispatchSize = GetSuitableDispatchSize(*dispatchSize, yCbCrExt, fmt == YCbCrPixelFormat::V210 ? 10 : 8, isOutInterlaced);
		return nosVulkan->ExecuteGPUNode(this, args);
	}
};

nosResult RegisterRGB2YCbCr(nosNodeFunctions* funcs)
{
	NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.MediaIO.RGB2YCbCr"), RGB2YCbCrNodeContext, funcs);
	return NOS_RESULT_SUCCESS;
}

NOS_REGISTER_NAME(Resolution);

struct YCbCr2RGBNodeContext : NodeContext
{
	YCbCr2RGBNodeContext(const nosFbNode* node) : NodeContext(node)
	{
		AddPinValueWatcher(NSN_Resolution, [this](const nos::Buffer& newVal, auto const& old, bool first) {
			auto newDispatchSize = nosVec2u(120, 120);
			nosEngine.SetPinValueByName(NodeId, NOS_NAME_STATIC("DispatchSize"), Buffer::From(newDispatchSize));
		});

	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		nos::NodeExecuteArgs execArgs(args);
		auto interlacedFlags = *InterpretPinValue<InterlacedFlags>(execArgs[NOS_NAME_STATIC("InterlacedFlags")].Data->Data);
		auto fmt = *InterpretPinValue<YCbCrPixelFormat>(execArgs[NOS_NAME("PixelFormat")].Data->Data);
		auto res = *InterpretPinValue<nos::fb::vec2u>(execArgs[NOS_NAME("Resolution")].Data->Data);
		auto& output = *InterpretPinValue<sys::vulkan::Texture>(execArgs[NOS_NAME_STATIC("Output")].Data->Data);
		auto& input = *InterpretPinValue<sys::vulkan::Buffer>(execArgs[NOS_NAME_STATIC("Source")].Data->Data);

		int isOutInterlaced = bool(interlacedFlags & (InterlacedFlags::OUTPUT_EVEN | InterlacedFlags::OUTPUT_ODD));

		nosVec2u ext = { res.x(), res.y()};

		nosVec2u yCbCrExt((fmt == YCbCrPixelFormat::V210) ? ((ext.x + (48 - ext.x % 48) % 48) / 3) << 1 : ext.x >> 1, 
						  ext.y >> isOutInterlaced);
		
		if (output.width() != res.x() || 
			output.height() != res.y())
		{
			nosResourceShareInfo tex{.Info = {
				.Type = NOS_RESOURCE_TYPE_TEXTURE,
				.Texture = {
					.Width = res.x(),
					.Height = res.y(),
					.Format = NOS_FORMAT_R16G16B16A16_UNORM
				}
			}};
			sys::vulkan::TTexture texDef = vkss::ConvertTextureInfo(tex);
			nosEngine.SetPinValueByName(NodeId, NOS_NAME_STATIC("Output"), Buffer::From(texDef));
		}
		auto* dispatchSize = execArgs.GetPinData<nosVec2u>(NOS_NAME_STATIC("DispatchSize"));
		*dispatchSize = GetSuitableDispatchSize(*dispatchSize, yCbCrExt, fmt == YCbCrPixelFormat::V210 ? 10 : 8, isOutInterlaced);
		return nosVulkan->ExecuteGPUNode(this, args);
	}
};

nosResult RegisterYCbCr2RGB(nosNodeFunctions* funcs)
{
	NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.MediaIO.YCbCr2RGB"), YCbCr2RGBNodeContext, funcs);
	return NOS_RESULT_SUCCESS;
}


struct GammaLUTNodeContext : NodeContext
{
	nosResourceShareInfo StagingBuffer{};
	static constexpr auto SSBO_SIZE = 10; // Can have a better name.
	GammaLUTNodeContext(const nosFbNode* node) : NodeContext(node)
	{
		StagingBuffer.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
		StagingBuffer.Info.Buffer.Size = (1 << (SSBO_SIZE)) * sizeof(u16);
		StagingBuffer.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC);
		StagingBuffer.Info.Buffer.MemoryFlags = NOS_MEMORY_FLAGS_HOST_VISIBLE;
		nosVulkan->CreateResource(&StagingBuffer);
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		nos::NodeExecuteArgs execArgs(args);
		const nosBuffer* outputPinData = execArgs[NOS_NAME_STATIC("LUT")].Data;
		const auto& curve = *InterpretPinValue<GammaCurve>(execArgs[NOS_NAME_STATIC("GammaCurve")].Data->Data);
		const auto& dir = *InterpretPinValue<GammaConversionType>(execArgs[NOS_NAME_STATIC("Type")].Data->Data);
		if (Curve == curve && Type == dir)
			return NOS_RESULT_SUCCESS;
		constexpr auto outMemoryFlags = NOS_MEMORY_FLAGS_DEVICE_MEMORY;
		if (!OutputBuffer.Memory.Handle || OutputBuffer.Info.Buffer.MemoryFlags != outMemoryFlags)
		{
			OutputBuffer.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
			OutputBuffer.Info.Buffer.Size = StagingBuffer.Info.Buffer.Size;
			OutputBuffer.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_DST | NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_STORAGE_BUFFER);
			OutputBuffer.Info.Buffer.MemoryFlags = outMemoryFlags;
			auto pinBuf = vkss::ConvertBufferInfo(OutputBuffer);
			nosEngine.SetPinValueByName(NodeId, NOS_NAME_STATIC("LUT"), Buffer::From(pinBuf));
		}
		const auto& output = *InterpretPinValue<sys::vulkan::Buffer>(outputPinData->Data);
		OutputBuffer = vkss::ConvertToResourceInfo(output);
		auto data = GetGammaLUT(dir == GammaConversionType::DECODE, curve, SSBO_SIZE);
		auto* ptr = nosVulkan->Map(&StagingBuffer);
		memcpy(ptr, data.data(), data.size() * sizeof(u16));
		nosEngine.LogI("GammaLUT: Buffer updated");
		Curve = curve;
		Type = dir;
		nosCmd cmd;
		nosVulkan->Begin("GammaLUT Staging Copy", &cmd);
		nosVulkan->Copy(cmd, &StagingBuffer, &OutputBuffer, 0);
		nosVulkan->End(cmd, nullptr);
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
	NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.MediaIO.GammaLUT"), GammaLUTNodeContext, funcs);
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
		auto fmt = *InterpretPinValue<YCbCrPixelFormat>(execArgs[NOS_NAME_STATIC("PixelFormat")].Data->Data);
		const auto& dir = *InterpretPinValue<GammaConversionType>(execArgs[NOS_NAME_STATIC("Type")].Data->Data);
		auto narrowRange = *InterpretPinValue<bool>(execArgs[NOS_NAME_STATIC("NarrowRange")].Data->Data);
		glm::mat4 matrix = GetMatrix<f64>(colorSpace, fmt == YCbCrPixelFormat::V210 ? 10 : 8, narrowRange);
		if(dir == GammaConversionType::DECODE)
			matrix = glm::inverse(matrix);
		nosEngine.SetPinValueByName(NodeId, NOS_NAME_STATIC("Output"), Buffer::From(matrix));
		return NOS_RESULT_SUCCESS;
	}
};

nosResult RegisterColorSpaceMatrix(nosNodeFunctions* funcs)
{
	NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.MediaIO.ColorSpaceMatrix"), ColorSpaceMatrixNodeContext, funcs);
	return NOS_RESULT_SUCCESS;
}

}
