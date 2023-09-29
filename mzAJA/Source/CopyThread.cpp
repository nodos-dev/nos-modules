
#include "CopyThread.h"
#include "AppEvents_generated.h"
#include "Ring.h"

#include <chrono>

namespace mz
{

mz::Name const& CTGetName(rc<CopyThread> const& c)
{
	return c->Name();
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

auto LUTFn(bool input, GammaCurve curve) -> f64 (*)(f64)
{
	switch (curve)
	{
	case GammaCurve::REC709:
	default:
		return input ? [](f64 c) -> f64 { return (c < 0.081) ? (c / 4.5) : pow((c + 0.099) / 1.099, 1.0 / 0.45); }
					 : [](f64 c) -> f64 { return (c < 0.018) ? (c * 4.5) : (pow(c, 0.45) * 1.099 - 0.099); };
	case GammaCurve::HLG:
		return input
			   ? [](f64 c)
					 -> f64 { return (c < 0.5) ? (c * c / 3) : (exp(c / 0.17883277 - 5.61582460179) + 0.02372241); }
			   : [](f64 c) -> f64 {
					 return (c < 1. / 12.) ? sqrt(c * 3) : (std::log(c - 0.02372241) * 0.17883277 + 1.00429346);
				 };
	case GammaCurve::ST2084:
		return input ? 
				[](f64 c) -> f64 { c = pow(c, 0.01268331); return pow(glm::max(c - 0.8359375f, 0.) / (18.8515625  - 18.6875 * c), 6.27739463); } : 
				[](f64 c) -> f64 { c = pow(c, 0.15930175); return pow((0.8359375 + 18.8515625 * c) / (1 + 18.6875 * c), 78.84375); };
	}
}

static std::vector<u16> GetGammaLUT(bool input, GammaCurve curve, u16 bits)
{
	std::vector<u16> re(1 << bits, 0.f);
	auto fn = LUTFn(input, curve);
	for (u32 i = 0; i < 1 << bits; ++i)
	{
		re[i] = u16(f64((1 << 16) - 1) * fn(f64(i) / f64((1 << bits) - 1)) + 0.5);
	}
	return re;
}

u32 CopyThread::GetRingSize()
{
	return CpuRing->Size;
}

mz::Name const& CopyThread::Name() const
{
	return PinName;
}

bool CopyThread::IsInput() const
{
	return PinKind == mz::fb::ShowAs::OUTPUT_PIN;
}

bool CopyThread::IsQuad() const
{
	return AJADevice::IsQuad(Mode);
}

bool CopyThread::LinkSizeMismatch() const
{
	auto in0 = Client->Device->GetVPID(Channel);
	const bool SLSignal = CNTV2VPID::VPIDStandardIsSingleLink(in0.GetStandard());
	if (!IsQuad())
	{
		return !SLSignal;
	}
	
	if(Channel & 3)
		return true;

	// Maybe squares
	if (SLSignal)
	{
		auto in1 = Client->Device->GetVPID(NTV2Channel(Channel + 1));
		auto in2 = Client->Device->GetVPID(NTV2Channel(Channel + 2));
		auto in3 = Client->Device->GetVPID(NTV2Channel(Channel + 3));

		auto fmt0 = in0.GetVideoFormat();
		auto fmt1 = in1.GetVideoFormat();
		auto fmt2 = in2.GetVideoFormat();
		auto fmt3 = in3.GetVideoFormat();

		auto std0 = in0.GetStandard();
		auto std1 = in0.GetStandard();
		auto std2 = in0.GetStandard();
		auto std3 = in0.GetStandard();

		return !((fmt0 == fmt1) && (fmt0 == fmt1) &&
				 (fmt2 == fmt3) && (fmt2 == fmt3) &&
				 (fmt0 == fmt2) && (fmt0 == fmt2));
	}

	return false;
}

bool CopyThread::Interlaced() const
{
	return !IsProgressivePicture(Format);
}

void CopyThread::StartThread()
{
	if (IsInput())
		Worker = std::make_unique<InputConversionThread>();
	else
		Worker = std::make_unique<OutputConversionThread>();
	CpuRing->Exit = false;
	GpuRing->Exit = false;
	Run = true;
	std::string threadName("AJA ");
	threadName += IsInput() ? "In" : "Out";
	threadName += ": " + Name().AsString();

	Thread = std::thread([this, threadName] {
		Worker->Start();
		flatbuffers::FlatBufferBuilder fbb;
		// TODO: Add mzEngine.SetThreadName call.
		mzEngine.HandleEvent(CreateAppEvent(fbb, mz::app::CreateSetThreadNameDirect(fbb, (u64)Thread.native_handle(), (threadName + " Conversion Thread").c_str())));
		switch (this->PinKind)
		{
		default:
			UNREACHABLE;
		case mz::fb::ShowAs::INPUT_PIN:
			this->AJAOutputProc();
			break;
		case mz::fb::ShowAs::OUTPUT_PIN:
			this->AJAInputProc();
			break;
		}
		Worker->Stop();
	});

	flatbuffers::FlatBufferBuilder fbb;
	mzEngine.HandleEvent(CreateAppEvent(fbb, mz::app::CreateSetThreadNameDirect(fbb, (u64)Thread.native_handle(), threadName.c_str())));
}

mzVec2u CopyThread::Extent() const
{
	u32 width, height;
	Client->Device->GetExtent(Format, Mode, width, height);
	return mzVec2u(width, height);
}

void CopyThread::Stop()
{
	Run = false;
	GpuRing->Stop();
	CpuRing->Stop();
	if (Thread.joinable())
		Thread.join();
}

void CopyThread::Restart(u32 ringSize)
{
	assert(ringSize && ringSize < 200);
	Stop();
	CreateRings(ringSize);
	StartThread();
}

void CopyThread::SetFrame(u32 doubleBufferIndex)
{
	u32 frameIndex = GetFrameIndex(doubleBufferIndex);
	IsInput() ? Client->Device->SetInputFrame(Channel, frameIndex)
			  : Client->Device->SetOutputFrame(Channel, frameIndex);
	if (IsQuad())
		for (u32 i = Channel + 1; i < Channel + 4; ++i)
			IsInput() ? Client->Device->SetInputFrame(NTV2Channel(i), frameIndex)
					  : Client->Device->SetOutputFrame(NTV2Channel(i), frameIndex);
}

u32 CopyThread::GetFrameIndex(u32 doubleBufferIndex) const { return 2 * Channel + doubleBufferIndex; }

#define SSBO_SIZE 10

void CopyThread::UpdateCurve(enum GammaCurve curve)
{
	GammaCurve = curve;
	auto data = GetGammaLUT(IsInput(), GammaCurve, SSBO_SIZE);
	auto ptr = mzEngine.Map(&SSBO->Res);
	memcpy(ptr, data.data(), data.size() * sizeof(data[0]));
}

std::array<f64, 2> CopyThread::GetCoeffs() const
{
	switch (Colorspace)
	{
	case Colorspace::REC601:
		return {.299, .114};
	case Colorspace::REC2020:
		return {.2627, .0593};
	case Colorspace::REC709:
	default:
		return {.2126, .0722};
	}
}

template<class T>
glm::mat<4,4,T> CopyThread::GetMatrix() const
{
	// https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#MODEL_CONVERSION
	const auto [R, B] = GetCoeffs();
	const T G = T(1) - R - B; // Colorspace

	/*
	* https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#QUANTIZATION_NARROW
		Dequantization:
			n = Bit Width {8, 10, 12}
			Although unnoticable, quantization scales differs between bit widths
			This is merely mathematical perfection the error terms is less than 0.001
	*/

	const T QuantizationScalar = T(1 << (BitWidth() - 8)) / T((1 << BitWidth()) - 1);
	const T Y  = NarrowRange ? 219 * QuantizationScalar : 1;
	const T C  = NarrowRange ? 224 * QuantizationScalar : 1;
	const T YT = NarrowRange ? 16 * QuantizationScalar : 0;
	const T CT = 128 * QuantizationScalar;
	const T CB = .5 * C / (B - 1);
	const T CR = .5 * C / (R - 1);

	const auto V0 = glm::vec<3,T>(R, G, B);
	const auto V1 = V0 - glm::vec<3,T>(0, 0, 1);
	const auto V2 = V0 - glm::vec<3,T>(1, 0, 0);

	return glm::transpose(glm::mat<4,4,T>( 
			glm::vec<4,T>(Y  * V0, YT), 
			glm::vec<4,T>(CB * V1, CT), 
			glm::vec<4,T>(CR * V2, CT), 
			glm::vec<4,T>(0, 0, 0,  1)));
}

float GetFrameDurationFromFrameRate(auto frameRate)
{

	switch (frameRate)
	{
	case NTV2_FRAMERATE_6000  : return 1.f / 60.f;
	case NTV2_FRAMERATE_5994  : return 1001.f / 60000.f;
	case NTV2_FRAMERATE_3000  : return 1.f / 30.f;
	case NTV2_FRAMERATE_2997  :  return 1001.f / 30000.f;
	case NTV2_FRAMERATE_2500  :  return 1.f / 25.f;
	case NTV2_FRAMERATE_2400  :  return 1.f / 24.f;
	case NTV2_FRAMERATE_2398  :  return 1001.f / 24000.f;
	case NTV2_FRAMERATE_5000  :  return 1.f / 50.f;
	case NTV2_FRAMERATE_4800  :  return 1.f / 48.f;
	case NTV2_FRAMERATE_4795  :  return 1001.f / 48000.f;
	case NTV2_FRAMERATE_12000 :  return 1.f / 120.f;
	case NTV2_FRAMERATE_11988 : return 1001.f / 120000.f;
	case NTV2_FRAMERATE_1500  :  return 1.f / 15.f;
	case NTV2_FRAMERATE_1498  :  return 1001.f / 15000.f;
	default : return .02f;
	}
}

void CopyThread::Refresh()
{
	Client->Device->CloseChannel(Channel, Client->Input, IsQuad());
	Client->Device->RouteSignal(Channel, Format, Client->Input, Mode, Client->FBFmt());
	Format = IsInput() ? Client->Device->GetInputVideoFormat(Channel) : Format;
	Client->Device->SetRegisterWriteMode(Interlaced() ? NTV2_REGWRITE_SYNCTOFIELD : NTV2_REGWRITE_SYNCTOFRAME, Channel);
	CreateRings(GetRingSize());
}

void CopyThread::CreateRings(u32 size)
{
	const auto ext = Extent();
	GpuRing = MakeShared<GPURing>(ext, size);
	mzVec2u compressedExt((10 == BitWidth()) ? ((ext.x + (48 - ext.x % 48) % 48) / 3) << 1 : ext.x >> 1, ext.y >> u32(Interlaced()));
	CpuRing = MakeShared<CPURing>(compressedExt, size);
	mzTextureInfo info = {};
	info.Width  = compressedExt.x;
	info.Height = compressedExt.y;
	info.Format = MZ_FORMAT_R8G8B8A8_UINT;
	CompressedTex = MakeShared<GPURing::Resource>(info);
}

void CopyThread::InputUpdate(AJADevice::Mode &prevMode)
{
	auto fmt = Client->Device->GetInputVideoFormat(Channel);
	if (fmt != Format)
	{
		const bool changeRes = GetNTV2FrameGeometryFromVideoFormat(fmt) != GetNTV2FrameGeometryFromVideoFormat(Format);
		Refresh();

		if (changeRes) ChangePinResolution(Extent());
		auto fmtStr = NTV2VideoFormatToString(Format, true);
		mzEngine.SetPinValueByName(Client->Mapping.NodeId, mz::Name(PinName.AsString() + " Video Format"), { fmtStr.data(), fmtStr.size() + 1});
	}


	if (Mode == AJADevice::AUTO)
	{
		auto curMode = Client->Device->GetMode(Channel);
		if (prevMode != curMode)
		{
			prevMode = curMode;
			Refresh();
		}
	}

#pragma push_macro("Q")
#pragma push_macro("R")
#define Q(N)                                                                                                           \
	case N: {                                                                                                          \
		reg = kRegRXSDI##N##FrameCountLow; /* Is only getting the low reg enough? */                                   \
		break;                                                                                                         \
	}
#define R(C)                                                                                                           \
	switch (C + 1)                                                                                                     \
	{                                                                                                                  \
		Q(1) Q(2) Q(3) Q(4) Q(5) Q(6) Q(7) Q(8)                                                                        \
	}
	NTV2RXSDIStatusRegister reg;
	R(Channel);
#pragma pop_macro("Q")
#pragma pop_macro("R")

	u32 val;
	Client->Device->ReadRegister(reg, val);
	FrameIDCounter.store(val);
}



CopyThread::DMAInfo CopyThread::GetDMAInfo(mzResourceShareInfo& buffer, u32 doubleBufferIndex) const
{
	return {
		.Buffer = (u32*)mzEngine.Map(&buffer),
		.Pitch = CompressedTex->Res.Info.Texture.Width * 4,
		.Segments = CompressedTex->Res.Info.Texture.Height,
		.FrameIndex = GetFrameIndex(doubleBufferIndex)
	};
}

bool CopyThread::WaitForVBL(mzTextureFieldType fieldType)
{
	bool ret;
	auto ajaFieldID = GetAJAFieldID(fieldType);
	if (Interlaced())
	{
		ret = IsInput() ? Client->Device->WaitForInputFieldID(ajaFieldID, Channel)
						: Client->Device->WaitForOutputFieldID(ajaFieldID, Channel);
	}
	else
	{
		ret = IsInput() ? Client->Device->WaitForInputVerticalInterrupt(Channel)
						: Client->Device->WaitForOutputVerticalInterrupt(Channel);
	}

	return ret;
}


void CopyThread::AJAInputProc()
{
	NotifyDrop();
	Orphan(false);
	{
		std::stringstream ss;
		ss << "AJAIn Thread: " << std::this_thread::get_id();
		mzEngine.LogI(ss.str().c_str());
	}

	auto prevMode = Client->Device->GetMode(Channel);

	// Reset interrupt event status
	Client->Device->UnsubscribeInputVerticalEvent(Channel);
	Client->Device->SubscribeInputVerticalEvent(Channel);

	u32 doubleBufferIndex = 0;
	SetFrame(doubleBufferIndex);

	mzTextureFieldType currentField = Interlaced() ? MZ_TEXTURE_FIELD_TYPE_EVEN : MZ_TEXTURE_FIELD_TYPE_PROGRESSIVE;

	Parameters params = {};

	WaitForVBL(currentField);

	DebugInfo.Time = std::chrono::nanoseconds(0);
	DebugInfo.Counter = 0;

	DropCount = 0;
	u32 framesSinceLastDrop = 0;

	while (Run && !GpuRing->Exit)
	{
		auto cpuRingReadSize = CpuRing->Read.Pool.size();
		auto gpuRingWriteSize = GpuRing->Write.Pool.size();
		auto cpuRingWriteSize = CpuRing->Write.Pool.size();
		auto gpuRingReadSize = GpuRing->Read.Pool.size();
		mzEngine.WatchLog("AJAIn CPU Ring Read Size", std::to_string(cpuRingReadSize).c_str());
		mzEngine.WatchLog("AJAIn CPU Ring Write Size", std::to_string(cpuRingWriteSize).c_str());
		mzEngine.WatchLog("AJAIn GPU Ring Read Size", std::to_string(gpuRingReadSize).c_str());
		mzEngine.WatchLog("AJAIn GPU Ring Write Size", std::to_string(gpuRingWriteSize).c_str());
		mzEngine.WatchLog("AJAIn Total Frame Count", std::to_string(TotalFrameCount()).c_str());

		InputUpdate(prevMode);

		if (LinkSizeMismatch())
		{
			Orphan(true, "Quad - Single link mismatch");
			do
			{
				if (!Run || GpuRing->Exit || CpuRing->Exit)
				{
					goto EXIT;
				}
				WaitForVBL(currentField);
				InputUpdate(prevMode);
			} while (LinkSizeMismatch());
			Orphan(false);
		}

		if (!WaitForVBL(currentField))
		{
			currentField = Flipped(currentField);
			Orphan(true, "AJA Input has no signal");
			while (!WaitForVBL(currentField))
			{
				currentField = Flipped(currentField);
				if (!Run || GpuRing->Exit || CpuRing->Exit)
				{
					goto EXIT;
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(4));
				InputUpdate(prevMode);
			}
			InputUpdate(prevMode);
			Orphan(false);
		}
		
		auto fieldId = (u32(currentField) - 1);

		NTV2RegisterReads regs = {
			NTV2RegInfo(kRegRXSDI1FrameCountLow + Channel * (kRegRXSDI2FrameCountLow - kRegRXSDI1FrameCountLow))};
		Client->Device->ReadRegisters(regs);
		params.FrameNumber = regs.front().registerValue;
		if (Interlaced())
			params.FrameNumber = params.FrameNumber * 2 + fieldId;

		int ringSize = GetRingSize();
		int totalFrameCount = TotalFrameCount();

		CPURing::Resource* targetCpuRingSlot = nullptr;
		if (totalFrameCount >= ringSize || (!(targetCpuRingSlot = CpuRing->TryPush())))
		{
			DropCount++;
			GpuRing->ResetFrameCount = true;
			framesSinceLastDrop = 0;
			continue;
		}

		auto [Buf, Pitch, Segments, FrameIndex] = GetDMAInfo(targetCpuRingSlot->Res, doubleBufferIndex);
		params.T0 = Clock::now();
		if (Interlaced())
		{
			Client->Device->DMAReadSegments(FrameIndex,
											Buf,									// target CPU buffer address
											fieldId * Pitch, // source AJA buffer address
											Pitch,									// length of one line
											Segments,								// number of lines
											Pitch,		// increment target buffer one line on CPU memory
											Pitch * 2); // increment AJA card source buffer double the size of one line
														// for ex. next odd line or next even line
		}
		else
			Client->Device->DMAReadFrame(FrameIndex, Buf, Pitch * Segments, Channel);

		CpuRing->EndPush(targetCpuRingSlot);
		++framesSinceLastDrop;
		if (DropCount && framesSinceLastDrop == 50)
			NotifyDrop();

		if (!Interlaced())
		{
			SetFrame(doubleBufferIndex);
			doubleBufferIndex ^= 1;
		}

		params.FieldType = currentField;
		params.T1 = Clock::now();
		params.GR = GpuRing;
		params.CR = CpuRing;
		params.Colorspace = glm::inverse(GetMatrix<f64>());
		params.Shader = Client->Shader;
		params.CompressedTex = CompressedTex;
		params.SSBO = SSBO;
		params.Name = Name();
		params.Debug = Client->Debug;
		params.DispatchSize = GetSuitableDispatchSize();
		params.TransferInProgress = & TransferInProgress;
		Worker->Enqueue(params);
		currentField = Flipped(currentField);
	}
EXIT:

	CpuRing->Stop();
	GpuRing->Stop();

	if (Run)
	{
		SendDeleteRequest();
	}
}

mzVec2u CopyThread::GetSuitableDispatchSize() const
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

	const u32 q = IsQuad();
	f32 x = glm::clamp<u32>(Client->DispatchSizeX.load(), 1, CompressedTex->Res.Info.Texture.Width) * (1 + q) * (.25 * BitWidth() - 1);
	f32 y = glm::clamp<u32>(Client->DispatchSizeY.load(), 1, CompressedTex->Res.Info.Texture.Height) * (1. + q) * (1 + Interlaced());

	return mzVec2u(BestFit(x + .5, CompressedTex->Res.Info.Texture.Width >> (BitWidth() - 5)),
					 BestFit(y + .5, CompressedTex->Res.Info.Texture.Height / 9));
}

void CopyThread::NotifyRestart(RestartParams const& params)
{
	mzEngine.LogW("%s is notifying path for restart", Name().AsCStr());
	auto id = Client->GetPinId(mz::Name(Name()));
	auto args = Buffer::From(params);
	mzEngine.SendPathCommand(mzPathCommand{
		.PinId = id,
		.Command = MZ_PATH_COMMAND_TYPE_RESTART,
		.Execution = MZ_PATH_COMMAND_EXECUTION_TYPE_WALKBACK,
		.Args = mzBuffer { args.data(), args.size() }
	});
}

void CopyThread::NotifyDrop()
{
	if (!ConnectedPinCount)
		return;
	mzEngine.LogW("%s is notifying path about a drop event", Name().AsCStr());
	auto id = Client->GetPinId(mz::Name(Name()));
	auto args = Buffer::From(GpuRing->Size);
	mzEngine.SendPathCommand(mzPathCommand{
		.PinId = id,
		.Command = MZ_PATH_COMMAND_TYPE_NOTIFY_DROP,
		.Execution = MZ_PATH_COMMAND_EXECUTION_TYPE_NOTIFY_ALL_CONNECTIONS,
		.Args = mzBuffer { args.data(), args.size() }
	});
}

u32 CopyThread::TotalFrameCount()
{
	return CpuRing->TotalFrameCount() + GpuRing->TotalFrameCount() - (TransferInProgress ? 1 : 0);
}

void CopyThread::AJAOutputProc()
{
	flatbuffers::FlatBufferBuilder fbb;
	auto id = Client->GetPinId(PinName);
	auto hungerSignal = CreateAppEvent(fbb, mz::app::CreateScheduleRequest(fbb, mz::app::ScheduleRequestKind::PIN, &id, false));
	mzEngine.HandleEvent(hungerSignal);
	Orphan(false);
	{
		std::stringstream ss;
		ss << "AJAOut Thread: " << std::this_thread::get_id();
		mzEngine.LogI(ss.str().c_str());
	}


	while (Run && !CpuRing->Exit && TotalFrameCount() < GetRingSize())
		std::this_thread::yield();

	//Reset interrupt event status
	Client->Device->UnsubscribeOutputVerticalEvent(Channel);
	Client->Device->SubscribeOutputVerticalEvent(Channel);

	u32 doubleBufferIndex = 0;
	SetFrame(doubleBufferIndex);

	DropCount = 0;
	u32 framesSinceLastDrop = 0;

	while (Run && !CpuRing->Exit)
	{
		auto cpuRingReadSize = CpuRing->Read.Pool.size();
		auto gpuRingWriteSize = GpuRing->Write.Pool.size();
		auto cpuRingWriteSize = CpuRing->Write.Pool.size();
		auto gpuRingReadSize = GpuRing->Read.Pool.size();
		mzEngine.WatchLog("AJAOut CPU Ring Read Size", std::to_string(cpuRingReadSize).c_str());
		mzEngine.WatchLog("AJAOut CPU Ring Write Size", std::to_string(cpuRingWriteSize).c_str());
		mzEngine.WatchLog("AJAOut GPU Ring Read Size", std::to_string(gpuRingReadSize).c_str());
		mzEngine.WatchLog("AJAOut GPU Ring Write Size", std::to_string(gpuRingWriteSize).c_str());
		mzEngine.WatchLog("AJAOut Total Frame Count", std::to_string(TotalFrameCount()).c_str());

		ULWord lastVBLCount;
		Client->Device->GetOutputVerticalInterruptCount(lastVBLCount, Channel);

		if (auto sourceCpuRingSlot = CpuRing->BeginPop())
		{
			static auto prevFieldType = MZ_TEXTURE_FIELD_TYPE_UNKNOWN;
			auto fieldType = sourceCpuRingSlot->Res.Info.Texture.FieldType;
			if (Interlaced())
			{
				if (prevFieldType == MZ_TEXTURE_FIELD_TYPE_UNKNOWN)
					prevFieldType = fieldType;
				else if (prevFieldType != Flipped(fieldType))
					mzEngine.LogW("AJA Out: Field Atti");
				prevFieldType = fieldType;
			}
			if (!WaitForVBL(fieldType))
				break;
			
			ULWord vblCount;
			Client->Device->GetOutputVerticalInterruptCount(vblCount, Channel);
			
			if (vblCount != lastVBLCount)
			{
				DropCount += vblCount - lastVBLCount;
				framesSinceLastDrop = 0;
			}
			else
			{
				++framesSinceLastDrop;
				if (DropCount && framesSinceLastDrop == 50)
				{
					mzEngine.LogE("AJAOut: Dropped frames, notifying restart");
					NotifyRestart({});
				}
			}

			auto [Buf, Pitch, Segments, FrameIndex] = GetDMAInfo(sourceCpuRingSlot->Res, doubleBufferIndex);
			if (Interlaced())
			{
				u64 addr, discardLength;
				auto fieldId = GetAJAFieldID(fieldType);
				Client->Device->DMAWriteSegments(FrameIndex, Buf, fieldId * Pitch, Pitch, Segments, Pitch, Pitch * 2);
			}
			else
				Client->Device->DMAWriteFrame(FrameIndex, Buf, Pitch * Segments, Channel);

			CpuRing->EndPop(sourceCpuRingSlot);
			mzEngine.HandleEvent(hungerSignal);

			if (!Interlaced())
			{
				SetFrame(doubleBufferIndex);
				doubleBufferIndex ^= 1;
			}
		}
		else
		{
			mzEngine.LogW((Name().AsString() + " dropped 1 frame").c_str(), "");
		}
	}
	GpuRing->Stop();
	CpuRing->Stop();

	mzEngine.HandleEvent(CreateAppEvent(fbb, 
		mz::app::CreateScheduleRequest(fbb, mz::app::ScheduleRequestKind::PIN, &id, true)));

	if (Run)
		SendDeleteRequest();
}

void CopyThread::SendDeleteRequest()
{
	flatbuffers::FlatBufferBuilder fbb;
	auto ids = Client->GeneratePinIDSet(mz::Name(Name()), Mode);
	mzEngine.HandleEvent(
		CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &Client->Mapping.NodeId, ClearFlags::NONE, &ids)));
}


void CopyThread::ChangePinResolution(mzVec2u res)
{
	fb::TTexture tex;
	tex.width = res.x;
	tex.height = res.y;
	tex.unscaled = true;
	tex.unmanaged = !IsInput();
	tex.format = mz::fb::Format::R16G16B16A16_UNORM;
	flatbuffers::FlatBufferBuilder fbb;
	fbb.Finish(fb::CreateTexture(fbb, &tex));
	auto val = fbb.Release();
	mzEngine.SetPinValueByName(Client->Mapping.NodeId, PinName, {val.data(), val.size()} );
}

void CopyThread::InputConversionThread::Consume(CopyThread::Parameters const& params)
{
	auto* sourceCPUSlot = params.CR->BeginPop();
	if (!sourceCPUSlot)
		return;
	*params.TransferInProgress = true;
	auto* destGPURes = params.GR->BeginPush();
	if (!destGPURes)
	{
		params.CR->EndPop(sourceCPUSlot);
		return;
	}

	std::vector<mzShaderBinding> inputs;

	glm::mat4 colorspace = params.Colorspace; // glm::inverse(Cpy->GetMatrix<f64>());

	uint32_t iflags = (params.Interlaced() ? u32(params.FieldType) : 0) | (params.Shader == ShaderType::Comp10) << 2;

	inputs.emplace_back(ShaderBinding(MZN_Colorspace, colorspace));
	inputs.emplace_back(ShaderBinding(MZN_Source, params.CompressedTex->Res));
	inputs.emplace_back(ShaderBinding(MZN_Interlaced, iflags));
	inputs.emplace_back(ShaderBinding(MZN_ssbo, params.SSBO->Res));

	auto MsgKey = "Input " + params.Name + " DMA";

	mzCmd cmd;
	mzEngine.Begin(&cmd);

	mzEngine.Copy(cmd, &sourceCPUSlot->Res, &params.CompressedTex->Res, params.Debug ? ("(GPUTransfer)" + MsgKey + ":" + std::to_string(params.Debug)).c_str() : 0);
	if (params.Shader != ShaderType::Frag8)
	{
		inputs.emplace_back(ShaderBinding(MZN_Output, destGPURes->Res));
		mzRunComputePassParams pass = {};
		pass.Key = MZN_AJA_YCbCr2RGB_Compute_Pass;
		pass.DispatchSize = params.DispatchSize;
		pass.Bindings = inputs.data();
		pass.BindingCount = inputs.size();
		pass.Benchmark = params.Debug;
		mzEngine.RunComputePass(cmd, &pass);
	}
	else
	{
		mzRunPassParams pass = {};
		pass.Key = MZN_AJA_YCbCr2RGB_Pass;
		pass.Output = destGPURes->Res;
		pass.Bindings = inputs.data();
		pass.BindingCount = inputs.size();
		pass.Benchmark = params.Debug;
		mzEngine.RunPass(cmd, &pass);
	}
	mzEngine.End(cmd);

	destGPURes->FrameNumber = params.FrameNumber;
	destGPURes->Res.Info.Texture.FieldType = (mzTextureFieldType)params.FieldType;
	params.GR->EndPush(destGPURes);
	*params.TransferInProgress = false;
	params.CR->EndPop(sourceCPUSlot);
}

CopyThread::ConversionThread::~ConversionThread()
{
	Stop();
}

void CopyThread::OutputConversionThread::Consume(const Parameters& params)
{
#if 0 // TODO: discard according to compatibility
	if (!(res->res.field_type & fb::FieldType(FieldIdx)))
	{
		GpuRing->EndPop(res);
		continue;
	}
#endif
	auto incoming = params.GR->BeginPop();
	if(!incoming)
		return;
	*params.TransferInProgress = true;
	auto outgoing = params.CR->BeginPush();
	if (!outgoing)
	{
		params.GR->EndPop(incoming);
		return;
	}
	outgoing->Res.Info.Texture.FieldType = params.FieldType;
	glm::mat4 colorspace = params.Colorspace;
	uint32_t iflags = (params.Interlaced() ? u32(params.FieldType) : 0) | ((params.Shader == ShaderType::Comp10) << 2);

	std::vector<mzShaderBinding> inputs;
	inputs.emplace_back(ShaderBinding(MZN_Colorspace, colorspace));
	inputs.emplace_back(ShaderBinding(MZN_Source, incoming->Res));
	inputs.emplace_back(ShaderBinding(MZN_Interlaced, iflags));
	inputs.emplace_back(ShaderBinding(MZN_ssbo, params.SSBO->Res));

	mzCmd cmd;
	mzEngine.Begin(&cmd);

	// watch out for th members, they are not synced
	if (params.Shader != ShaderType::Frag8)
	{
		inputs.emplace_back(ShaderBinding(MZN_Output, params.CompressedTex->Res));
		mzRunComputePassParams pass = {};
		pass.Key = MZN_AJA_RGB2YCbCr_Compute_Pass;
		pass.DispatchSize = params.DispatchSize;
		pass.Bindings = inputs.data();
		pass.BindingCount = inputs.size();
		pass.Benchmark = params.Debug;
		mzEngine.RunComputePass(cmd, &pass);
	}
	else
	{
		mzRunPassParams pass = {};
		pass.Key = MZN_AJA_RGB2YCbCr_Pass;
		pass.Output = params.CompressedTex->Res;
		pass.Bindings = inputs.data();
		pass.BindingCount = inputs.size();
		pass.Benchmark = params.Debug;
		mzEngine.RunPass(cmd, &pass);
	}

	mzEngine.Copy(cmd, &params.CompressedTex->Res, &outgoing->Res, 0);
	mzEngine.End(cmd);
	params.GR->EndPop(incoming);
	*params.TransferInProgress = false;
	params.CR->EndPush(outgoing);
}

CopyThread::CopyThread(struct AJAClient *client, u32 ringSize, u32 spareCount, mz::fb::ShowAs kind, 
					   NTV2Channel channel, NTV2VideoFormat initalFmt,
					   AJADevice::Mode mode, enum class Colorspace colorspace, enum class GammaCurve curve,
					   bool narrowRange, const fb::Texture* tex)
	: PinName(GetChannelStr(channel, mode)), Client(client), PinKind(kind), Channel(channel), SpareCount(spareCount), Mode(mode),
	  Colorspace(colorspace), GammaCurve(curve), NarrowRange(narrowRange), Format(initalFmt)
{

	{
		mzBufferInfo info = {};
		info.Size = (1<<(SSBO_SIZE)) * sizeof(u16);
		info.Usage = MZ_BUFFER_USAGE_STORAGE_BUFFER; // | MZ_BUFFER_USAGE_DEVICE_MEMORY;
		SSBO = MakeShared<CPURing::Resource>(info);
		UpdateCurve(GammaCurve);
	}

	if(IsInput())
	{
		Format = client->Device->GetInputVideoFormat(channel);
	}

	client->Device->SetRegisterWriteMode(Interlaced() ? NTV2_REGWRITE_SYNCTOFIELD : NTV2_REGWRITE_SYNCTOFRAME, Channel);

	CreateRings(ringSize);
	StartThread();
}

CopyThread::~CopyThread()
{
	Stop();
	Client->Device->CloseChannel(Channel, IsInput(), IsQuad());
}

void CopyThread::Orphan(bool orphan, std::string const& message)
{
	IsOrphan = orphan;
	PinUpdate(mz::fb::TOrphanState{.is_orphan=orphan, .message=message}, Action::NOP);
}

void CopyThread::Live(bool b)
{
	PinUpdate(std::nullopt, b ? Action::SET : Action::RESET);
}

void CopyThread::PinUpdate(std::optional<mz::fb::TOrphanState> orphan, mz::Action live)
{
	flatbuffers::FlatBufferBuilder fbb;
	auto ids = Client->GeneratePinIDSet(mz::Name(Name()), Mode);
	std::vector<flatbuffers::Offset<PartialPinUpdate>> updates;
	std::transform(ids.begin(), ids.end(), std::back_inserter(updates),
				   [&fbb, orphan, live](auto id) { return mz::CreatePartialPinUpdateDirect(fbb, &id, 0, orphan ? mz::fb::CreateOrphanState(fbb, &*orphan) : false, live); });
	mzEngine.HandleEvent(
		CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &Client->Mapping.NodeId, ClearFlags::NONE, 0, 0, 0,
															  0, 0, 0, 0, &updates)));
}

u32 CopyThread::BitWidth() const
{
	return Client->BitWidth();
}

} // namespace mz
