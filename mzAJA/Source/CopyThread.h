#pragma once

#include "AJAClient.h"
#include "glm/glm.hpp"
#include "Ring.h"
#include <mzUtil/Thread.h>

namespace mz
{

using Clock = std::chrono::high_resolution_clock;
using Milli = std::chrono::duration<double, std::milli>;
using Micro = std::chrono::duration<double, std::micro>;

// almost all scheduler thread
struct TrackSync 
{
    // worker thread access
	std::mutex DroppedFramesMutex;
	std::vector<u32> DroppedFrames;
    std::atomic<u32> FrameIDCounter = 0; 
    // worker thread access

	u32 LastFrameID = 0;
	u32 FramesSinceLastDrop = 0;
	
    bool ResetTrackNow = false;
	bool ResetTrackWhenStable = false;


    virtual u32 GetRingSize() = 0;
    virtual mz::Name const& Name() const = 0;

	void UpdateLastFrameID() // called from scheduler thread
	{
		ResetTrackNow = true;
		ResetTrackWhenStable = false;
		
        FramesSinceLastDrop = 0;
		LastFrameID = FrameIDCounter.load() - GetRingSize();

		//if (true)
		{
			mzEngine.LogI("Resetting AJA %s", Name().AsCStr());
		}
	}
	void UpdateDropCount() // called from scheduler thread
	{
		std::vector<u32> dropCounts;
		DequeueDroppedFrames(dropCounts);
		ResetTrackNow = false;
		if (!dropCounts.empty())
		{
			ResetTrackWhenStable = true;
			u32 totalDrops = 0;
			for (const auto& dropCount : dropCounts)
			{
				totalDrops += dropCount;
			}

			mzEngine.LogI("Drop on %s: %ul", Name().AsCStr(), totalDrops);
		}
		else
		{
			FramesSinceLastDrop++;
			if (ResetTrackWhenStable && FramesSinceLastDrop > 50)
			{
				UpdateLastFrameID();
			}
		}
	}

	void DequeueDroppedFrames(std::vector<u32>& dropCounts) // called from scheduler thread
	{
		std::lock_guard<std::mutex> ScopedLock(DroppedFramesMutex);
		dropCounts = std::move(DroppedFrames);
	}

	void EnqueueDroppedFrames(u32 dropCount) // called from worker thread
	{
		std::lock_guard<std::mutex> ScopedLock(DroppedFramesMutex);
        DroppedFrames.push_back(dropCount);
	}

};

inline mzTextureFieldType Flipped(mzTextureFieldType field)
{
	return field == MZ_TEXTURE_FIELD_TYPE_PROGRESSIVE || field == MZ_TEXTURE_FIELD_TYPE_UNKNOWN ? field 
		: (field == MZ_TEXTURE_FIELD_TYPE_EVEN ? MZ_TEXTURE_FIELD_TYPE_ODD : MZ_TEXTURE_FIELD_TYPE_EVEN);
}

struct CopyThread : TrackSync
{
	mz::Name PinName;
	u32 ConnectedPinCount = 0;
    std::atomic_bool Run = true;
    mz::fb::ShowAs PinKind;
    rc<GPURing> GpuRing;
	
    rc<CPURing> CpuRing;
	std::atomic_bool TransferInProgress = false; // TODO: Combine these rings into a double ring structure
	// The ring objects above are overwritten on path restart.
	// TODO: Find out other synchronization issues and fix them all
    std::atomic_uint32_t SpareCount = 0;
    std::thread Thread;
    NTV2Channel Channel;
    u32 DropCount = 0;
    struct AJAClient *Client = 0;
    NTV2VideoFormat Format = NTV2_FORMAT_UNKNOWN;
    AJADevice::Mode Mode = AJADevice::SL;
    std::atomic<Colorspace> Colorspace = Colorspace::REC709;
    std::atomic<GammaCurve> GammaCurve = GammaCurve::REC709;
    std::atomic_bool NarrowRange = true;
	std::atomic_bool IsOrphan = false;
    
	rc<CPURing::Resource> SSBO;
	rc<GPURing::Resource> CompressedTex;

	struct
	{
		std::chrono::nanoseconds Time;
		u32 Counter;
	} DebugInfo;

	struct Parameters
	{
		mzTextureFieldType FieldType = MZ_TEXTURE_FIELD_TYPE_PROGRESSIVE;
		u32 FrameNumber;
		Clock::time_point T0;
		Clock::time_point T1;
		rc<GPURing> GR;
		rc<CPURing> CR;
		glm::mat4 Colorspace;
		ShaderType Shader;
		rc<GPURing::Resource> CompressedTex;
		rc<CPURing::Resource> SSBO;
		std::string Name;
		uint32_t Debug = 0;
		mzVec2u DispatchSize;
		std::atomic_bool* TransferInProgress = 0;

		bool Interlaced() const { return !(FieldType == MZ_TEXTURE_FIELD_TYPE_PROGRESSIVE || FieldType == MZ_TEXTURE_FIELD_TYPE_UNKNOWN); }
	};

	struct ConversionThread : ConsumerThread<Parameters>
	{
		ConversionThread(CopyThread* parent) : Parent(parent) {}
		virtual ~ConversionThread();
		std::thread Handle;
		CopyThread* Parent;
	};
	struct InputConversionThread : ConversionThread
	{
		using ConversionThread::ConversionThread;
		void Consume(const Parameters& item) override;
	};
	struct OutputConversionThread : ConversionThread
	{
		using ConversionThread::ConversionThread;
		void Consume(const Parameters& item) override;
	};

	ru<ConversionThread> Worker;

    CopyThread(struct AJAClient *client, u32 ringSize, u32 spareCount, mz::fb::ShowAs kind, NTV2Channel channel, 
                NTV2VideoFormat initalFmt, 
                AJADevice::Mode mode,
                enum class Colorspace colorspace, enum class GammaCurve curve, bool narrowRange, const fb::Texture* tex);

    virtual u32 GetRingSize() override;
    virtual mz::Name const& Name() const override;
	mzVec2u GetSuitableDispatchSize() const;
	mzVec2u Extent() const;
    bool IsInput() const;
    void AJAInputProc();
    void AJAOutputProc();
    void CreateRings(u32 size);
    void SendDeleteRequest();
    void ChangePinResolution(mzVec2u res);
    void InputUpdate(AJADevice::Mode &prevMode);
	bool WaitForVBL(mzTextureFieldType fieldType);
	
	void Refresh();
    bool IsQuad() const;
    bool Interlaced() const;
	bool LinkSizeMismatch() const;

    void StartThread();
    void Orphan(bool, std::string const& msg = "");
    void Live(bool);
    void PinUpdate(std::optional<mz::fb::TOrphanState>, Action live);

    void Stop();
	void Restart(u32 ringSize);
    void SetFrame(u32 doubleBufferIndex);
	u32 GetFrameIndex(u32 doubleBufferIndex) const;

	struct DMAInfo
	{
		u32* Buffer;
		u32 Pitch;
		u32 Segments;
		u32 FrameIndex;
	};

	DMAInfo GetDMAInfo(mzResourceShareInfo& buffer, u32 doubleBufferIndex) const;

    ~CopyThread();

    u32 BitWidth() const;

    void UpdateCurve(enum GammaCurve curve);

    std::array<f64, 2> GetCoeffs() const;

    template<class T>
    glm::mat<4,4,T> GetMatrix() const;

	void NotifyRestart(RestartParams const& params);
	void NotifyDrop();

	u32 TotalFrameCount();

	void SendRingStats();
};

} // namespace mz