#pragma once
#include <stdint.h>
#include "rtc_base/ssl_adapter.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "api/audio/audio_mixer.h"
#include "api/audio_codecs/audio_decoder_factory.h"
#include "api/audio_codecs/audio_encoder_factory.h"
#include "api/audio_codecs/builtin_audio_decoder_factory.h"
#include "api/audio_codecs/builtin_audio_encoder_factory.h"
#include "api/audio_options.h"
#include "api/create_peerconnection_factory.h"
#include "api/rtp_sender_interface.h"
#include "api/video_codecs/video_decoder_factory.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "defaults.h"
#include "modules/audio_device/include/audio_device.h"
#include "modules/audio_processing/include/audio_processing.h"
#include "modules/video_capture/video_capture.h"
#include "modules/video_capture/video_capture_factory.h"
#include "p2p/base/port_allocator.h"
#include "pc/video_track_source.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/rtc_certificate_generator.h"
#include "rtc_base/strings/json.h"
#include "test/vcm_capturer.h"
#include "api/video/i420_buffer.h"
#include "api/video/video_frame.h"
#include "media/base/video_broadcaster.h"
#include "media/base/video_common.h"
#include "media/base/video_source_base.h"  
#include "media/base/adapted_video_track_source.h"
#include "rtc_base/ref_counter.h"

class CustomVideoSource : public rtc::AdaptedVideoTrackSource {
public:
    CustomVideoSource();
	virtual ~CustomVideoSource() = default;

	void StartThread();
	void MaybePushFrame();
	void PushFrame(uint8_t* data, int width, int height);

	/* Begin UE::PixelStreaming::AdaptedVideoTrackSource overrides */
	virtual webrtc::MediaSourceInterface::SourceState state() const override { return CurrentState; }
	virtual bool remote() const override { return false; }
	virtual bool is_screencast() const override { return false; }
	virtual absl::optional<bool> needs_denoising() const override { return false; }
	/* End UE::PixelStreaming::AdaptedVideoTrackSource overrides */
    // Inherited via AdaptedVideoTrackSource
    void AddRef() const override;
    rtc::RefCountReleaseStatus Release() const override;

private:
	webrtc::MediaSourceInterface::SourceState CurrentState;
    rtc::Thread* thread;
    mutable webrtc::webrtc_impl::RefCounter ref_count_{0};
	std::vector<uint8_t> yPlane;
	std::vector<uint8_t> uPlane;
	std::vector<uint8_t> vPlane;


};
