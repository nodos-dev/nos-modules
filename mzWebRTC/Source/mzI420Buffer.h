#pragma once
#include "api/video/i420_buffer.h"
#include "api/video/video_frame.h"
class mzI420Buffer : public webrtc::I420BufferInterface
{
public:
	mzI420Buffer(unsigned int width, unsigned int height, uint8_t*& dataY, uint8_t*& dataU, uint8_t*& dataV)
	:m_width(width), m_height(height), m_dataY(dataY), m_dataU(dataU), m_dataV(dataV) {
		m_strideY = width;
		m_strideU = (width + 1) / 2;
		m_strideV = (width + 1) / 2;
	}
	virtual ~mzI420Buffer() = default;
	virtual int width() const override { return m_width; }
	virtual int height() const override { return m_height; }
	
	virtual const uint8_t* DataY() const override { return m_dataY; }
	virtual const uint8_t* DataU() const override { return m_dataU; }
	virtual const uint8_t* DataV() const override { return m_dataV; }
	
	virtual int StrideY() const override { return m_strideY; }
	virtual int StrideU() const override { return m_strideU; }
	virtual int StrideV() const override { return m_strideV; }

	void AddRef() const override { ref_count_.IncRef(); }

	rtc::RefCountReleaseStatus Release() const override {
		const auto status = ref_count_.DecRef();
		if (status == rtc::RefCountReleaseStatus::kDroppedLastRef) {
			delete this;
		}
		return status;
	}
private:
	unsigned int m_width;
	unsigned int m_height;
	uint8_t* m_dataY;
	uint8_t* m_dataU;
	uint8_t* m_dataV;
	int m_strideY;
	int m_strideU;
	int m_strideV;
	mutable webrtc::webrtc_impl::RefCounter ref_count_{ 0 };

};