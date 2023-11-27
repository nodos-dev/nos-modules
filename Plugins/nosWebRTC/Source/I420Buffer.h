#pragma once
#include "api/video/i420_buffer.h"
#include "api/video/video_frame.h"
class nosI420Buffer : public webrtc::I420BufferInterface
{
public:
	nosI420Buffer(unsigned int width, unsigned int height)
	:m_width(width), m_height(height) {
		m_strideY = width;
		m_strideU = (width + 1) / 2;
		m_strideV = (width + 1) / 2;
	}
	virtual ~nosI420Buffer() = default;
	virtual int width() const override { return m_width; }
	virtual int height() const override { return m_height; }
	
	virtual const uint8_t* DataY() const override { return m_dataY; }
	virtual const uint8_t* DataU() const override { return (m_dataY + m_width * m_height); }
	virtual const uint8_t* DataV() const override { return (m_dataY + m_width * m_height + m_width / 2 * m_height / 2); }
	
	virtual int StrideY() const override { return m_strideY; }
	virtual int StrideU() const override { return m_strideU; }
	virtual int StrideV() const override { return m_strideV; }

	void AddRef() const override { ref_count_.IncRef(); }

	void SetDataY(uint8_t* dataY) { m_dataY = dataY; }

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
	int m_strideY;
	int m_strideU;
	int m_strideV;
	mutable webrtc::webrtc_impl::RefCounter ref_count_{ 0 };

};