/*
 * Copyright MediaZ AS. All Rights Reserved.
 */

#pragma once
#include "api/video/i420_buffer.h"
#include "api/video/video_frame.h"
class nosLinearI420Buffer : public webrtc::I420BufferInterface
{
public:
	nosLinearI420Buffer(unsigned int width, unsigned int height)
	:m_width(width), m_height(height){
		m_strideY = width;
		m_strideU = (width + 1) / 2;
		m_strideV = (width + 1) / 2;
		m_dataY = std::make_unique<uint8_t[]>(width * height * 3 / 2);
	}
	virtual ~nosLinearI420Buffer() = default;
	virtual int width() const override { return m_width; }
	virtual int height() const override { return m_height; }
	
	virtual const uint8_t* DataY() const override { return m_dataY.get(); }
	virtual const uint8_t* DataU() const override { return (m_dataY.get() + m_width * m_height); }
	virtual const uint8_t* DataV() const override { return (m_dataY.get() + m_width * m_height + m_width/2 * m_height/2); }
	
	virtual int StrideY() const override { return m_strideY; }
	virtual int StrideU() const override { return m_strideU; }
	virtual int StrideV() const override { return m_strideV; }

	uint8_t* GetY(){ return m_dataY.get(); }

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
	std::unique_ptr<uint8_t[]> m_dataY;
	int m_strideY;
	int m_strideU;
	int m_strideV;
	mutable webrtc::webrtc_impl::RefCounter ref_count_{ 0 };

};