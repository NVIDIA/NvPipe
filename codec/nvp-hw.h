/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef NVPIPE_CODEC_264_H_
#define NVPIPE_CODEC_264_H_

#ifdef __cplusplus
extern "C"
{
#endif
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#ifdef __cplusplus
}
#endif

#include <string>
#include "codec/nvp-abstract.h"

/*! \brief H.264 Encoder/Decoder
 *
 *  Implementation using ffmpeg libx264 and nvenc/cuvid
 *
 *  Encoder takes image as input and generates packets as output. While
 *  Decoder works the opposite.
 *
 */
class NvPipeCodec264 : public NvPipeCodec {
public:
	virtual nvp_err_t encode(void* buffer, size_t& size, nvp_fmt_t format);

	virtual nvp_err_t decode(void* picture, size_t& width, size_t& height,
	                         size_t& size, nvp_fmt_t format);

	virtual void setImageSize(size_t width, size_t height);
	virtual void setBitrate(int64_t bitrate);

	virtual void setInputFrameBuffer(const void* frame_buffer,
	                                 size_t buffer_size);

	NvPipeCodec264();
	virtual ~NvPipeCodec264();

protected:
	AVCodecContext *encoder_context_;
	AVCodec *encoder_codec_;
	AVFrame *encoder_frame_;
	AVPacket encoder_packet_;

	AVCodecContext *decoder_context_;
	AVCodec *decoder_codec_;
	AVFrame *decoder_frame_;
	AVPacket decoder_packet_;

private:
	nvpipeMemGPU mem_gpu_;

	nvp_err_t getFormatConversionEnum(nvp_fmt_t format, bool encoder_flag,
                                    enum NVPipeImageFormatConversion& convflag,
	                                  enum AVPixelFormat& pixel_format);

	// append 2 dummy access delimiter NAL units at the end.
	void appendDummyNAL(void* buffer, size_t offset);

	bool encoder_config_dirty_;
	bool decoder_config_dirty_;

	bool encoder_frame_buffer_dirty_;

	enum NVPipeImageFormatConversion encoder_conversion_flag_;
	void* encoder_converted_image_buffer_;
	size_t encoder_converted_image_buffer_size_;

	enum NVPipeImageFormatConversion decoder_conversion_flag_;

	enum AVPixelFormat encoder_frame_pixel_format_;

	std::string getEncoderName() const;
	std::string getDecoderName() const;

	nvp_err_t configureEncoderContext();
};
#endif //NVPIPE_CODEC_264_H_
