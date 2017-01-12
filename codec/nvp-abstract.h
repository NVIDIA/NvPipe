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
#ifndef NVPIPE_CODEC_H_
#define NVPIPE_CODEC_H_
#include "util/format.h"
#include <cstdlib>

/*! \brief enum for codec implementation
 * 
 * FFMPEG_LIBX is utilizing the libx264 through ffmpeg
 * NV_CODEC is utilizing nvenc/cuvid through ffmpeg
 */
enum NvPipeCodecImplementation {
    FFMPEG_LIBX,
    NV_CODEC
};

/*! \brief Abstract Encoder/Decoder
 *
 *  Abstract NvPipe codec. Defines common interface to codecs.
 */
class NvPipeCodec {
public:
	NvPipeCodec();
	virtual ~NvPipeCodec();

	virtual void setImageSize(size_t width, size_t height);

	virtual void setCodecImplementation(enum NvPipeCodecImplementation type);
	virtual void setInputFrameBuffer(const void* fb, size_t size);
	virtual void setInputPacketBuffer(const void* pb, size_t size);

	virtual nvp_err_t encode(void* frame, size_t& size, nvp_fmt_t format)=0;
	virtual nvp_err_t decode(void* packet, size_t& width, size_t& height,
	                         size_t& size, nvp_fmt_t format)=0; 

	virtual void setBitrate(int64_t bitrate);
	void setGopSize(int gop_size);
	void setFramerate(int framerate);

protected:
	size_t width_;
	size_t height_;

	const void* frame_;
	size_t frame_buffer_size_;
	nvp_fmt_t encoder_format_;
	const void* packet_;
	size_t packet_buffer_size_;

	nvp_fmt_t decoder_format_;

	bool bitrate_overwrite_flag_;
	int64_t bitrate_;

	int gop_size_;

	int framerate_;

	enum NvPipeCodecImplementation codec_;

	enum NvPipeCodecImplementation getCodec() const;
};
#endif //NVPIPE_CODEC_H_
