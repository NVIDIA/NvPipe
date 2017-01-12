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
#include "codec/nvp-abstract.h"

NvPipeCodec::NvPipeCodec() {
	width_ = 0;
	height_ = 0;
	encoder_format_ = NVPIPE_RGB;
	decoder_format_ = NVPIPE_RGB;

	frame_ = NULL;
	frame_buffer_size_ = 0;
	packet_ = NULL;
	packet_buffer_size_ = 0;

	bitrate_ = 1000000;
	framerate_ = 30;
	gop_size_ = 60;

	bitrate_overwrite_flag_ = false;

	codec_ = NV_CODEC;
}

NvPipeCodec::~NvPipeCodec() {
}

void
NvPipeCodec::setImageSize(size_t width, size_t height) {
	width_ = width;
	height_ = height;
}

void
NvPipeCodec::setInputPacketBuffer(const void *buf, size_t bufsz) {
	packet_ = buf;
	packet_buffer_size_ = bufsz;
}

void
NvPipeCodec::setInputFrameBuffer(const void *frame_buffer, size_t buffer_size) {
	frame_ = frame_buffer;
	frame_buffer_size_ = buffer_size;
}

void
NvPipeCodec::setBitrate(int64_t bitrate) {
	if(bitrate != 0) {
		bitrate_overwrite_flag_ = true;
		bitrate_ = bitrate;
	} else {
		bitrate_overwrite_flag_ = false;
	}
}

void
NvPipeCodec::setGopSize(int gop_size) {
	gop_size_ = gop_size;
}

void
NvPipeCodec::setFramerate(int framerate) {
	framerate_ = framerate;
}

enum NvPipeCodecImplementation
NvPipeCodec::getCodec() const {
	return codec_;
}

void
NvPipeCodec::setCodecImplementation(enum NvPipeCodecImplementation type) {
	codec_ = type;
}
