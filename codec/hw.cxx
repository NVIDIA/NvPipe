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
#include "codec/nvp-hw.h"
#include <cstdlib>
#include <cmath>
#include <libavutil/log.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include "debug.h"
#include "util/formatConversionCuda.h"

#define AVFRAME_LINESIZE_ALIGNMENT 32

DECLARE_CHANNEL(ff);
static const char* latency_errstr =
	"Output from FFMpeg is not ready.  This cannot happen if you "
	"have modified your FFMpeg source to disable frame latency.  You "
	"may have linked against the wrong version of FFMpeg.";

NvPipeCodec264::NvPipeCodec264() {
	encoder_context_ = NULL;
	encoder_codec_ = NULL;
	encoder_frame_ = NULL;

	decoder_context_ = NULL;
	decoder_codec_ = NULL;
	decoder_frame_ = NULL;

	encoder_frame_pixel_format_ = AV_PIX_FMT_RGB24;

	encoder_config_dirty_ = true;
	decoder_config_dirty_ = true;

	encoder_frame_buffer_dirty_ = true;

	encoder_conversion_flag_ = NVPIPE_IMAGE_FORMAT_CONVERSION_NULL;
	decoder_conversion_flag_ = NVPIPE_IMAGE_FORMAT_CONVERSION_NULL;

	encoder_converted_image_buffer_ = NULL;
	encoder_converted_image_buffer_size_ = 0;

	if(getenv("NVPIPE_VERBOSE") != NULL) {
		av_log_set_level(AV_LOG_TRACE);
	}
	// register all available file formats and codecs
	// could be initialized multiple times.
	av_register_all();

	initializeMemGPU(&mem_gpu_);
}

NvPipeCodec264::~NvPipeCodec264() {
	avcodec_close(decoder_context_);
	av_free(decoder_context_);
	av_frame_free(&decoder_frame_);
	free(encoder_converted_image_buffer_);
	avcodec_close(encoder_context_);
	av_free(encoder_context_);
	av_frame_free(&encoder_frame_);
	destroyMemGPU(&mem_gpu_);
}

void
NvPipeCodec264::setImageSize(size_t width, size_t height) {
	if(width != width_ || height != height_) {
		encoder_config_dirty_ = true;
		decoder_config_dirty_ = true;
	}
	NvPipeCodec::setImageSize(width, height);
}

void
NvPipeCodec264::setInputFrameBuffer(const void *frame_buffer,
                                    size_t buffer_size) {
	if(encoder_conversion_flag_ == NVPIPE_IMAGE_FORMAT_CONVERSION_NULL
		 && frame_buffer != frame_) {
		encoder_frame_buffer_dirty_ = true;
	}
	NvPipeCodec::setInputFrameBuffer(frame_buffer, buffer_size);
}

void
NvPipeCodec264::setBitrate(int64_t bitrate) {
	NvPipeCodec::setBitrate(bitrate);
	encoder_config_dirty_ = true;

	if(encoder_context_ != NULL) {
		encoder_context_->bit_rate = bitrate;
	}
}

nvp_err_t
NvPipeCodec264::encode(void *buffer,
                       size_t& output_buffer_size, nvp_fmt_t format) {
	if(width_ == 0 || height_ == 0) {
		return NVPIPE_EINVAL;
	}
	nvp_err_t nerr = NVPIPE_SUCCESS;

	if(format != encoder_format_) {
		nerr = getFormatConversionEnum(format, true, encoder_conversion_flag_,
		                               encoder_frame_pixel_format_);
		if(nerr != NVPIPE_SUCCESS) {
			return nerr;
		}
		encoder_config_dirty_ = true;
		encoder_format_ = format;
	}
	// Check if encoder codec has been initialized
	if(encoder_codec_ == NULL) {
		encoder_codec_ = avcodec_find_encoder_by_name(getEncoderName().c_str());
		if(encoder_codec_ == NULL) {
			ERR(ff, "avcodec cannot find '%s' encoder", getEncoderName().c_str());
			return NVPIPE_ENOENT;
		}

		encoder_frame_ = av_frame_alloc();
		if(encoder_frame_ == NULL) {
			return NVPIPE_ENOMEM;
		}

		nvp_err_t res;
		if((res = configureEncoderContext()) != NVPIPE_SUCCESS) {
			return res;
		}

		encoder_config_dirty_ = false;
		encoder_frame_buffer_dirty_ = true;
	}
	// Check if encoder frame parameter has been updated
	if(encoder_config_dirty_) {
		avcodec_close(encoder_context_);
		av_free(encoder_context_);

		nvp_err_t res;
		if((res = configureEncoderContext()) != NVPIPE_SUCCESS) {
			return res;
		}

		encoder_config_dirty_ = false;
		encoder_frame_buffer_dirty_ = true;
	}

	if(encoder_frame_buffer_dirty_) {
		const uint8_t *frame_image_ptr;

		encoder_frame_->format = encoder_frame_pixel_format_;
		encoder_frame_->width = width_;
		encoder_frame_->height = height_;

		float linesize = width_;
		linesize = std::ceil(linesize / AVFRAME_LINESIZE_ALIGNMENT)
			* AVFRAME_LINESIZE_ALIGNMENT;
		size_t num_pixels = linesize;
		num_pixels *= height_;

		switch (encoder_conversion_flag_) {
		case NVPIPE_IMAGE_FORMAT_CONVERSION_RGBA_TO_NV12:
		case NVPIPE_IMAGE_FORMAT_CONVERSION_RGB_TO_NV12:
			if(encoder_converted_image_buffer_size_ < num_pixels * 3 / 2) {
				free(encoder_converted_image_buffer_);
				encoder_converted_image_buffer_size_ = num_pixels * 3 / 2;
				encoder_converted_image_buffer_ = (void *)
					malloc(sizeof(uint8_t) * encoder_converted_image_buffer_size_);
			}
			frame_image_ptr = (const uint8_t *)encoder_converted_image_buffer_;
			break;
		default:
			frame_image_ptr = (const uint8_t *)frame_;
			break;
		}

		// setup input data buffer to encoder_frame_
		// Note the allocation of data buffer is done by user
		if(av_image_fill_arrays(encoder_frame_->data,
														encoder_frame_->linesize,
														frame_image_ptr,
														encoder_frame_pixel_format_,
														width_, height_, AVFRAME_LINESIZE_ALIGNMENT) < 0) {
			ERR(ff, "av_image_fill_arrays failed");
			return NVPIPE_EMAP;
		}
		encoder_frame_buffer_dirty_ = false;
	}

	nvtxRangePushA("encodingFormatConversionSession");

	switch (encoder_conversion_flag_) {
	case NVPIPE_IMAGE_FORMAT_CONVERSION_RGB_TO_NV12:
	case NVPIPE_IMAGE_FORMAT_CONVERSION_RGBA_TO_NV12:
		formatConversionReuseMemory(width_, height_,
																AVFRAME_LINESIZE_ALIGNMENT,
																frame_,
																encoder_converted_image_buffer_,
																encoder_conversion_flag_, &mem_gpu_);
		break;
	default:
		break;
	}

	nvtxRangePop();

	nvtxRangePushA("encodingFfmpegAPISession");

	av_init_packet(&encoder_packet_);
	encoder_packet_.data = NULL;
	encoder_packet_.size = 0;

	int ret = avcodec_send_frame(encoder_context_, encoder_frame_);
	if(ret < 0) {
		ERR(ff, "send_frame: %d", ret);
		return NVPIPE_EENCODE;
	}

	ret = avcodec_receive_packet(encoder_context_, &encoder_packet_);
	if(ret < 0) {
		if(ret == AVERROR(EAGAIN)) {
			ERR(ff, "%s", latency_errstr);
			return NVPIPE_EAGAIN;
		}
		ERR(ff, "receive_packet: %d", ret);
		av_packet_unref(&encoder_packet_);
		output_buffer_size = 0;
		return NVPIPE_EENCODE;
	}

	unsigned int packet_size = encoder_packet_.size > 0 ?
		encoder_packet_.size + 10 : 10;
	if(packet_size >= output_buffer_size) {
		return NVPIPE_EOVERFLOW;
	}
	memcpy(buffer, encoder_packet_.data, encoder_packet_.size);
	appendDummyNAL(buffer, encoder_packet_.size);

	// output the packet size;
	output_buffer_size = encoder_packet_.size + 10;
	av_packet_unref(&encoder_packet_);

	nvtxRangePop();

	return NVPIPE_SUCCESS;
}

nvp_err_t
NvPipeCodec264::decode(void* output_picture,
                       size_t& width, size_t& height,
                       size_t& output_size, nvp_fmt_t format) {
	enum AVPixelFormat pixel_format;

	// Check if decoder codec has been initialized
	if(decoder_codec_ == NULL) {
		decoder_codec_ = avcodec_find_decoder_by_name(getDecoderName().c_str());
		if(decoder_codec_ == NULL) {
			ERR(ff, "avcodec cannot find '%s' decoder", getDecoderName().c_str());
			return NVPIPE_ENOENT;
		}

		decoder_context_ = avcodec_alloc_context3(decoder_codec_);
		if(decoder_context_ == NULL) {
			return NVPIPE_ENOMEM;
		}

		decoder_frame_ = av_frame_alloc();
		if(decoder_frame_ == NULL) {
			return NVPIPE_ENOMEM;
		}

		decoder_context_->delay = 0;
		if(avcodec_open2(decoder_context_, decoder_codec_, NULL) < 0) {
			ERR(ff, "avcodec_open2 failed");
			return NVPIPE_ENOENT;
		}
		decoder_config_dirty_ = false;
	}

	nvp_err_t result = getFormatConversionEnum(format, false,
	                                           decoder_conversion_flag_,
	                                           pixel_format);
	if(result != 0) {
		return result;
	}

	if(decoder_config_dirty_) {
		avcodec_close(decoder_context_);
		if(avcodec_open2(decoder_context_, decoder_codec_, NULL) < 0) {
			ERR(ff, "avcodec_open2 failed");
			return NVPIPE_ENOENT;
		}
		decoder_config_dirty_ = false;
	}

	nvtxRangePushA("decodingFfmpegAPISession");

	av_init_packet(&decoder_packet_);
	decoder_packet_.data = (uint8_t *) packet_;
	decoder_packet_.size = packet_buffer_size_;

	int snd = 0;
	if((snd = avcodec_send_packet(decoder_context_, &decoder_packet_)) != 0) {
		ERR(ff, "send_packet: %d", snd);
		return NVPIPE_EENCODE;
	}

	int ret = avcodec_receive_frame(decoder_context_,
																	decoder_frame_);
	if(ret < 0 && ret == AVERROR(EAGAIN)) {
		ERR(ff, "%s", latency_errstr);
		return NVPIPE_EAGAIN;
	}
	if(ret < 0) {
		ERR(ff, "receive_frame: %d", ret);
		av_packet_unref(&decoder_packet_);
		output_size = 0;
		width = 0;
		height = 0;
		return NVPIPE_EDECODE;
	}

	nvtxRangePop();

	width = decoder_frame_->width;
	height = decoder_frame_->height;

	nvtxRangePushA("decodingFormatConversionSession");

	// should really check the decoder_frame_->format
	switch (decoder_conversion_flag_) {
	case NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGBA:
		{
			size_t frameSize = width * height * 4;
			if(frameSize > output_size) {
				output_size = frameSize;
				av_packet_unref(&decoder_packet_);
				return NVPIPE_EOVERFLOW;
			}
			output_size = frameSize;
			formatConversionAVFrameRGBAReuseMemory(decoder_frame_,
																						 AVFRAME_LINESIZE_ALIGNMENT,
																						 output_picture, &mem_gpu_);
			av_packet_unref(&decoder_packet_);
			nvtxRangePop();

			return NVPIPE_SUCCESS;
			break;
		}
	case NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGB:
		{
			size_t frameSize = width * height * 3;
			if(frameSize > output_size) {
				output_size = frameSize;
				av_packet_unref(&decoder_packet_);
				return NVPIPE_EOVERFLOW;
			}
			output_size = frameSize;
			formatConversionAVFrameRGBReuseMemory(decoder_frame_,
																						AVFRAME_LINESIZE_ALIGNMENT,
																						output_picture, &mem_gpu_);
			av_packet_unref(&decoder_packet_);
			nvtxRangePop();

			return NVPIPE_SUCCESS;
			break;
		}
	default:
		{
			size_t frameSize = 0;
			for(int i = 0; i < AV_NUM_DATA_POINTERS; i++) {
				if(decoder_frame_->linesize[i] > 0) {
					frameSize += decoder_frame_->height * decoder_frame_->linesize[i];
				}
			}

			if(frameSize > output_size) {
				output_size = frameSize;
				av_packet_unref(&decoder_packet_);
				return NVPIPE_EOVERFLOW;
			}
			output_size = frameSize;

			frameSize = 0;
			char *output_buffer_ptr = (char *)output_picture;
			for(int i = 0; i < AV_NUM_DATA_POINTERS; i++) {
				if(decoder_frame_->linesize[i] > 0) {
					frameSize = decoder_frame_->height * decoder_frame_->linesize[i];
					memcpy(output_buffer_ptr, decoder_frame_->data[i], frameSize);
					output_buffer_ptr += frameSize;
				}
			}

			av_packet_unref(&decoder_packet_);
			return NVPIPE_SUCCESS;
		}
	}
}

nvp_err_t
NvPipeCodec264::getFormatConversionEnum(nvp_fmt_t format,
                                        bool encoder_flag,
                                        enum NVPipeImageFormatConversion&
                                          conversion_flag,
                                        enum AVPixelFormat& pixel_format) {
	switch (format) {
	case NVPIPE_RGBA:
		pixel_format = AV_PIX_FMT_NV12;
		conversion_flag = encoder_flag ?
			NVPIPE_IMAGE_FORMAT_CONVERSION_RGBA_TO_NV12 :
			NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGBA;
		break;
	case NVPIPE_RGB:
		pixel_format = AV_PIX_FMT_NV12;
		conversion_flag = encoder_flag ?
			NVPIPE_IMAGE_FORMAT_CONVERSION_RGB_TO_NV12 :
			NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGB;
		break;
	}
	return NVPIPE_SUCCESS;
}

void
NvPipeCodec264::appendDummyNAL(void *buffer, size_t offset) {
	uint8_t* pkt_ptr = (uint8_t*)buffer;
	pkt_ptr += offset;
	pkt_ptr[0] = 0;
	pkt_ptr[1] = 0;
	pkt_ptr[2] = 1;
	pkt_ptr[3] = 9;
	pkt_ptr[4] = 0;
	pkt_ptr[5] = 0;
	pkt_ptr[6] = 0;
	pkt_ptr[7] = 1;
	pkt_ptr[8] = 9;
	pkt_ptr[9] = 0;
}

nvp_err_t
NvPipeCodec264::configureEncoderContext() {
	encoder_context_ = avcodec_alloc_context3(encoder_codec_);
	if(encoder_context_ == NULL) {
		return NVPIPE_ENOMEM;
	}
	/*
	 * setup codecContext
	 * Default low latency setup for nvenc
	 */
	if(!bitrate_overwrite_flag_) {
		bitrate_ = width_ * height_ * framerate_ * 4.0 * 0.07;
	}
	encoder_context_->bit_rate = bitrate_;
	// frames per second
	encoder_context_->time_base = (AVRational) { 1, framerate_};
	encoder_context_->gop_size = gop_size_;
	encoder_context_->max_b_frames = 0;
	encoder_context_->width = width_;
	encoder_context_->height = height_;
	encoder_context_->pix_fmt = encoder_frame_pixel_format_;
	// nvenc private setting
	switch (getCodec()) {
	case NV_CODEC:
		av_opt_set(encoder_context_->priv_data, "preset", "llhq", 0);
		av_opt_set(encoder_context_->priv_data, "rc", "ll_2pass_quality", 0);
		av_opt_set_int(encoder_context_->priv_data, "cbr", 1, 0);
		av_opt_set_int(encoder_context_->priv_data, "2pass", 1, 0);
		av_opt_set_int(encoder_context_->priv_data, "delay", 0, 0);
		break;
	case FFMPEG_LIBX:
		av_opt_set(encoder_context_->priv_data, "tune", "zerolatency", 0);
		break;
	}
	if(avcodec_open2(encoder_context_, encoder_codec_, NULL)
		 != 0) {
		ERR(ff, "avcodec_open2 failed");
		return NVPIPE_ENOENT;
	}

	return NVPIPE_SUCCESS;
}

std::string
NvPipeCodec264::getEncoderName() const {
	switch (getCodec()) {
	case FFMPEG_LIBX:
		return "libx264";
		break;
	case NV_CODEC:
		return "h264_nvenc";
		break;
	default:
		break;
	}
	return "";
}

std::string
NvPipeCodec264::getDecoderName() const {
	switch (getCodec()) {
	case FFMPEG_LIBX:
		return "h264";
		break;
	case NV_CODEC:
		return "h264_cuvid";
		break;
	default:
		break;
	}
	return "";
}
