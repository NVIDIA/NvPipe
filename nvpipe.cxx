/*
 * Copyright (c) 2016 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual
 * property and proprietary rights in and to this software,
 * related documentation and any modifications thereto.  Any use,
 * reproduction, disclosure or distribution of this software and
 * related documentation without an express license agreement from
 * NVIDIA CORPORATION is strictly prohibited.
 *
 */
#include <cstdlib>
#include <cstring>
#include "nvpipe.h"
#include "codec/nvp-abstract.h"
#include "codec/nvp-hw.h"
#include "util/formatConversionCuda.h"

#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

struct nvpipe_ {
	enum NVPipeCodecID type_;
	NvPipeCodec* codec_ptr_;
};

struct profile_blk {
	profile_blk(const char* const s) {
		nvtxRangePushA(s);
	}
	~profile_blk() {
		nvtxRangePop();
	}
};

nvpipe*
nvpipe_create(enum NVPipeCodecID id, uint64_t bitrate)
{
	const int64_t api_bitrate = static_cast<int64_t>(bitrate);
	if(api_bitrate < 0) {
		return NULL;
	}

	nvpipe_* rv = (nvpipe_*) calloc(sizeof(nvpipe_), 1);
	rv->codec_ptr_ = new NvPipeCodec264();
	switch (id) {
	case NVPIPE_CODEC_ID_NULL:
		rv->type_ = id;
		rv->codec_ptr_ = NULL;
		break;
	case NVPIPE_CODEC_ID_H264_HARDWARE:
		{
			rv->type_ = NVPIPE_CODEC_ID_H264_HARDWARE;
			rv->codec_ptr_->setBitrate(bitrate);
			rv->codec_ptr_->setCodecImplementation(NV_CODEC);
			break;
		}
	case NVPIPE_CODEC_ID_H264_SOFTWARE:
		{
			rv->type_ = NVPIPE_CODEC_ID_H264_SOFTWARE;
			rv->codec_ptr_->setBitrate(bitrate);
			rv->codec_ptr_->setCodecImplementation(FFMPEG_LIBX);
			break;
		}
	}

	return rv;
}

nvp_err_t
nvpipe_destroy(nvpipe* const __restrict cdc)
{
	nvp_err_t result = NVPIPE_SUCCESS;

	if(cdc == NULL) {
		return NVPIPE_ERR_INVALID_NVPIPE_INSTANCE;
	}
	nvpipe_* codec = static_cast<nvpipe_*>(cdc);

	switch (codec->type_) {
	case NVPIPE_CODEC_ID_NULL:
		memset(codec, 0, sizeof(nvpipe_));
		free(codec);
		break;
	case NVPIPE_CODEC_ID_H264_HARDWARE: /* FALLTHROUGH */
	case NVPIPE_CODEC_ID_H264_SOFTWARE:
		delete codec->codec_ptr_;
		memset(codec, 0, sizeof(nvpipe_));
		free(codec);
		break;
	}

	return result;
}

nvp_err_t
nvpipe_encode(nvpipe* const __restrict cdc,
              void* const __restrict input_buffer,
              const size_t input_buffer_size,
              void* const __restrict output_buffer,
              size_t* const __restrict output_buffer_size,
              const int width, const int height, enum NVPipeImageFormat format)
{
	nvpipe_* codec = static_cast<nvpipe_*>(cdc);

	/* input images must be a power of two */
	if(((width | height) & 1) != 0) {
		return NVPIPE_ERR_INVALID_RESOLUTION;
	}

	if(codec == NULL) {
		return NVPIPE_ERR_INVALID_NVPIPE_INSTANCE;
	}

	NvPipeCodec* codec_ptr = static_cast<NvPipeCodec*>(codec->codec_ptr_);

	profile_blk enc_("encodingSession");

	codec_ptr->setImageSize(width, height);
	codec_ptr->setInputFrameBuffer(input_buffer, input_buffer_size);
	return codec_ptr->encode(output_buffer, *output_buffer_size, format);
}

nvp_err_t
nvpipe_decode(nvpipe* const __restrict cdc,
              const void* const __restrict input_buffer,
              const size_t input_buffer_size,
              void* const __restrict output_buffer,
              size_t output_buffer_size,
              size_t* const __restrict width,
              size_t* const __restrict height, enum NVPipeImageFormat format)
{
	/* input images must be a power of two */
	if(((*width | *height) & 1) != 0) {
		return NVPIPE_ERR_INVALID_RESOLUTION;
	}

	if(cdc == NULL) {
		return NVPIPE_ERR_INVALID_NVPIPE_INSTANCE;
	}

	if(input_buffer_size == 0) {
		return NVPIPE_ERR_INPUT_BUFFER_EMPTY_MEMORY;
	}
	nvpipe_* codec = static_cast<nvpipe_*>(cdc);

	NvPipeCodec *codec_ptr = static_cast<NvPipeCodec*>(codec->codec_ptr_);

	profile_blk decodesess("decodingSession");

	codec_ptr->setImageSize(*width, *height);
	codec_ptr->setInputPacketBuffer(input_buffer, input_buffer_size);
	return codec_ptr->decode(output_buffer,
	                         *width, *height, output_buffer_size, format);
}
