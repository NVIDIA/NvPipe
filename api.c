/*
 * Copyright (c) 2016 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual
 * property and proprietary rights in and to this software,
 * related documentation and any modifications thereto.  Any use,
 * reproduction, disclosure or distribution of this software and
 * related documentation without an express license agreement from
 * NVIDIA CORPORATION is strictly prohibited.
 */
#include "config.nvp.h"
#include "debug.h"
#include "nvpipe.h"
#include "internal-api.h"

DECLARE_CHANNEL(api);

NVPIPE_VISIBLE nvpipe*
nvpipe_create_encoder(nvp_codec_t id, uint64_t bitrate) {
	switch(id) {
	case NVPIPE_H264_NV:
		return nvp_create_encoder(bitrate);
		break;
#if NVPIPE_FFMPEG == 1
	case NVPIPE_H264_NVFFMPEG:
		return nvp_create_ffmpeg(true, bitrate);
		break;
	case NVPIPE_H264_FFMPEG:
		return nvp_create_ffmpeg(false, bitrate);
		break;
#else
	case NVPIPE_H264_NVFFMPEG: /* fallthrough */
	case NVPIPE_H264_FFMPEG:
		ERR(api, "nvpipe: FFMpeg support not compiled in.");
		return NULL;
		break;
#endif
	}
	return NULL;
}

NVPIPE_VISIBLE nvpipe*
nvpipe_create_decoder(nvp_codec_t id) {
	switch(id) {
	case NVPIPE_H264_NV:
		return nvp_create_decoder();
		break;
#if NVPIPE_FFMPEG == 1
	case NVPIPE_H264_NVFFMPEG:
		return nvp_create_ffmpeg(true, 0);
		break;
	case NVPIPE_H264_FFMPEG:
		return nvp_create_ffmpeg(false, 0);
		break;
#else
	case NVPIPE_H264_NVFFMPEG: /* fallthrough */
	case NVPIPE_H264_FFMPEG:
		ERR(api, "nvpipe: FFMpeg support not compiled in.");
		return NULL;
#endif
	}
	return NULL;
}

nvp_err_t
nvpipe_encode(nvpipe* const __restrict cdc,
              const void* const __restrict ibuf,
              const size_t ibuf_sz,
              void* const __restrict obuf,
              size_t* const __restrict obuf_sz,
              const size_t width, const size_t height, nvp_fmt_t format) {
	if(cdc == NULL) {
		return NVPIPE_EINVAL;
	}
	nvp_impl_t* enc = (nvp_impl_t*)cdc;
	return enc->encode(enc, ibuf,ibuf_sz, obuf,obuf_sz, width,height, format);
}

nvp_err_t
nvpipe_decode(nvpipe* const __restrict codec,
              const void* const __restrict ibuf,
              const size_t ibuf_sz,
              void* const __restrict obuf,
              size_t width,
              size_t height) {
	if(codec == NULL) {
		return NVPIPE_EINVAL;
	}
	nvp_impl_t* dec = (nvp_impl_t*)codec;
	return dec->decode(dec, ibuf,ibuf_sz, obuf, width,height);
}

void
nvpipe_destroy(nvpipe* const __restrict codec) {
	if(codec == NULL) {
		return;
	}
	nvp_impl_t* nvp = (nvp_impl_t*)codec;
	nvp->destroy(nvp);
}

nvp_err_t
nvpipe_bitrate(nvpipe* const __restrict codec, uint64_t br) {
	if(codec == NULL) {
		return NVPIPE_EINVAL;
	}
	nvp_impl_t* nvp = (nvp_impl_t*)codec;
	return nvp->bitrate(codec, br);
}
