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
#ifndef NVPIPE_INTERNAL_API_H
#define NVPIPE_INTERNAL_API_H

#include <stdbool.h>
#include "config.nvp.h"
#include "nvpipe.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef nvp_err_t (fqn_encode)(
	nvpipe * const __restrict codec,
	const void *const __restrict ibuf,
	const size_t ibuf_sz,
	void *const __restrict obuf,
	size_t* const __restrict obuf_sz,
	const size_t width, const size_t height,
	nvp_fmt_t format
);
typedef nvp_err_t (fqn_bitrate)(nvpipe* codec, uint64_t);
typedef nvp_err_t (fqn_decode)(
	nvpipe* const __restrict codec,
	const void* const __restrict ibuf, const size_t ibuf_sz,
	void* const __restrict obuf,
	size_t width, size_t height
);
typedef void (fqn_destroy)(nvpipe* const __restrict);

enum objtype {
	ENCODER=0,
	DECODER,
#if NVPIPE_FFMPEG == 1
	FFMPEG
#endif
};

typedef struct nvp_impl_ {
	enum objtype type;
	fqn_encode* encode;
	fqn_bitrate* bitrate;
	fqn_decode* decode;
	fqn_destroy* destroy;
} nvp_impl_t;

nvp_impl_t* nvp_create_encoder(uint64_t bitrate);
nvp_impl_t* nvp_create_decoder();
nvp_impl_t* nvp_create_ffmpeg(bool nvidia, uint64_t bitrate);

#ifdef __cplusplus
}
#endif

#endif
