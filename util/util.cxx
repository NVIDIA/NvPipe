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
#include "nvpipe.h"

/* Note these need to sync up with the ordering for the nvp_err_t enum in
 * nvpipe.h! */
static const char* nvp_errors[] = {
	"success",
	"invalid image format",
	"invalid codec id",
	"invalid nvpipe instance",
	"invalid resolution",
	"invalid bitrate",
	"empty input buffer",
	"output buffer overflow",
	"CUDA error",
	"FFmpeg error",
	"FFmpeg cannot find encoder",
	"FFmpeg cannot find decoder",
	"FFmpeg cannot allocate frame",
	"FFmpeg cannot allocate codec context",
	"FFmpeg cannot open codec",
	"FFmpeg cannot bound frame buffer",
	"FFmpeg output latency, not ready try again",
	"FFmpeg send frame error",
	"FFmpeg send packet error",
	"FFmpeg receive packet error",
	"FFmpeg receive frame error",
	"unknown error"
};

const char*
nvpipe_strerror(nvp_err_t error_code) {
	const size_t i = static_cast<size_t>(error_code);
	if(i >= (sizeof(nvp_errors) / sizeof(nvp_errors[0]))) {
		return "invalid NVPipe error code!";
	}
	return nvp_errors[i];
}
