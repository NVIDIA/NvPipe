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

struct errstr {
	nvp_err_t code;
	const char* msg;
};
static struct errstr nvp_errors[] = {
	{ NVPIPE_SUCCESS, "success" },
	{ NVPIPE_EINVAL, "invalid value"},
	{ NVPIPE_ENOMEM, "out of memory"},
	{ NVPIPE_EMAP, "map resource"},
	{ NVPIPE_EUNMAP, "unmap resource"},
	{ NVPIPE_ENOENT, "file or resource not found"},
	{ NVPIPE_EENCODE, "encode error from NvEnc"},
	{ NVPIPE_EDECODE, "decode error from cuvid"},
	{ NVPIPE_EOVERFLOW, "buffer would overflow"},
	{ NVPIPE_EAGAIN, "not ready yet"},
};

const char*
nvpipe_strerror(nvp_err_t ecode) {
	const size_t n = sizeof(nvp_errors) / sizeof(nvp_errors[0]);
	for(size_t i=0; i < n; ++i) {
		if(ecode == nvp_errors[i].code) {
			return nvp_errors[i].msg;
		}
	}
	return "unknown";
}
