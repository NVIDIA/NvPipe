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
#include "libnvpipeutil/nvpipeError.h"

static const struct {
    NVPipeErrorType index_;
    const char * description_;
} nvpipe_error[] = {
    { NVPIPE_SUCCESS,
      "success"},
    { NVPIPE_ERR_INVALID_IMAGE_FORMAT,
      "invalid image format"}, 
    { NVPIPE_ERR_INVALID_CODEC_ID,
      "invalid codec id"},
    { NVPIPE_ERR_INVALID_NVPIPE_INSTANCE,
      "invalid nvpipe instance"},
    { NVPIPE_ERR_INVALID_RESOLUTION,
      "invalid resolution"},
    { NVPIPE_ERR_OUTPUT_BUFFER_OUT_OF_MEMORY,
      "output buffer out of memory"},
    { NVPIPE_ERR_LATENCY_OUTPUT_NOT_READY,
      "output not ready, try again"},
    { NVPIPE_ERR_CUDA_ERROR,
      "CUDA error"},
    { NVPIPE_ERR_CUDA_FFMPEG_ERROR,
      "FFmpeg error"},
    { NVPIPE_ERR_UNIDENTIFIED_ERROR_CODE,
      "unidentified error code"}
};

const char * nvpipe_check_error( NVPipeErrorID error_code ) {
    int index = error_code; 
    if (index != 0) {
        index -= ERROR_INDEX;
        index += 1;
        NVPipeErrorType errorEnum = 
                static_cast<NVPipeErrorType>(error_code);
        if ( nvpipe_error[index].index_ != errorEnum ) {
           index = NVPIPE_ERR_UNIDENTIFIED_ERROR_CODE;
           index -= ERROR_INDEX;
           index += 1;
        }
    }
    return nvpipe_error[index].description_;
}

