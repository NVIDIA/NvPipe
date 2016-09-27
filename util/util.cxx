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
#include "util/nvpipeError.h"
#include "nvpipe.h"

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
    { NVPIPE_ERR_INVALID_BITRATE,
      "invalid bitrate"},
    { NVPIPE_ERR_INPUT_BUFFER_EMPTY_MEMORY,
      "empty input buffer"},
    { NVPIPE_ERR_OUTPUT_BUFFER_OVERFLOW,
      "output buffer overflow"},
    { NVPIPE_ERR_CUDA_ERROR,
      "CUDA error"},
    { NVPIPE_ERR_FFMPEG_ERROR,
      "FFmpeg error"},
    { NVPIPE_ERR_FFMPEG_CAN_NOT_FIND_ENCODER,
      "FFmpeg cannot find encoder"},
    { NVPIPE_ERR_FFMPEG_CAN_NOT_FIND_DECODER,
      "FFmpeg cannot find decoder"},
    { NVPIPE_ERR_FFMPEG_CAN_NOT_ALLOCATE_FRAME,
      "FFmpeg cannot allocate frame"},
    { NVPIPE_ERR_FFMPEG_CAN_NOT_ALLOCATE_CONTEXT,
      "FFmpeg cannot allocate codec context"},
    { NVPIPE_ERR_FFMPEG_CAN_NOT_OPEN_CODEC,
      "FFmpeg cannot open codec"},
    { NVPIPE_ERR_FFMPEG_CAN_NOT_BOUND_FRAME,
      "FFmpeg cannot bound frame buffer"},
    { NVPIPE_ERR_FFMPEG_LATENCY_OUTPUT_NOT_READY,
      "FFmpeg output latency, not ready try again"},
    { NVPIPE_ERR_FFMPEG_SEND_FRAME,
      "FFmpeg send frame error"},
    { NVPIPE_ERR_FFMPEG_SEND_PACKET,
      "FFmpeg send packet error"},
    { NVPIPE_ERR_FFMPEG_RECEIVE_PACKET,
      "FFmpeg receive packet error"},
    { NVPIPE_ERR_FFMPEG_RECEIVE_FRAME,
      "FFmpeg receive frame error"},
    { NVPIPE_ERR_UNIDENTIFIED_ERROR_CODE,
      "nidentified error code"}
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

