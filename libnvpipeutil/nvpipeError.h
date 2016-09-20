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
#ifndef NVPIPE_ERROR_H_
#define NVPIPE_ERROR_H_

#define ERROR_INDEX -100

enum NVPipeErrorType {
    NVPIPE_SUCCESS=0,
    NVPIPE_ERR_INVALID_IMAGE_FORMAT=ERROR_INDEX,
    NVPIPE_ERR_INVALID_CODEC_ID,
    NVPIPE_ERR_INVALID_NVPIPE_INSTANCE,
    NVPIPE_ERR_INVALID_RESOLUTION,
    NVPIPE_ERR_INPUT_BUFFER_EMPTY_MEMORY,
    NVPIPE_ERR_OUTPUT_BUFFER_OVERFLOW,
    NVPIPE_ERR_CUDA_ERROR,
    NVPIPE_ERR_FFMPEG_ERROR,
    NVPIPE_ERR_FFMPEG_CAN_NOT_FIND_ENCODER,
    NVPIPE_ERR_FFMPEG_CAN_NOT_FIND_DECODER,
    NVPIPE_ERR_FFMPEG_CAN_NOT_ALLOCATE_FRAME,
    NVPIPE_ERR_FFMPEG_CAN_NOT_ALLOCATE_CONTEXT,
    NVPIPE_ERR_FFMPEG_CAN_NOT_OPEN_CODEC,
    NVPIPE_ERR_FFMPEG_CAN_NOT_BOUND_FRAME,
    NVPIPE_ERR_FFMPEG_LATENCY_OUTPUT_NOT_READY,
    NVPIPE_ERR_FFMPEG_SEND_FRAME,
    NVPIPE_ERR_FFMPEG_SEND_PACKET,
    NVPIPE_ERR_FFMPEG_RECEIVE_PACKET,
    NVPIPE_ERR_FFMPEG_RECEIVE_FRAME,PACKET,

#endif
