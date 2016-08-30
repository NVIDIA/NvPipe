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
#include "libnvpipecodec/nvpipecodec.h"
#include <limits>

NvPipeCodec::NvPipeCodec() {
    width_ = 0;
    height_ = 0;
    encoder_format_ = NVPIPE_IMAGE_FORMAT_NULL;
    decoder_format_ = NVPIPE_IMAGE_FORMAT_NULL;

    frame_ = NULL;
    frame_buffer_size_ = 0;
    packet_ = NULL;
    packet_buffer_size_ = 0;

    bitrate_ = 1000000;
    //gop_size_ = std::numeric_limits<int>::max();
    framerate_ = 30;
    gop_size_ = 60;

    bitrate_overwrite_flag_ = false;

    codec_ = NV_CODEC;
}

NvPipeCodec::~NvPipeCodec() {

}

void NvPipeCodec::setImageSize(int width, int height) {
    width_ = width;
    height_ = height;
}

void NvPipeCodec::setInputPacketBuffer(void* packet_buffer, size_t buffer_size) {
    packet_ = packet_buffer;
    packet_buffer_size_ = buffer_size;
}

void NvPipeCodec::setInputFrameBuffer(void* frame_buffer, size_t buffer_size) {
    frame_ = frame_buffer;
    frame_buffer_size_ = buffer_size;
}

void NvPipeCodec::setBitrate( int64_t bitrate ) {
    if ( bitrate != 0) {
        bitrate_overwrite_flag_ = true;
        bitrate_ = bitrate;
    } else {
        bitrate_overwrite_flag_ = false;
    }
}

void NvPipeCodec::setGopSize( int gop_size ) {
    gop_size_ = gop_size;
}

void NvPipeCodec::setFramerate( int framerate ) {
    framerate_ = framerate;
}

enum NvPipeCodecImplementation NvPipeCodec::getCodec() {
    return codec_;
}

void NvPipeCodec::setCodecImplementation(
            enum NvPipeCodecImplementation type) {
    codec_ = type;
}

