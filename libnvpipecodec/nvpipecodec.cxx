/*
 * Copyright (c) 2016 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual
 * property and proprietary rights in and to this software,
 * related documentation and any modifications thereto.  Any use,
 * reproduction, disclosure or distribution of this software and
 * related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE
 * IS PROVIDED *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL
 * WARRANTIES, EITHER EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED
 * TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE
 * LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL
 * DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
 * LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
 * INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF
 * OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGES
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

