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
#pragma once


#include "libnvpipecodec/nvpipecodec264.h"
#include <cstdio>

#define NVPIPE_H264_ENCODER_NAME "h264_nvenc"
#define NVPIPE_H264_DECODER_NAME "h264_cuvid"

NvPipeCodec264::NvPipeCodec264() {
    printf("nv_codec_h264 created\n");
    
    encoder_context_ = NULL;
    encoder_codec_ = NULL;
    encoder_frame_ = NULL;

    decoder_context_ = NULL;
    decoder_codec_ = NULL;
    decoder_frame_ = NULL;
    
    frame_pixel_format_ = AV_PIX_FMT_RGB24; 
    
    // doesn't really matter
    encoder_config_corrupted_ = true;
    decoder_config_corrupted_ = true;
    // register all available file formats and codecs
    // could be initialized multiple times.

    printf("test\n");
    av_register_all();
    printf("test2\n");
}

NvPipeCodec264::~NvPipeCodec264() {
    printf("nv_codec_h264 destroyed\n");
    
    if (decoder_codec_ && decoder_context_) {
        avcodec_close(decoder_context_);
    }
    if (decoder_context_) {
        av_free(decoder_context_);
    }
    if (decoder_frame_) {
        av_frame_free(&decoder_frame_);
    }
    
    
    if (encoder_codec_ && encoder_context_) {
        avcodec_close(encoder_context_);
    }
    if (encoder_context_) {
        av_free(encoder_context_);
    }
    if (encoder_frame_) {
        av_frame_free(&encoder_frame_);
    }

}

void NvPipeCodec264::setImageSize(   int width,
                                int height,
                                enum NVPipeImageFormat format ) {
    if ( width != width_ || height != height_ || format != format_ ) {
        encoder_config_corrupted_ = true;

        switch( format ) {
        case NVPIPE_IMAGE_FORMAT_RGBA:
            frame_pixel_format_ = AV_PIX_FMT_RGBA;
            break;
        default:
        case NVPIPE_IMAGE_FORMAT_RGB:
            frame_pixel_format_ = AV_PIX_FMT_RGB24; 
            break;
        }
        
        NvPipeCodec::setImageSize(width, height, format);
    }
}

void NvPipeCodec264::setFrameBuffer(void* frame_buffer, size_t buffer_size) {
    if ( frame_buffer != frame_ || buffer_size != frame_buffer_size_ ) {
        encoder_config_corrupted_ = true;
        
        NvPipeCodec::setFrameBuffer(frame_buffer, buffer_size);
    }
}

int NvPipeCodec264::encode(void* buffer, size_t &size) {

    if (width_ == 0 || height_ == 0 
        || format_ == NVPIPE_IMAGE_FORMAT_NULL ) {
            printf("input frame has to be defined \
                    before calling NvPipeCodec264::encoding");
            return -1;
    }

    // Check if encoder has been initialized
    if (encoder_codec_ == NULL) {
        encoder_codec_ = avcodec_find_encoder_by_name(NVPIPE_H264_ENCODER_NAME);
        if (encoder_codec_ == NULL) {
            printf("cannot find encoder: %s", NVPIPE_H264_ENCODER_NAME);
            return -1;
        }
    }
    
    if (encoder_context_ == NULL) {
        encoder_context_ = avcodec_alloc_context3( encoder_codec_ );
        if (encoder_context_ == NULL) {
            printf("cannot allocate codec context");
            return -1;
        }
        
    }

    if (encoder_frame_ == NULL) {
        encoder_frame_ = av_frame_alloc();
        if (encoder_frame_ == NULL) {
            printf("cannot allocate frame");
            return -1;
        }
        
    }

    if (encoder_config_corrupted_) {
        /* put sample parameters */
        encoder_context_->bit_rate = 400000;
        /* frames per second */
        encoder_context_->time_base = (AVRational){1,25};
        /* emit one intra frame every ten frames
        * check frame pict_type before passing frame
        * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
        * then gop_size is ignored and the output of encoder
        * will always be I frame irrespective to gop_size
        */
        encoder_context_->gop_size = 10;
        encoder_context_->max_b_frames = 1;

        encoder_context_->width = width_;
        encoder_context_->height = height_;
        //encoder_context_->pix_fmt = frame_pixel_format_;
        encoder_context_->pix_fmt = AV_PIX_FMT_YUV420P;
        
        //encoder_frame_->format = frame_pixel_format_;
        encoder_frame_->format = AV_PIX_FMT_YUV420P;
        encoder_frame_->width = width_;
        encoder_frame_->height = height_;

        // I don't really under stand this at all.. Maybe framerate?
        encoder_context_->time_base= (AVRational){1,25};
        
        if (avcodec_open2(encoder_context_, encoder_codec_, NULL) != 0) {
            printf("cannot open codec\n");
            return -1;
        }

        int ret;
        /*if ( av_image_fill_arrays ( encoder_frame_->data, 
                                    encoder_frame_->linesize,
                                    (const uint8_t*) frame_, 
                                    //frame_pixel_format_,
                                    AV_PIX_FMT_YUV420P,
                                    width_,
                                    height_,
                                    64 ) < 0 ) {
            printf("could not associate image buffer to frame");
            return -1;
        }*/
        /*ret = av_image_fill_arrays ( encoder_frame_->data, 
                                    encoder_frame_->linesize,
                                    (const uint8_t*) frame_, 
                                    //frame_pixel_format_,
                                    AV_PIX_FMT_YUV420P,
                                    width_,
                                    height_,
                                    32 );
        */
        printf("image size: width %d, height %d\n", width_, height_);
        ret = av_image_alloc(encoder_frame_->data,encoder_frame_->linesize,width_, height_, AV_PIX_FMT_YUV420P, 32);
        
        printf("image_fill: %zu, %d\n", encoder_frame_->linesize[0]*height_, ret);
        
        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                encoder_frame_->data[0][y * encoder_frame_->linesize[0] + x] = x + y;
            }
        }
        /* Cb and Cr */
        for (int y = 0; y < height_/2; y++) {
            for (int x = 0; x < width_/2; x++) {
                encoder_frame_->data[1][y * encoder_frame_->linesize[1] + x] = 128 + y;
                encoder_frame_->data[2][y * encoder_frame_->linesize[2] + x] = 64 + x;
            }
        }

        encoder_frame_->pts = 1;
        
        encoder_config_corrupted_ = false;
    }

    av_init_packet(&encoder_packet_);
    encoder_packet_.data = NULL;
    encoder_packet_.size = 0;

    if ( avcodec_send_frame(encoder_context_, encoder_frame_) != 0 ) {
        printf("send_frame went wrong!");
    }

    if ( avcodec_receive_packet(encoder_context_, 
                                &encoder_packet_) != 0 ) {
        printf("receive_packet went wrong!");
    }

    int got_output;
    //int ret = avcodec_encode_video2(encoder_context_, &encoder_packet_, encoder_frame_, &got_output);
    if (ret < 0) {
        printf("error encoding frame %d\n", ret);
        return -1;
    }
    /*int ret = avcodec_receive_packet(encoder_context_, &encoder_packet_);
    printf("\nreceive_packet went wrong! %d\n", ret);
    switch(ret) {
        case AVERROR(EOF):
            printf("eof\n");
            break;
        case AVERROR(EAGAIN):
            printf("EAGAIN\n");
            break;
        case AVERROR(EINVAL):
            printf("EINVAL\n");
            break;
        case AVERROR(ENOMEM):
            printf("ENOMEN\n");
            break;
    }*/
    
    if ( encoder_packet_.size > size ) {
        size = encoder_packet_.size;
        av_packet_unref(&encoder_packet_);
        printf("packet size larger than  buffer_size went wrong!");
        return -1;
    }

    memcpy(buffer, encoder_packet_.data, encoder_packet_.size);
    size = encoder_packet_.size;
    av_packet_unref(&encoder_packet_);
    printf("encoding finished\n");
    printf("packet data: %d, packet size: %zu\n", buffer, size);
    // remove this;
    av_freep(encoder_frame_->data[0]);
    
    return 0;
}

int NvPipeCodec264::decode(void* picture, int &width, int &height, size_t &size) {
    
    /*   
    if ( width_ == 0 || height_ == 0 ) {
        printf("input picture has to defined\
                 before calling NvPipeCodec264::decoding");
        return -1;
    }
    */
        
    // Check if encoder has been initialized
    if (decoder_codec_ == NULL) {
        decoder_codec_ = avcodec_find_decoder_by_name(NVPIPE_H264_DECODER_NAME);
        if (decoder_codec_ == NULL) {
            printf("cannot find encoder: %s", NVPIPE_H264_DECODER_NAME);
            return -1;
        }
    }
        
    if (decoder_context_ == NULL) {
        decoder_context_ = avcodec_alloc_context3(decoder_codec_);
        if (decoder_context_ == NULL) {
            printf("cannot allocate codec context");
            return -1;
        }
    }
        
    if (avcodec_open2(decoder_context_, decoder_codec_, NULL) < 0) {
        printf("cannot open codec\n");
        return -1;
    }
    
    decoder_frame_ = av_frame_alloc();
    if (decoder_frame_ == NULL) {
        printf("cannot allocate frame");
        return -1;
    }

    //decoder_context_->width = width_;
    //decoder_context_->height = height_;
    //decoder_context_->pix_fmt = AV_PIX_FMT_RGB24;
    //frame->pts = frame->ptr+1;

    av_init_packet(&decoder_packet_);
    
    decoder_packet_.data = (uint8_t *) packet_;
    decoder_packet_.size = packet_buffer_size_;
    printf(" @@ packet data: %d, packet size: %zu\n", packet_, packet_buffer_size_);
    if ( avcodec_send_packet(   decoder_context_,
                                &decoder_packet_) != 0 ) {
        printf("send_packet went wrong!\n");
    }

    if ( avcodec_receive_frame( decoder_context_,
                                decoder_frame_) != 0 ) {
        printf("receive_frame went wrong!\n");
    }

    size_t frameSize = decoder_frame_->height;
    frameSize *= decoder_frame_->linesize[0];
    
    width = decoder_frame_->width;
    height = decoder_frame_->height;
    
    if (size <= frame_buffer_size_ ) {
        size = frameSize;
        printf("frame size larger than frame_buffer_size_,\
                something went wrong!\n");
        return -1;
    }

    memcpy(frame_, decoder_frame_->data[0], frameSize);
    size = frameSize;
    printf("decoding!");
    return 0;
}
