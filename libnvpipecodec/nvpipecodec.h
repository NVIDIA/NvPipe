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
#pragma once

#include "libnvpipeutil/format.h"
#include <cstdlib>


/*! \brief enum for codec implementation
 * 
 * FFMPEG_LIBX is utilizing the libx264 through ffmpeg
 * NV_CODEC is utilizing nvenc/cuvid through ffmpeg
 */
enum NvPipeCodecImplementation {
    FFMPEG_LIBX,
    NV_CODEC
};

/*! \brief Abstract Encoder/Decoder
 *
 *  Abstract NvPipe codec. Defines common interface to codecs.
 *
 */
class NvPipeCodec {
public:
    virtual void setImageSize(int width, int height);

    virtual void setCodecImplementation(
                    enum NvPipeCodecImplementation type);

    virtual void setInputFrameBuffer(   void* frame_buffer,
                                        size_t buffer_size);

    virtual void setInputPacketBuffer(  void* packet_buffer,
                                        size_t buffer_size);

    virtual int encode( void* frame,
                        size_t &size,
                        enum NVPipeImageFormat format)=0;

    virtual int decode( void* packet,
                        int &width,
                        int &height,
                        size_t &size,
                        enum NVPipeImageFormat format)=0; 

    virtual void setBitrate( int64_t bitrate );

    void setGopSize( int gop_size );

    void setFramerate( int framerate );

protected:
    NvPipeCodec();
    virtual ~NvPipeCodec();

    int width_;
    int height_;

    void* frame_;
    size_t frame_buffer_size_;
    enum NVPipeImageFormat encoder_format_;
    void* packet_;
    size_t packet_buffer_size_;

    enum NVPipeImageFormat decoder_format_;

    bool bitrate_overwrite_flag_;
    int64_t bitrate_;

    int gop_size_;

    int framerate_;
    
    enum NvPipeCodecImplementation codec_;

    enum NvPipeCodecImplementation getCodec();

private:

};
