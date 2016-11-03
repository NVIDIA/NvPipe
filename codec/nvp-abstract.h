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
#ifndef NVPIPE_CODEC_H_
#define NVPIPE_CODEC_H_
#include "util/format.h"
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
    NvPipeCodec();
    virtual ~NvPipeCodec();

    virtual void setImageSize(size_t width, size_t height);

    virtual void setCodecImplementation(
                    enum NvPipeCodecImplementation type);

    virtual void setInputFrameBuffer(const void* frame_buffer,
                                     size_t buffer_size);

    virtual void setInputPacketBuffer(const void* packet_buffer,
                                      size_t buffer_size);

    virtual nvp_err_t encode(void* frame,
                        size_t &size,
                        enum NVPipeImageFormat format)=0;

    virtual nvp_err_t decode(void* packet,
                        size_t &width,
                        size_t &height,
                        size_t &size,
                        enum NVPipeImageFormat format)=0; 

    virtual void setBitrate( int64_t bitrate );

    void setGopSize( int gop_size );

    void setFramerate( int framerate );

protected:
    size_t width_;
    size_t height_;

    const void* frame_;
    size_t frame_buffer_size_;
    enum NVPipeImageFormat encoder_format_;
    const void* packet_;
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
#endif //NVPIPE_CODEC_H_
