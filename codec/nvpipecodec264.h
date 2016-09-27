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
#ifndef NVPIPE_CODEC_264_H_
#define NVPIPE_CODEC_264_H_

#ifdef __cplusplus
extern "C"
{
#endif
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#ifdef __cplusplus
}
#endif

#include "codec/nvpipecodec.h"
#include "util/formatConversionCuda.h"
#include <string>

/*! \brief H.264 Encoder/Decoder
 *
 *  Implementation using ffmpeg libx264 and nvenc/cuvid
 *
 *  Encoder takes image as input and generates packets as output. While
 *  Decoder works the opposite.
 *
 */
class NvPipeCodec264 : public NvPipeCodec {
public:
    virtual nvp_err_t encode( void* buffer,
                        size_t &size,
                        enum NVPipeImageFormat format);

    virtual nvp_err_t decode( void* picture,
                        size_t& width,
                        size_t& height,
                        size_t &size,
                        enum NVPipeImageFormat format);

    virtual void setImageSize(size_t width, size_t height);

    virtual void setBitrate( int64_t bitrate );

    virtual void setInputFrameBuffer(   void* frame_buffer, 
                                        size_t buffer_size);

    NvPipeCodec264();
    virtual ~NvPipeCodec264();

protected:
    AVCodecContext *encoder_context_;
    AVCodec *encoder_codec_;
    AVFrame *encoder_frame_;
    AVPacket encoder_packet_;

    AVCodecContext *decoder_context_;
    AVCodec *decoder_codec_;
    AVFrame *decoder_frame_;
    AVPacket decoder_packet_;

private:

    nvpipeMemGPU mem_gpu_;

    nvp_err_t getFormatConversionEnum(
            enum NVPipeImageFormat format,
            bool encoder_flag,
            enum NVPipeImageFormatConversion &conversion_flag,
            enum AVPixelFormat &pixel_format);

    // append 2 dummy access delimiter NAL units at the end.
    void appendDummyNAL(void* buffer, size_t offset);

    bool encoder_config_dirty_;
    bool decoder_config_dirty_;

    bool encoder_frame_buffer_dirty_;

    enum NVPipeImageFormatConversion encoder_conversion_flag_;
    void* encoder_converted_image_buffer_;
    size_t encoder_converted_image_buffer_size_;

    enum NVPipeImageFormatConversion decoder_conversion_flag_;

    enum AVPixelFormat encoder_frame_pixel_format_;

    std::string getEncoderName();
    std::string getDecoderName();

    nvp_err_t configureEncoderContext();
};
#endif //NVPIPE_CODEC_264_H_
