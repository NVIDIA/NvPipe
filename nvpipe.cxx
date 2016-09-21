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
#include <cmath>
#include "nvpipe.h"
#include "libnvpipecodec/nvpipecodec.h"
#include "libnvpipecodec/nvpipecodec264.h"
#include "libnvpipeutil/formatConversionCuda.h"
#include "libnvpipeutil/nvpipeError.h"

// profiling
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

NVPipeErrorID nvpipe_create_instance( enum NVPipeCodecID id,
                                      nvpipe** const codec,
                                      uint64_t bitrate)
{
    NVPipeErrorID result = static_cast<NVPipeErrorID>(NVPIPE_SUCCESS);

    switch(id) {
    case NVPIPE_CODEC_ID_NULL:
        *codec = (nvpipe*) calloc( sizeof(nvpipe), 1 );
        (*codec)->type_ = id;
        (*codec)->codec_ptr_ = NULL;
        break;
    case NVPIPE_CODEC_ID_H264_HARDWARE:
        {
        int64_t api_bitrate = static_cast<int64_t>(bitrate);
        if (api_bitrate < 0) {
            (*codec) = NULL;
            return static_cast<NVPipeErrorID>(
                   NVPIPE_ERR_INVALID_BITRATE);
        }
        (*codec) = (nvpipe*) calloc( sizeof(nvpipe), 1 );
        (*codec)->type_ = NVPIPE_CODEC_ID_H264_HARDWARE;
        NvPipeCodec264* ptr = new NvPipeCodec264();
        ptr->setBitrate(bitrate);
        ptr->setCodecImplementation(NV_CODEC);
        (*codec)->codec_ptr_ = ptr;
        break;
        }
    case NVPIPE_CODEC_ID_H264_SOFTWARE:
        {
        int64_t api_bitrate = static_cast<int64_t>(bitrate);
        if (api_bitrate < 0) {
            (*codec) = NULL;
            return static_cast<NVPipeErrorID>(
                   NVPIPE_ERR_INVALID_BITRATE);
        }
        (*codec) = (nvpipe*) calloc( sizeof(nvpipe), 1 );
        (*codec)->type_ = NVPIPE_CODEC_ID_H264_SOFTWARE;
        NvPipeCodec264* ptr = new NvPipeCodec264();
        ptr->setBitrate(bitrate);
        ptr->setCodecImplementation(FFMPEG_LIBX);
        (*codec)->codec_ptr_ = ptr;
        break;
        }
    default:
        (*codec) = NULL;
        break;
    }

    return result;
}

NVPipeErrorID nvpipe_destroy_instance( nvpipe * const __restrict codec)
{
    NVPipeErrorID result = static_cast<NVPipeErrorID>(NVPIPE_SUCCESS);

    if (codec == NULL)
        return static_cast<NVPipeErrorID>(
               NVPIPE_ERR_INVALID_NVPIPE_INSTANCE);
    
    switch(codec->type_) {
    case NVPIPE_CODEC_ID_NULL:
        memset( codec, 0, sizeof(nvpipe) );
        free(codec);
        break;
    case NVPIPE_CODEC_ID_H264_HARDWARE:
    case NVPIPE_CODEC_ID_H264_SOFTWARE:
        delete (NvPipeCodec264*) codec->codec_ptr_;
        memset( codec, 0, sizeof(nvpipe) );
        free(codec);
        break;
    default:
        memset( codec, 0, sizeof(nvpipe) );
        free(codec);
        break;
    }

    return result;
}



NVPipeErrorID nvpipe_encode(nvpipe* const __restrict codec, 
                            void* const __restrict input_buffer,
                            const size_t input_buffer_size,
                            void* const __restrict output_buffer,
                            size_t* const __restrict output_buffer_size,
                            const int width,
                            const int height,
                            enum NVPipeImageFormat format) {
    NVPipeErrorID result = static_cast<NVPipeErrorID>(NVPIPE_SUCCESS);

    if (((width|height)&1) != 0)
        return static_cast<NVPipeErrorID>(
               NVPIPE_ERR_INVALID_RESOLUTION);

    if (codec == NULL)
        return static_cast<NVPipeErrorID>(
               NVPIPE_ERR_INVALID_NVPIPE_INSTANCE);

    NvPipeCodec *codec_ptr = static_cast<NvPipeCodec*> 
                                (codec->codec_ptr_);

    // profiling
    cudaProfilerStart();
    nvtxRangePushA("encodingSession");

    codec_ptr->setImageSize(width, height);
    codec_ptr->setInputFrameBuffer(input_buffer, input_buffer_size);
    result = codec_ptr->encode(output_buffer,
                               *output_buffer_size,
                               format);

    // profiling
    nvtxRangePop();
    cudaProfilerStop();

    return result;

}

NVPipeErrorID nvpipe_decode(  nvpipe* const __restrict codec, 
                              void* const __restrict input_buffer,
                              size_t input_buffer_size,
                              void* const __restrict output_buffer,
                              size_t output_buffer_size,
                              int * const __restrict width,
                              int * const __restrict height,
                              enum NVPipeImageFormat format){
    NVPipeErrorID result = static_cast<NVPipeErrorID>(NVPIPE_SUCCESS);
    
    if (((*width|*height)&1) != 0)
        return static_cast<NVPipeErrorID>(
               NVPIPE_ERR_INVALID_RESOLUTION);

    if (codec == NULL)
        return static_cast<NVPipeErrorID>(
               NVPIPE_ERR_INVALID_NVPIPE_INSTANCE);

    if ( input_buffer_size == 0 ) {
        return static_cast<NVPipeErrorID>(
               NVPIPE_ERR_INPUT_BUFFER_EMPTY_MEMORY);
    }

    NvPipeCodec *codec_ptr = static_cast<NvPipeCodec*> 
                                (codec->codec_ptr_);

    // profiling
    cudaProfilerStart();
    nvtxRangePushA("decodingSession");

    codec_ptr->setImageSize(*width, *height);
    codec_ptr->setInputPacketBuffer(input_buffer, input_buffer_size);
    result = codec_ptr->decode(  output_buffer,
                                 *width,
                                 *height,
                                 output_buffer_size,
                                 format);
    // profiling
    nvtxRangePop();
    cudaProfilerStop();

    return result;
}
