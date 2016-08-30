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

#include <stdlib.h>
#include "libavformat/avformat.h"
#include "nvpipe.h"

#ifdef __cplusplus
extern "C" {
#endif


/*! \brief image conversion flag
 *
 *  NVPIPE_IMAGE_FORMAT_CONVERSION_X_TO_Y
 *  encoder:
 *      convert image from X to Y for encoding
 *  decoder:
 *      convert image from Y to X for decoding
 */
enum NVPipeImageFormatConversion {
    NVPIPE_IMAGE_FORMAT_CONVERSION_NULL,
    NVPIPE_IMAGE_FORMAT_CONVERSION_RGB_TO_NV12,
    NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGB,
    NVPIPE_IMAGE_FORMAT_CONVERSION_ARGB_TO_NV12,
    NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_ARGB,
    NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGBA,
    NVPIPE_IMAGE_FORMAT_CONVERSION_RGBA_TO_NV12
};

/// memory struct used to hold device buffer used for image format conversion
typedef struct _nvpipeMemGpu2 {
    unsigned int* d_buffer_1_;
    unsigned int* d_buffer_2_;
    size_t d_buffer_1_size_;
    size_t d_buffer_2_size_;
} nvpipeMemGpu2;

void initializeMemGpu2(nvpipeMemGpu2 *mem_gpu);

/// allocate GPU memory for format conversion. free any memory already allocated
void allocateMemGpu2(   nvpipeMemGpu2 *mem_gpu, 
                        size_t size_1, size_t size_2);

/// free allocated GPU memory.
void destroyMemGpu2(nvpipeMemGpu2 *mem_gpu);

/*! \brief format conversion
 *
 * expecting input and output buffer on the host memory from user.
 * conversion done through cuda.
 * allocate and free GPU memory for every conversion.
 */
int formatConversion( int w, int h,
                        void* imagePtrSrc,
                        void* imagePtrDes,
                        enum NVPipeImageFormatConversion);

/*! \brief format conversion with reusing GPU memory
 *
 * expecting input and output buffer on the host memory from user.
 * conversion done through cuda.
 * use reusable GPU memory struct mem_gpu2.
 */
int formatConversionReuseMemory( int w, int h,
                        void* imagePtrSrc,
                        void* imagePtrDes,
                        enum NVPipeImageFormatConversion,
                        nvpipeMemGpu2 *mem_gpu2);

/*! \brief Convert AVframe (ffmpeg) frame to RGB buffer
 *
 * expecting frame and buffer pointing to host memory.
 * conversion done through cuda.
 * allocate and free GPU memory for every conversion.
 */
int formatConversionAVFrameRGB( AVFrame *frame,
                                void *buffer);

/*! \brief Convert AVframe (ffmpeg) frame to RGB buffer with reusing GPU memory
 *
 * expecting frame and buffer pointing to host memory.
 * conversion done through cuda.
 * use reusable GPU memory struct mem_gpu2.
 */
int formatConversionAVFrameRGBReuseMemory( AVFrame *frame,
                                void *buffer,
                                nvpipeMemGpu2 *mem_gpu2);

/*! \brief Convert AVframe (ffmpeg) frame to RGBA buffer
 *
 * expecting frame and buffer pointing to host memory.
 * conversion done through cuda.
 * allocate and free GPU memory for every conversion.
 */
int formatConversionAVFrameRGBA( AVFrame *frame,
                                 void *buffer);

/*! \brief Convert AVframe (ffmpeg) frame to RGBA buffer with reusing GPU memory
 *
 * expecting frame and buffer pointing to host memory.
 * conversion done through cuda.
 * use reusable GPU memory struct mem_gpu2.
 */
int formatConversionAVFrameRGBAReuseMemory( AVFrame *frame,
                                void *buffer,
                                nvpipeMemGpu2 *mem_gpu2);

#ifdef __cplusplus
}
#endif
