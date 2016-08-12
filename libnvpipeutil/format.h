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

#include <stdlib.h>
#include "libavformat/avformat.h"
#include "nvpipe.h"

#ifdef __cplusplus
extern "C" {
#endif


/*!
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

typedef struct _nvpipeMemGpu2 {
    unsigned int* d_buffer_1_;
    unsigned int* d_buffer_2_;
    size_t d_buffer_1_size_;
    size_t d_buffer_2_size_;
} nvpipeMemGpu2;

void initializeMemGpu2(nvpipeMemGpu2 *mem_gpu);

void allocateMemGpu2(   nvpipeMemGpu2 *mem_gpu, 
                        size_t size_1, size_t size_2);
void destroyMemGpu2(nvpipeMemGpu2 *mem_gpu);

int formatConversion( int w, int h,
                        void* imagePtrARGB,
                        void* imagPtrNV12,
                        enum NVPipeImageFormatConversion);

int formatConversionReuseMemory( int w, int h,
                        void* imagePtrARGB,
                        void* imagPtrNV12,
                        enum NVPipeImageFormatConversion,
                        nvpipeMemGpu2 *mem_gpu2);

int formatConversionAVFrameRGB( AVFrame *frame,
                                void *buffer);

int formatConversionAVFrameRGBReuseMemory( AVFrame *frame,
                                void *buffer,
                                nvpipeMemGpu2 *mem_gpu2);

int formatConversionAVFrameRGBA( AVFrame *frame,
                                 void *buffer);

int formatConversionAVFrameRGBAReuseMemory( AVFrame *frame,
                                void *buffer,
                                nvpipeMemGpu2 *mem_gpu2);

#ifdef __cplusplus
}
#endif
