/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef FORMAT_H_
#define FORMAT_H_

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
    NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGBA,
    NVPIPE_IMAGE_FORMAT_CONVERSION_RGBA_TO_NV12
};

/// memory struct used to hold device buffer used for image format conversion
typedef struct _nvpipeMemGPU {
    unsigned int* d_buffer_1_;
    unsigned int* d_buffer_2_;
    size_t d_buffer_1_size_;
    size_t d_buffer_2_size_;
} nvpipeMemGPU;

void initializeMemGPU(nvpipeMemGPU *mem_gpu);

/// allocate GPU memory for format conversion. free any memory already allocated
void allocateMemGPU(   nvpipeMemGPU *mem_gpu, 
                        size_t size_1, size_t size_2);

/// free allocated GPU memory.
void destroyMemGPU(nvpipeMemGPU *mem_gpu);

/*! \brief format conversion with reusing GPU memory
 *
 * expecting input and output buffer on the host memory from user.
 * conversion done through cuda.
 * use reusable GPU memory struct mem_gpu2.
 */
int formatConversionReuseMemory( int w, int h, int align,
                        const void* imagePtrSrc,
                        void* imagePtrDes,
                        enum NVPipeImageFormatConversion,
                        nvpipeMemGPU *mem_gpu);

/*! \brief Convert AVframe (ffmpeg) frame to RGB buffer with reusing GPU memory
 *
 * expecting frame and buffer pointing to host memory.
 * conversion done through cuda.
 * use reusable GPU memory struct mem_gpu2.
 */
int formatConversionAVFrameRGBReuseMemory( AVFrame *frame, int align,
                                void *buffer,
                                nvpipeMemGPU *mem_gpu);

/*! \brief Convert AVframe (ffmpeg) frame to RGBA buffer with reusing GPU memory
 *
 * expecting frame and buffer pointing to host memory.
 * conversion done through cuda.
 * use reusable GPU memory struct mem_gpu2.
 */
int formatConversionAVFrameRGBAReuseMemory( AVFrame *frame, int align,
                                void *buffer,
                                nvpipeMemGPU *mem_gpu);

#ifdef __cplusplus
}
#endif

#endif //FORMAT_H_
