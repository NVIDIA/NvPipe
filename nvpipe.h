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
#include "libnvpipeutil/format.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _nvpipe {
    enum NVPipeCodecID type_;
    void *codec_ptr_;
} nvpipe;

/*!
    API function calls
 */
nvpipe* nvpipe_create_instance(enum NVPipeCodecID);
void nvpipe_destroy_instance( nvpipe *codec );

int nvpipe_encode(  nvpipe *codec, 
                    void *input_buffer,
                    const size_t input_buffer_size,
                    void *output_buffer,
                    size_t* output_buffer_size,
                    const int width,
                    const int height,
                    enum NVPipeImageFormat format
                    );

int nvpipe_decode(  nvpipe *codec, 
                    void *input_buffer,
                    const size_t input_buffer_size,
                    void *output_buffer,
                    size_t output_buffer_size,
                    int* width,
                    int* height,
                    enum NVPipeImageFormat format
                    );



#ifdef __cplusplus
}
#endif
