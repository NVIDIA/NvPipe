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
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (nvpipe_fptr_size2) (void*, size_t, size_t);
typedef void (nvpipe_fptr_voidPtr) (void*, void*);
typedef void (nvpipe_fptr_size) (void*, size_t);
typedef void (nvpipe_fptr_void) (void*);

typedef int (nvpipe_fptr_encode_frame) (void* codec, size_t* buffer_size);
typedef int (nvpipe_fptr_decode_frame) (void* codec, size_t* width, size_t* height); 

enum NVPipeCodecID {
    NVPIPE_CODEC_ID_NULL,
    NVPIPE_CODEC_ID_NVENC,
    NVPIPE_CODEC_ID_CUVID
};

typedef struct _nvpipecodec {

    enum NVPipeCodecID type;

    nvpipe_fptr_decode_frame *decodeFnc;
    nvpipe_fptr_encode_frame *encodeFnc;
    nvpipe_fptr_size2 *setSize;
    nvpipe_fptr_voidPtr *setPicturePtr;
    nvpipe_fptr_voidPtr *setVideoPtr;
    nvpipe_fptr_size *setBufferSize;
    nvpipe_fptr_void *destroy;

    void *codecPtr;

    void *videoPtr;
    void *picturePtr;
    size_t width;
    size_t height;
    size_t buffer;

} nvpipecodec;

/*!
    API function calls
 */
nvpipecodec* nvpipe_create_codec(enum NVPipeCodecID);
int nvpipe_encode(nvpipecodec *codec, size_t* buffer_size);
int nvpipe_decode(nvpipecodec *codec, size_t* width, size_t* height);


/*!
 */
nvpipecodec* nvpipe_create_decoder_cuvid();
nvpipecodec* nvpipe_create_encoder_nvenc();



#ifdef __cplusplus
}
#endif
