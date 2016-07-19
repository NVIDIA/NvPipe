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

#include "nvpipe.h"
#include "libnvpipecodec/nvpipedecodercuvid.h"
#include "libnvpipecodec/nvpipeencodernvenc.h"

static void
setSize(void* codec, size_t width, size_t height ) {
    
}

static void
setPicturePtr(void* codec, void* picture ) {
    
}

static void
setVideoPtr(void* codec, void* video ) {
    
}

static void
setBufferSize(void* codec, size_t buffer_size ) {

}

static void
destroy(void* codec) {
}

static int
encodeFnc (void* codec, size_t* buffer_size ) {
    return 0;
}

static int
decodeFnc (void* codec, size_t* width, size_t* height ) {
    return 0;
}

nvpipecodec* nvpipe_create_encoder_nvenc() {
    nvpipecodec* codec = (nvpipecodec*) calloc(sizeof(nvpipecodec),1);
    codec->type = NVPipeCodecID::NVPIPE_CODEC_ID_CUVID;
    codec->decodeFnc = decodeFnc;
    codec->setSize = setSize;
    codec->setPicturePtr = setPicturePtr;
    codec->setVideoPtr = setVideoPtr;
    codec->setBufferSize = setBufferSize;
    codec->destroy = destroy;
    
    codec->codecPtr = new NvPipeDecoderCuvid();
    
    return codec;
}

nvpipecodec* nvpipe_create_decoder_cuvid() {
    nvpipecodec* codec = (nvpipecodec*) calloc(sizeof(nvpipecodec),1);
    codec->type = NVPipeCodecID::NVPIPE_CODEC_ID_NVENC;
    codec->decodeFnc = decodeFnc;
    codec->setSize = setSize;
    codec->setPicturePtr = setPicturePtr;
    codec->setVideoPtr = setVideoPtr;
    codec->setBufferSize = setBufferSize;
    codec->destroy = destroy;
    
    codec->codecPtr = new NvPipeEncoderNvenc();
    
    return codec;
}
