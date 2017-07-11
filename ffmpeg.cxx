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
#include <cassert>
#include <cstdlib>
#include <cstring>
#include "config.nvp.h"
#include "codec/nvp-abstract.h"
#include "codec/nvp-hw.h"
#include "internal-api.h"
#include "nvpipe.h"
#include "util/formatConversionCuda.h"

#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

struct nvpipe_ {
    nvp_impl_t impl;
    NvPipeCodec* codec_ptr_;
};

struct profile_blk {
    profile_blk(const char* const s) {
        nvtxRangePushA(s);
    }
    ~profile_blk() {
        nvtxRangePop();
    }
};

static void
nvp_ffmpeg_destroy(nvpipe* const __restrict cdc) {
    if(cdc == NULL) {
        return;
    }
    nvpipe_* codec = static_cast<nvpipe_*>(cdc);
    assert(codec->impl.type == FFMPEG);

    delete codec->codec_ptr_;
    memset(codec, 0, sizeof(nvpipe_));
    free(codec);
}

static nvp_err_t
nvp_ffmpeg_encode(nvpipe* const __restrict cdc,
                  const void* const __restrict input_buffer,
                  const size_t input_buffer_size,
                  void* const __restrict output_buffer,
                  size_t* const __restrict output_buffer_size,
                  const size_t width, const size_t height, nvp_fmt_t format)
{
    assert(cdc);
    nvpipe_* codec = static_cast<nvpipe_*>(cdc);

    /* input images must be a power of two */
    if(((width | height) & 1) != 0) {
        return NVPIPE_EINVAL;
    }

    NvPipeCodec* codec_ptr = static_cast<NvPipeCodec*>(codec->codec_ptr_);

    profile_blk enc_("encodingSession");

    codec_ptr->setImageSize(width, height);
    codec_ptr->setInputFrameBuffer(input_buffer, input_buffer_size);
    return codec_ptr->encode(output_buffer, *output_buffer_size, format);
}

static nvp_err_t
nvp_ffmpeg_decode(nvpipe* const __restrict cdc,
                  const void* const __restrict input_buffer,
                  const size_t input_buffer_size,
                  void* const __restrict output_buffer,
                  size_t width, size_t height) {
    assert(cdc);
    /* input images must be a power of two */
    if(((width | height) & 1) != 0) {
        return NVPIPE_EINVAL;
    }

    if(input_buffer_size == 0) {
        return NVPIPE_EINVAL;
    }
    nvpipe_* codec = static_cast<nvpipe_*>(cdc);

    NvPipeCodec *codec_ptr = static_cast<NvPipeCodec*>(codec->codec_ptr_);

    profile_blk decodesess("decodingSession");

    codec_ptr->setImageSize(width, height);
    codec_ptr->setInputPacketBuffer(input_buffer, input_buffer_size);
    size_t sz = width*height*3;
    return codec_ptr->decode(output_buffer, width, height, sz, NVPIPE_RGB);
}

static nvp_err_t
nvp_ffmpeg_bitrate(nvpipe* cdc, uint64_t br) {
    nvpipe_* codec = static_cast<nvpipe_*>(cdc);
    codec->codec_ptr_->setBitrate(static_cast<int64_t>(br));
    return NVPIPE_SUCCESS;
}

nvp_impl_t*
nvp_create_ffmpeg(bool nvidia, uint64_t bitrate) {
    const int64_t api_bitrate = static_cast<int64_t>(bitrate);
    if(api_bitrate < 0) {
        return NULL;
    }

    nvpipe_* rv = (nvpipe_*) calloc(sizeof(nvpipe_), 1);
    rv->impl.type = FFMPEG;
    rv->impl.encode = nvp_ffmpeg_encode;
    rv->impl.bitrate = nvp_ffmpeg_bitrate;
    rv->impl.decode = nvp_ffmpeg_decode;
    rv->impl.destroy = nvp_ffmpeg_destroy;

    rv->codec_ptr_ = new NvPipeCodec264();
    rv->codec_ptr_->setBitrate(bitrate);
    if(nvidia) {
        rv->codec_ptr_->setCodecImplementation(NV_CODEC);
    } else {
        rv->codec_ptr_->setCodecImplementation(FFMPEG_LIBX);
    }

    return (nvp_impl_t*)rv;
}
