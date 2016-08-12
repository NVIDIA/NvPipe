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

#ifdef __cplusplus
extern "C" {
#endif


enum NVPipeCodecID {
    NVPIPE_CODEC_ID_NULL,
    NVPIPE_CODEC_ID_H264
};

enum NVPipeImageFormat {
    NVPIPE_IMAGE_FORMAT_NULL,
    NVPIPE_IMAGE_FORMAT_RGB,
    NVPIPE_IMAGE_FORMAT_ARGB,
    NVPIPE_IMAGE_FORMAT_RGBA,
    NVPIPE_IMAGE_FORMAT_YUV420P,
    NVPIPE_IMAGE_FORMAT_YUV444P,
    NVPIPE_IMAGE_FORMAT_NV12
};

typedef struct _nvpipe {
    enum NVPipeCodecID type_;
    void *codec_ptr_;
} nvpipe;

/************************************************************
 *     API function calls
 *
 ************************************************************/

/*  create nvpipe instance
 *      return a handle to the instance;
 *      used to initiate nvpipe_encode/nvpipe_decode API call
 */
nvpipe* nvpipe_create_instance(enum NVPipeCodecID);

/*  free nvpipe instance
 *      clean up each instance created by nvpipe_create_instance();
 */

void nvpipe_destroy_instance( nvpipe *codec );

/*  encode picture(frame) to video(packet) 
 *      User should provide pointer to both input and output buffer
 * 
 *      return 0 if success, otherwise, return < 0;
 *          error message:
 *              Not enough space, required space will be written to 
 *              output_buffer_size
 * 
 *      Upon success, packet data will be copied to output_buffer.
 *      The packet data size will be written to output_buffer_size.
 */
int nvpipe_encode(  
                    // [in] handler to nvpipe instance
                    nvpipe *codec, 
                    // [in] pointer to picture buffer
                    void *input_buffer,
                    // [in] picture buffer size
                    const size_t input_buffer_size,
                    // [in] pointer to output packet buffer
                    void *output_buffer,
                    // [in] available packet buffer
                    // [out] packet data size
                    size_t* output_buffer_size,
                    // [in] picture width/height (in pixels)
                    const int width,                    
                    const int height,
                    // [in] pixel format
                    enum NVPipeImageFormat format
                    );

/*  decode video(packet) to picture(frame)
 *      User should provide pointer to both input and output buffer
 * 
 *      return 0 if success, otherwise, return < 0;
 *          error message:
 * 
 *      Upon success, picture will be copied to output_buffer.
 *      Retrieved image resolution will be set to width/height.
 */
int nvpipe_decode(  
                    // [in] handler to nvpipe instance
                    nvpipe *codec, 
                    // [in] pointer to packet buffer
                    void *input_buffer,
                    // [in] packet data size
                    const size_t input_buffer_size,
                    // [in] pointer to output picture buffer
                    void *output_buffer,
                    // [in] available output buffer size
                    size_t output_buffer_size,
                    // [in] expected picture width/height (in pixels)
                    // [out] retrived picture width/height (in pixels)
                    int* width,
                    int* height,
                    // [in] pixel format
                    enum NVPipeImageFormat format
                    );

/*  set bitrate used for encoder
 *      bitrate = 0 to enable automatic adjust bitrate
 *          [image width] x [image height] x [framerate] x 0.07
 *
 *  guideline for bitrate adjustment:
 *  [image width] x [image height] x [framerate] x [motion rank] x 0.07
 *      [motion rank]:  1 being low motion;
 *                      2 being medium motion;
 *                      4 being high motion;
 *  source:
 *      http://www.adobe.com/content/dam/Adobe/en/devnet/
 *           video/articles/h264_primer/h264_primer.pdf
 */
int nvpipe_set_bitrate(
                        nvpipe *codec,
                        int64_t bitrate
                        );



#ifdef __cplusplus
}
#endif
