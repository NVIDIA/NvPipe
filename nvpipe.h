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
#ifndef NVPIPE_H_
#define NVPIPE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

/************************************************************
 *     API enums
 *
 ************************************************************/

/*! \brief NvPipe codec enumerator
 *
 * Determines which codec to use for encoding/decoding session
 */
enum NVPipeCodecID {
    NVPIPE_CODEC_ID_NULL,
    NVPIPE_CODEC_ID_H264_HARDWARE,
    NVPIPE_CODEC_ID_H264_SOFTWARE
};

/*! \brief NvPipe image format enumerator
 *
 * Specify input/output image format.
 */
enum NVPipeImageFormat {
    NVPIPE_IMAGE_FORMAT_NULL,
    NVPIPE_IMAGE_FORMAT_RGB,
    NVPIPE_IMAGE_FORMAT_RGBA,
    NVPIPE_IMAGE_FORMAT_NV12
};

/************************************************************
 *     API struct
 *
 ************************************************************/

/*! \brief NvPipe struct
 *
 *  before use:
 *      created through nvpipe_create_instance(enum NVPipeCodecID);
 *  after use:
 *      destroy through nvpipe_destroy_instance(nvpipe *codec);
 */
typedef struct _nvpipe {
    enum NVPipeCodecID type_;
    void *codec_ptr_;
} nvpipe;

/*! \brief NvPipe error handler
 *
 *  error status for nvpipe function calls.
 */
typedef unsigned int NVPipeErrorID;

/************************************************************
 *     API function calls
 *
 ************************************************************/

/*! \brief create nvpipe instance
 *
 *      return a handle to the instance;
 *      used to initiate nvpipe_encode/nvpipe_decode API call
 *
 *  set bitrate used for encoder
 *      bitrate = 0 to use default bitrate calculated using Kush Gauge
 *          motion rank = 4
 *          framerate = 30
 *
 *  Kush Gauge for bitrate calculation (default motion rank = 4):
 *  [image width] x [image height] x [framerate] x [motion rank] x 0.07
 *      [motion rank]:  1 being low motion;
 *                      2 being medium motion;
 *                      4 being high motion;
 *  source:
 *      http://www.adobe.com/content/dam/Adobe/en/devnet/
 */
nvpipe* nvpipe_create_instance( enum NVPipeCodecID id, 
                                uint64_t bitrate=0 );

/*! \brief free nvpipe instance
 *
 *      clean up each instance created by nvpipe_create_instance();
 */
void nvpipe_destroy_instance( nvpipe *codec );

/*! \brief encode/compress images
 *
 *  encode picture / frame to video / packet 
 *      User should provide pointer to both input and output buffer
 * 
 *      return 0 on success, otherwise, return < 0;
 *          error message:
 *              Not enough space, required space will be written to 
 *              output_buffer_size
 * 
 *      Upon success, packet data will be copied to output_buffer.
 *      The packet data size will be written to output_buffer_size.
 */
NVPipeErrorID nvpipe_encode(  
                    // [in] handler to nvpipe instance
                    nvpipe *codec, 
                    // [in] pointer to picture buffer
                    void * const restrict input_buffer,
                    // [in] picture buffer size
                    const size_t input_buffer_size,
                    // [in] pointer to output packet buffer
                    void * const restrict output_buffer,
                    // [in] available packet buffer
                    // [out] packet data size
                    size_t* const restrict output_buffer_size,
                    // [in] picture width/height (in pixels)
                    const int width,                    
                    const int height,
                    // [in] pixel format
                    enum NVPipeImageFormat format
                    );

/*! \brief decode/decompress packets
 *
 *  decode video / packet to picture / frame
 *      User should provide pointer to both input and output buffer
 * 
 *      return 0 on success, otherwise, return < 0;
 *          error message:
 * 
 *      Upon success, picture will be copied to output_buffer.
 *      Retrieved image resolution will be set to width/height.
 */
NVPipeErrorID nvpipe_decode(  
                    // [in] handler to nvpipe instance
                    nvpipe *codec, 
                    // [in] pointer to packet buffer
                    void * const restrict input_buffer,
                    // [in] packet data size
                    const size_t input_buffer_size,
                    // [in] pointer to output picture buffer
                    void * const restrict output_buffer,
                    // [in] available output buffer size
                    size_t output_buffer_size,
                    // [in] expected picture width/height (in pixels)
                    // [out] retrived picture width/height (in pixels)
                    int * restrict width,
                    int * restrict height,
                    // [in] pixel format
                    enum NVPipeImageFormat format
                    );

/*!  /brief Retrieve error message
 *
 *  return error string from error_code.
 */
const char * nvpipe_check_error( NVPipeErrorID error_code );

#ifdef __cplusplus
}
#endif

