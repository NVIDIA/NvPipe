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
#include <stdint.h>

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

typedef enum nvpipe_error_codes {
    NVPIPE_SUCCESS=0,
    NVPIPE_ERR_INVALID_IMAGE_FORMAT,
    NVPIPE_ERR_INVALID_CODEC_ID,
    NVPIPE_ERR_INVALID_NVPIPE_INSTANCE,
    NVPIPE_ERR_INVALID_RESOLUTION,
    NVPIPE_ERR_INVALID_BITRATE,
    NVPIPE_ERR_INPUT_BUFFER_EMPTY_MEMORY,
    NVPIPE_ERR_OUTPUT_BUFFER_OVERFLOW,
    NVPIPE_ERR_CUDA_ERROR,
    NVPIPE_ERR_FFMPEG_ERROR,
    NVPIPE_ERR_FFMPEG_CAN_NOT_FIND_ENCODER,
    NVPIPE_ERR_FFMPEG_CAN_NOT_FIND_DECODER,
    NVPIPE_ERR_FFMPEG_CAN_NOT_ALLOCATE_FRAME,
    NVPIPE_ERR_FFMPEG_CAN_NOT_ALLOCATE_CONTEXT,
    NVPIPE_ERR_FFMPEG_CAN_NOT_OPEN_CODEC,
    NVPIPE_ERR_FFMPEG_CAN_NOT_BOUND_FRAME,
    NVPIPE_ERR_FFMPEG_LATENCY_OUTPUT_NOT_READY,
    NVPIPE_ERR_FFMPEG_SEND_FRAME,
    NVPIPE_ERR_FFMPEG_SEND_PACKET,
    NVPIPE_ERR_FFMPEG_RECEIVE_PACKET,
    NVPIPE_ERR_FFMPEG_RECEIVE_FRAME,
    NVPIPE_ERR_UNIDENTIFIED_ERROR_CODE
} nvp_err_t;

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

/************************************************************
 *     API function calls
 *
 ************************************************************/

/*! \brief create nvpipe instance
 *
 *  return 0 on success, otherwise, return < 0;
 *  used to initiate context for nvpipe_encode / nvpipe_decode
 *  API call
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
nvpipe* nvpipe_create_instance(
	// [in] specify codec type
	enum NVPipeCodecID id,
	// [in] average bitrate
	uint64_t bitrate
);

/*! \brief free nvpipe instance
 *
 *      clean up each instance created by nvpipe_create_instance();
 */
nvp_err_t nvpipe_destroy_instance(nvpipe* const __restrict codec);

/*! \brief encode/compress images
 *
 *  encode picture / frame to video / packet 
 *      User should provide pointer to both input and output buffer
 * 
 *      return 0 on success, otherwise, return < 0;
 * 
 *      Upon success, packet data will be copied to output_buffer.
 *      The packet data size will be written to output_buffer_size.
 */
nvp_err_t nvpipe_encode(
                    // [in] handler to nvpipe instance
                    nvpipe* const __restrict codec, 
                    // [in] pointer to picture buffer
                    void* const __restrict input_buffer,
                    // [in] picture buffer size
                    const size_t input_buffer_size,
                    // [in] pointer to output packet buffer
                    void* const __restrict output_buffer,
                    // [in] available packet buffer
                    // [out] packet data size
                    size_t* const __restrict output_buffer_size,
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
nvp_err_t nvpipe_decode(
                    // [in] handler to nvpipe instance
                    nvpipe* const __restrict codec, 
                    // [in] pointer to packet buffer
                    void* const __restrict input_buffer,
                    // [in] packet data size
                    const size_t input_buffer_size,
                    // [in] pointer to output picture buffer
                    void* const __restrict output_buffer,
                    // [in] available output buffer size
                    size_t output_buffer_size,
                    // [in] expected picture width/height (in pixels)
                    // [out] actual picture width/height (in pixels)
                    size_t* const __restrict width,
                    size_t* const __restrict height,
                    // [in] pixel format
                    enum NVPipeImageFormat format
                    );

/*!  /brief Retrieve error message
 *
 *  return error string from error_code.
 */
const char* nvpipe_check_error(nvp_err_t error_code);

#ifdef __cplusplus
}
#endif

#endif //NVPIPE_H_
