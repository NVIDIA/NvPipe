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

#include <stdlib.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/** Codecs usable for the encoding/decoding session */
enum NVPipeCodecID {
	NVPIPE_CODEC_ID_NULL,
	NVPIPE_CODEC_ID_H264_HARDWARE,
	NVPIPE_CODEC_ID_H264_SOFTWARE
};

/** Supported NvPipe image formats. */
enum NVPipeImageFormat {
	NVPIPE_IMAGE_FORMAT_NULL,
	NVPIPE_IMAGE_FORMAT_RGB,
	NVPIPE_IMAGE_FORMAT_RGBA,
	NVPIPE_IMAGE_FORMAT_NV12
};

/** Error codes that library calls can return.  See nvpipe_strerror. */
typedef enum nvpipe_error_codes {
	NVPIPE_SUCCESS = 0,
	NVPIPE_ERR_INVALID_IMAGE_FORMAT,
	NVPIPE_ERR_INVALID_CODEC_ID,
	NVPIPE_ERR_INVALID_NVPIPE_INSTANCE,
	NVPIPE_ERR_INVALID_RESOLUTION,
	NVPIPE_ERR_INVALID_BITRATE,	/* 5 */
	NVPIPE_ERR_INPUT_BUFFER_EMPTY_MEMORY,
	NVPIPE_ERR_OUTPUT_BUFFER_OVERFLOW,
	NVPIPE_ERR_CUDA_ERROR,
	NVPIPE_ERR_FFMPEG_ERROR,
	NVPIPE_ERR_FFMPEG_CAN_NOT_FIND_ENCODER,	/* 10 */
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

typedef void nvpipe;

#ifdef __GNUC__
#	define NVPIPE_VISIBLE __attribute__((visibility("default")))
#else
#	define NVPIPE_VISIBLE /* no visibility attribute */
#endif

/** @fn create nvpipe instance
 *
 *  return 0 on success, otherwise, return < 0;
 *  used to initiate context for nvpipe_encode / nvpipe_decode
 *  API call
 *  @param[in] id codec type, HW or software.
 *  @param[in] bitrate rate to use; 0 specifies intelligent default based on
 *             Kush gauge.
 *
 *  bitrate = 0 to use default bitrate calculated using Kush Gauge
 *    motion rank = 4
 *    framerate = 30
 *
 *  Kush Gauge for bitrate calculation (default motion rank = 4):
 *  [image width] x [image height] x [framerate] x [motion rank] x 0.07
 *      [motion rank]:  1 being low motion;
 *                      2 being medium motion;
 *                      4 being high motion;
 *  source:
 *      http://www.adobe.com/content/dam/Adobe/en/devnet/
 */
NVPIPE_VISIBLE nvpipe*
nvpipe_create(enum NVPipeCodecID id, uint64_t bitrate);


/** \brief free nvpipe instance
 *
 * clean up each instance created by nvpipe_create_instance();
 */
NVPIPE_VISIBLE nvp_err_t
nvpipe_destroy(nvpipe* const __restrict codec);

/** encode/compress images
 *
 * User provides pointers for both input and output buffers.  The output buffer
 * must be large enough to accommodate the compressed data.  Sizing the output
 * buffer may be difficult; the call will return OUTPUT_BUFFER_OVERFLOW to
 * indicate that the user must increase the size of the buffer.  The parameter
 * will also be modified to indicate how much of the output buffer was actually
 * used.
 * 
 * @param[in] codec library instance to use
 * @param[in] ibuf  input buffer to compress
 * @param[in] ibuf_sz number of bytes in the input buffer
 * @param[out] obuf buffer to place compressed data into 
 * @param[in,out] obuf_sz number of bytes available in 'obuf', output is number
 *                        of bytes that were actually filled.
 * @param[in] width number of pixels in X of the input buffer
 * @param[in] height number of pixels in Y of the input buffer
 * @param[in] format the format of ibuf.
 *
 * @return NVPIPE_SUCCESS on success, nonzero on error.
 */
NVPIPE_VISIBLE nvp_err_t
nvpipe_encode(nvpipe * const __restrict codec,
              const void *const __restrict ibuf,
              const size_t ibuf_sz,
              void *const __restrict obuf,
              size_t* const __restrict obuf_sz,
              const int width, const int height,
              enum NVPipeImageFormat format);

/** decode/decompress packets
 *
 * Decode a frame into the given buffer.
 * 
 * @param[in] codec instance variable
 * @param[in] ibuf the compressed frame
 * @param[in] ibuf_sz  the size in bytes of the compressed data
 * @param[out] obuf where the output frame will be written.
 * @param[in] obuf_sz the number of bytes available in 'obuf'.
 * @param[in,out] width expected (in) and actual (out) image width
 * @param[in,out] height expected (in) and actual (out) image height
 * @param[in] format the desired format of the output buffer
 *
 * @return NVPIPE_SUCCESS on success, nonzero on error.
 */
NVPIPE_VISIBLE nvp_err_t
nvpipe_decode(nvpipe* const __restrict codec,
              const void* const __restrict ibuf,
              const size_t ibuf_sz,
              void* const __restrict output_buffer,
              size_t output_buffer_size,
              size_t* const __restrict width,
              size_t* const __restrict height,
              enum NVPipeImageFormat format);

/*! Retrieve human-readable error message for the given error code.  Note that
 * this is a pointer to constant memory that must NOT be freed or manipulated
 * by the memory. */
NVPIPE_VISIBLE const char*
nvpipe_strerror(nvp_err_t error_code);

#  ifdef __cplusplus
}
#  endif
#endif
