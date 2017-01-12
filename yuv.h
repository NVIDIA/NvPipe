/*
 * Copyright (c) 2016 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual
 * property and proprietary rights in and to this software,
 * related documentation and any modifications thereto.  Any use,
 * reproduction, disclosure or distribution of this software and
 * related documentation without an express license agreement from
 * NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef NVPIPE_YUV_H
#define NVPIPE_YUV_H

#include <stdbool.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef CUresult (fut_fqn_submit)(void* obj, const CUdeviceptr src,
                                  size_t width, size_t height, CUdeviceptr dst,
                                  unsigned pitch);
typedef CUresult (fut_fqn_sync)(void*);
typedef CUstream (fut_fqn_stream)(const void*);
typedef void (fut_fqn_destroy)(void*);

/** Future abstraction for data reorganization/conversion.
 * Usage:
 *   1. create the future via some creation function
 *   2. submit() its workload.  similar to 'execute' in a vis pipeline.
 *   3. use 'strm' to enqueue your own post-workload GPU work
 *   4. sync() when you need the result.
 *   5. goto 2 to have it reorganize another frame.
 *   6. destroy(self) when you are done to clean everything up. */
typedef struct cu_convert_future {
	/** Converts a WIDTHxHEIGHT image between nv12 and RGB[A] formats.
	 * NV12 organization is WxH bytes of the Y channel, followed by 2*(W/2xH/2)
	 * bytes of the interleaved U and V channels.
	 * @param obj future you are submitting this into
	 * @param src the nv12 or RGB[A] data you wish to convert
	 * @param width the width of the input and output image
	 * @param height the height of the input and output image
	 * @param pitch the pitch for nv12 memory. RGB[A] memory is unpitched!
	 * @param dst the output buffer. */
	fut_fqn_submit* submit;
	/** Synchronize the stream used to submit work.  You must call this before
	 * using the 'dst' output of submit().
	 * This is just cuStreamSynchronize.  You could submit() multiple frames
	 * before sync()ing, if desired. */
	fut_fqn_sync* sync;
	/** Clean up internal resources. */
	fut_fqn_destroy* destroy;
	/** The stream work will be submitted under. */
	CUstream strm;
} nv_fut_t;

/** a future that reorganizes RGB[A] data into nv12 data. */
nv_fut_t* rgb2nv12(size_t components);
/** a future that reorganizes nv12 data into RGB data. */
nv_fut_t* nv122rgb();

#ifdef __cplusplus
}
#endif

#endif
