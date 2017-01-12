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
#include <cstddef>
#include <cinttypes>
#include <device_functions.h>

static inline __device__ float
clamp(const float v, const float low, const float high) {
	return v < low ? low : v > high ? high : v;
}

static inline __device__ float
rgb2y(const uint8_t r, const uint8_t g, const uint8_t b) {
	return 0.299f*(float)r + 0.587f*(float)g + 0.114f*(float)b;
}
static inline __device__ float
rgb2u(const uint8_t r, const uint8_t g, const uint8_t b) {
	const float y = rgb2y(r,g,b);
	return clamp(-(((-(float)b + y) / 1.732446f) - 128.f), 0.f, 255.f);
}
static inline __device__ float
rgb2v(const uint8_t r, const uint8_t g, const uint8_t b) {
	const float y = rgb2y(r,g,b);
	const float u = rgb2u(r,g,b);
	return clamp((y - (0.337633f*(u-128.f)) - (float)g) / 0.698001f + 128.f,
	             0.f, 255.f);
}

/* Converts from RGB data to NV12.  NV12's "U" and "V" channels are interleaved
 * and subsampled 2x2.  Note the RGB data are not pitched. */
extern "C" __global__ void
rgb2yuv(const uint8_t* const __restrict rgb,
        const size_t width, const size_t height, const size_t c/*omponents*/,
        uint8_t* const __restrict yuv, unsigned pitch) {
	const size_t x = blockIdx.x*blockDim.x + threadIdx.x;
	const size_t y = blockIdx.y*blockDim.y + threadIdx.y;
	const size_t i = y*pitch + x;
	const size_t j = y*width + x;
	if(x >= width || y >= height || i >= pitch*height || j >= width*height) {
		return;
	}
	assert(pitch >= width);
	assert(i < pitch*height);
	assert(j < width*height);
	assert(width <= 4096);
	assert(height <= 4096);
	assert(c == 3 || c == 4);
	assert(pitch <= 4096);
	uint8_t* __restrict Y = yuv;
	Y[i] = (uint8_t)clamp(rgb2y(rgb[j*c+0], rgb[j*c+1], rgb[j*c+2]), 0, 255);
	/* U+V are downsampled 2x per dimension.  So kill off 3 of every 4 threads
	 * that reach here; only one will do the writes into U and V. */
	/* thought: use x0 to write into U and x1 to write into V, to spread load? */
	if(x&1 == 1 || y&1 == 1) {
		return;
	}
	uint8_t* __restrict uv = yuv + pitch*height;
	const size_t uvidx = y/2*(pitch/2) + x/2;
	const size_t idx[4] = {
		min((y+0)*width + x+0, width*height),
		min((y+0)*width + x+1, width*height),
		min((y+1)*width + x+0, width*height),
		min((y+1)*width + x+1, width*height),
	};
	const float u[4] = {
		rgb2u(rgb[idx[0]*c+0], rgb[idx[0]*c+1], rgb[idx[0]*c+2]),
		rgb2u(rgb[idx[1]*c+0], rgb[idx[1]*c+1], rgb[idx[1]*c+2]),
		rgb2u(rgb[idx[2]*c+0], rgb[idx[2]*c+1], rgb[idx[2]*c+2]),
		rgb2u(rgb[idx[3]*c+0], rgb[idx[3]*c+1], rgb[idx[3]*c+2])
	};
	const float v[4] = {
		rgb2v(rgb[idx[0]*c+0], rgb[idx[0]*c+1], rgb[idx[0]*c+2]),
		rgb2v(rgb[idx[1]*c+0], rgb[idx[1]*c+1], rgb[idx[1]*c+2]),
		rgb2v(rgb[idx[2]*c+0], rgb[idx[2]*c+1], rgb[idx[2]*c+2]),
		rgb2v(rgb[idx[3]*c+0], rgb[idx[3]*c+1], rgb[idx[3]*c+2])
	};
	uv[uvidx*2+0] = (uint8_t)clamp((u[0] + u[1] + u[2] + u[3]) / 4.0, 0, 255);
	uv[uvidx*2+1] = (uint8_t)clamp((v[0] + v[1] + v[2] + v[3]) / 4.0, 0, 255);
}

static inline __device__ float
yuv2r(const uint8_t y, const uint8_t u, const uint8_t v) {
	(void)u;
	return (y-16)*1.164f + (1.596f * (v-128));
}
static inline __device__ float
yuv2g(const uint8_t y, const uint8_t u, const uint8_t v) {
	return (y-16)*1.164f + (u-128)*-0.392f + (v-128)*-0.813f;
}
static inline __device__ float
yuv2b(const uint8_t y, const uint8_t u, const uint8_t v) {
	(void)v;
	return (y-16)*1.164 + (u-128)*2.017;
}

/* Convert back from NV12 to RGB.   Note the RGB buffer is not pitched. */
extern "C" __global__ void
yuv2rgb(const uint8_t* const __restrict yuv,
        const size_t width, const size_t height, unsigned pitch,
        uint8_t* const __restrict rgb) {
	const size_t x = blockIdx.x*blockDim.x + threadIdx.x;
	const size_t y = blockIdx.y*blockDim.y + threadIdx.y;
	const size_t i = y*pitch + x;
	const size_t j = y*width + x;
	if(x >= width || y >= height || i >= pitch*height || j >= width*height) {
		return;
	}
	assert(i < pitch*height);
	assert(j < width*height);
	assert(width <= 4096);
	assert(height <= 4096);
	assert(pitch <= 4096);
	const uint8_t* __restrict Y = yuv;
	const uint8_t* __restrict uv = yuv + pitch*height;
	const size_t idx[4] = {
		min((y+0)/2*pitch/2 + (x+0)/2, height*pitch),
		min((y+0)/2*pitch/2 + (x+1)/2, height*pitch),
		min((y+1)/2*pitch/2 + (x+0)/2, height*pitch),
		min((y+1)/2*pitch/2 + (x+1)/2, height*pitch),
	};
	const uint8_t u[4] = {
		uv[idx[0]*2+0], uv[idx[1]*2+0], uv[idx[2]*2+0], uv[idx[3]*2+0]
	};
	const uint8_t v[4] = {
		uv[idx[0]*2+1], uv[idx[1]*2+1], uv[idx[2]*2+1], uv[idx[3]*2+1]
	};
	rgb[j*3+0] = clamp(
		(yuv2r(Y[i], u[0], v[0]) + yuv2r(Y[i], u[1], v[1]) +
		 yuv2r(Y[i], u[2], v[2]) + yuv2r(Y[i], u[3], v[3])) / 4.0, 0, 255
	);
	rgb[j*3+1] = clamp(
		(yuv2g(Y[i], u[0], v[0]) + yuv2g(Y[i], u[1], v[1]) +
		 yuv2g(Y[i], u[2], v[2]) + yuv2g(Y[i], u[3], v[3])) / 4.0, 0, 255
	);
	rgb[j*3+2] = clamp(
		(yuv2b(Y[i], u[0], v[0]) + yuv2b(Y[i], u[1], v[1]) +
		 yuv2b(Y[i], u[2], v[2]) + yuv2b(Y[i], u[3], v[3])) / 4.0, 0, 255
	);
}
