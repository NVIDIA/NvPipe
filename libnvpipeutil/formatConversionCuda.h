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
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/************************************************************
 *     launch CUDA kernels for format conversion
 *
 ************************************************************/

typedef cudaError (*conversionFunctionPtr) (
    int, int, CUdeviceptr, CUdeviceptr);

cudaError launch_CudaARGB2NV12Process(  int w, int h, 
                                        CUdeviceptr pARGBImage, 
                                        CUdeviceptr pNV12Image);

cudaError launch_CudaNV12TOARGBProcess( int w, int h, 
                                        CUdeviceptr pNV12Image, 
                                        CUdeviceptr pARGBImage);

cudaError launch_CudaRGBA2NV12Process(  int w, int h, 
                                        CUdeviceptr pARGBImage, 
                                        CUdeviceptr pNV12Image);

cudaError launch_CudaNV12TORGBAProcess( int w, int h, 
                                        CUdeviceptr pNV12Image, 
                                        CUdeviceptr pARGBImage);

cudaError launch_CudaRGB2NV12Process(   int w, int h, 
                                        CUdeviceptr pRGBImage, 
                                        CUdeviceptr pNV12Image);

cudaError launch_CudaNV12TORGBProcess(  int w, int h, 
                                        CUdeviceptr pNV12Image, 
                                        CUdeviceptr pRGBImage);

cudaError launch_CudaNV12TORGBProcessDualChannel( int w, int h,
                                        CUdeviceptr pYPlane,
                                        CUdeviceptr pUVPlane,
                                        CUdeviceptr pRGBImage);

cudaError launch_CudaNV12TORGBAProcessDualChannel( int w, int h,
                                        CUdeviceptr pYPlane,
                                        CUdeviceptr pUVPlane,
                                        CUdeviceptr pRGBAImage);

cudaError launch_CudaYUV420PTORGBAProcessTriChannel( int w, int h,
                                        CUdeviceptr pYPlane,
                                        CUdeviceptr pUPlane,
                                        CUdeviceptr pVPlane,
                                        CUdeviceptr pRGBAImage);

#ifdef __cplusplus
}
#endif
