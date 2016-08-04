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

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef cudaError (*conversionFunctionPtr) (
    int, int, CUdeviceptr, CUdeviceptr);

cudaError launch_CudaARGB2NV12Process(  int w, int h, 
                                        CUdeviceptr pARGBImage, 
                                        CUdeviceptr pNV12Image);

cudaError launch_CudaNV12TOARGBProcess(  int w, int h, 
                                        CUdeviceptr pNV12Image, 
                                        CUdeviceptr pARGBImage);

cudaError launch_CudaRGB2NV12Process(  int w, int h, 
                                        CUdeviceptr pRGBImage, 
                                        CUdeviceptr pNV12Image);

cudaError launch_CudaNV12TORGBProcess(  int w, int h, 
                                        CUdeviceptr pNV12Image, 
                                        CUdeviceptr pRGBImage);

cudaError launch_CudaNV12TORGBProcessDualChannel( int w, int h,
                                        CUdeviceptr pYPlane,
                                        CUdeviceptr pUVPlane,
                                        CUdeviceptr pRGBImage);

#ifdef __cplusplus
}
#endif
