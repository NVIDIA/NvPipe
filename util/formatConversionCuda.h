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
#ifndef FORMAT_CONVERSION_CUDA_H_
#define FORMAT_CONVERSION_CUDA_H_

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/************************************************************
 *     launch CUDA kernels for format conversion
 *
 ************************************************************/

typedef cudaError (*conversionFunctionPtr) (
    int, int, int, CUdeviceptr, CUdeviceptr);

cudaError launch_CudaRGB2NV12Process(   int w, int h, int align,
                                        CUdeviceptr pRGBImage, 
                                        CUdeviceptr pNV12Image);

cudaError launch_CudaNV12TORGBProcessDualChannel( int w, int h, int align,
                                        CUdeviceptr pYPlane,
                                        CUdeviceptr pUVPlane,
                                        CUdeviceptr pRGBImage);

cudaError launch_CudaRGBA2NV12Process(  int w, int h, int align,
                                        CUdeviceptr pARGBImage, 
                                        CUdeviceptr pNV12Image);

cudaError launch_CudaNV12TORGBAProcessDualChannel( int w, int h, int align,
                                        CUdeviceptr pYPlane,
                                        CUdeviceptr pUVPlane,
                                        CUdeviceptr pRGBAImage);

cudaError launch_CudaYUV420PTORGBAProcessTriChannel( int w, int h, int align,
                                        CUdeviceptr pYPlane,
                                        CUdeviceptr pUPlane,
                                        CUdeviceptr pVPlane,
                                        CUdeviceptr pRGBAImage);

/************************************************************
 *     NOT used any more
 *
 ************************************************************/
cudaError launch_CudaNV12TORGBProcess(  int w, int h, int align,
                                        CUdeviceptr pNV12Image, 
                                        CUdeviceptr pRGBImage);

cudaError launch_CudaNV12TORGBAProcess( int w, int h, int align,
                                        CUdeviceptr pNV12Image, 
                                        CUdeviceptr pARGBImage);
#ifdef __cplusplus
}
#endif

#endif //FORMAT_CONVERSION_CUDA_H_
