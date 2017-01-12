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
#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__forceinline__  __device__ float clamp(float x, float a, float b)
{
  return max(a, min(b, x));
}

__forceinline__  __device__ float RGBA2Y(uchar4  rgba)
{
  return clamp((0.257 * rgba.x) + (0.504 * rgba.y) + (0.098 * rgba.z) + 16,0,255);
}

__forceinline__  __device__ float RGB2Y(uchar3  rgb)
{
  return clamp((0.257 * rgb.x) + (0.504 * rgb.y) + (0.098 * rgb.z) + 16,0,255);
}

__forceinline__  __device__ float ARGB2Y(uchar4  argb)
{
  return clamp((0.257 * argb.y) + (0.504 * argb.z) + (0.098 * argb.w) + 16,0,255);
}

__global__ static void CudaProcessRGBA2Y(int w, int h, int linesize, uchar4 * pRGBAImage, unsigned char * pNV12ImageY)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  if(i<w && j<h)
  {
    uchar4 rgba=pRGBAImage[w*j+i];
    pNV12ImageY[linesize*j+i]= RGBA2Y(rgba);
  }
}

__global__ static void CudaProcessRGB2Y(int w, int h, int linesize, uchar3 * pRGBImage, unsigned char * pNV12ImageY)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  if(i<w && j<h)
  {
    uchar3 rgb=pRGBImage[w*j+i];
    pNV12ImageY[linesize*j+i]= RGB2Y(rgb);
  }
}

__forceinline__  __device__ float ARGB2U(uchar4  argb)
{
  return clamp(-(0.148 * argb.y) - (0.291 * argb.z) + (0.439 * argb.w)+128.0,0,255);
}

__forceinline__  __device__ float ARGB2V(uchar4  argb)
{
  return clamp((0.439 * argb.y) - (0.368 * argb.z) - (0.0701 * argb.w)+128.0,0,255);
}

__forceinline__  __device__ float RGBA2U(uchar4  rgba)
{
  return clamp(-(0.148 * rgba.x) - (0.291 * rgba.y) + (0.439 * rgba.z)+128.0,0,255);
}

__forceinline__  __device__ float RGBA2V(uchar4  rgba)
{
  return clamp((0.439 * rgba.x) - (0.368 * rgba.y) - (0.0701 * rgba.z)+128.0,0,255);
}


__forceinline__  __device__ float RGB2U(uchar3  rgb)
{
  return clamp(-(0.148 * rgb.x) - (0.291 * rgb.y) + (0.439 * rgb.z)+128.0,0,255);
}

__forceinline__  __device__ float RGB2V(uchar3  rgb)
{
  return clamp((0.439 * rgb.x) - (0.368 * rgb.y) - (0.0701 * rgb.z)+128.0,0,255);
}

__global__ static void CudaProcessRGBA2UV(int w, int h, int linesize, uchar4 * pRGBAImage, unsigned char * pNV12ImageUV)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  unsigned int fi = i*2;//full size image i
  unsigned int fj = j*2;//full size image j
  unsigned int fw = w*2;//full size image w
  unsigned int fh = h*2;//full size image h
  unsigned int u_idx = i*2 + j*linesize;
  unsigned int v_idx = u_idx + 1;
  if(fi<fw-1 && fj<fh-1)
  {
    uchar4 rgba1=pRGBAImage[fj*fw+fi];
    uchar4 rgba2=pRGBAImage[fj*fw+fi+1];
    uchar4 rgba3=pRGBAImage[(fj+1)*fw+fi];
    uchar4 rgba4=pRGBAImage[(fj+1)*fw+fi+1];

    float U  = RGBA2U(rgba1);
    float U2 = RGBA2U(rgba2);
    float U3 = RGBA2U(rgba3);
    float U4 = RGBA2U(rgba4);

    float V =  RGBA2V(rgba1);
    float V2 = RGBA2V(rgba2);
    float V3 = RGBA2V(rgba3);
    float V4 = RGBA2V(rgba4);

    pNV12ImageUV[u_idx] = (U+U2+U3+U4)/4.0;
    pNV12ImageUV[v_idx] = (V+V2+V3+V4)/4.0;
  }
}


__global__ static void CudaProcessRGB2UV(int w, int h, int linesize, uchar3 * pRGBImage, unsigned char * pNV12ImageUV)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  unsigned int fi = i*2;//full size image i
  unsigned int fj = j*2;//full size image j
  unsigned int fw = w*2;//full size image w
  unsigned int fh = h*2;//full size image h
  unsigned int u_idx = i*2 + j*linesize;
  unsigned int v_idx = i*2 + j*linesize + 1;
  if(fi<fw-1 && fj<fh-1)
  {
    uchar3 rgb1=pRGBImage[fj*fw+fi];
    uchar3 rgb2=pRGBImage[fj*fw+fi+1];
    uchar3 rgb3=pRGBImage[(fj+1)*fw+fi];
    uchar3 rgb4=pRGBImage[(fj+1)*fw+fi+1];

    float U  = RGB2U(rgb1);
    float U2 = RGB2U(rgb2);
    float U3 = RGB2U(rgb3);
    float U4 = RGB2U(rgb4);

    float V =  RGB2V(rgb1);
    float V2 = RGB2V(rgb2);
    float V3 = RGB2V(rgb3);
    float V4 = RGB2V(rgb4);

    pNV12ImageUV[u_idx] = (U+U2+U3+U4)/4.0;
    pNV12ImageUV[v_idx] = (V+V2+V3+V4)/4.0;
  }
}

__global__ static void CudaProcessNV122RGBA(int w, int h, int linesize, unsigned char * pNV12Image, uchar4 * pRGBAImage)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  if(i<w && j<h)
  {
    unsigned int u_idx = (j/2*linesize/2 + i/2)*2;
    unsigned int v_idx = u_idx + 1;
    
    unsigned int offset_idx = linesize*h;
    unsigned int pixel_idx = j*linesize+i;
    unsigned int channel = threadIdx.z;
    
    if ( channel == 0 ) {
        pRGBAImage[w*j+i].w = 255;
    } else if ( channel == 1) {
        pRGBAImage[w*j+i].x = clamp(
                                (pNV12Image[pixel_idx]-16) * 1.164 +
                                0 +
                                (pNV12Image[offset_idx+v_idx]-128) * 1.596
                                , 0, 255);
    } else if ( channel == 2) {
        pRGBAImage[w*j+i].y = clamp(
                                (pNV12Image[pixel_idx]-16) * 1.164 +
                                (pNV12Image[offset_idx+u_idx]-128) * -0.392 +
                                (pNV12Image[offset_idx+v_idx]-128) * -0.813
                                , 0, 255);
    } else if ( channel == 3) {
        pRGBAImage[w*j+i].z = clamp(
                                (pNV12Image[pixel_idx]-16) * 1.164 +
                                (pNV12Image[offset_idx+u_idx]-128) * 2.017 +
                                0
                                , 0, 255);
    }
  }
}


__global__ static void CudaProcessNV122RGB(int w, int h, int linesize, unsigned char * pNV12Image, uchar3 * pRGBImage)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  if(i<w && j<h)
  {
    unsigned int u_idx = (j/2*linesize/2 + i/2)*2;
    unsigned int v_idx = u_idx + 1;
    unsigned int offset_idx = linesize*h;
    unsigned int pixel_idx = j*linesize+i;
    unsigned int channel = threadIdx.z;

    if ( channel == 1) {
        pRGBImage[w*j+i].x = clamp(
                                (pNV12Image[pixel_idx]-16) * 1.164 +
                                0 +
                                (pNV12Image[offset_idx+v_idx]-128) * 1.596
                                , 0, 255);
    } else if ( channel == 2) {
        pRGBImage[w*j+i].y = clamp(
                                (pNV12Image[pixel_idx]-16) * 1.164 +
                                (pNV12Image[offset_idx+u_idx]-128) * -0.392 +
                                (pNV12Image[offset_idx+v_idx]-128) * -0.813
                                , 0, 255);
    } else if ( channel == 3) {
        pRGBImage[w*j+i].z = clamp(
                                (pNV12Image[pixel_idx]-16) * 1.164 +
                                (pNV12Image[offset_idx+u_idx]-128) * 2.017 +
                                0
                                , 0, 255);
    }
  }
}

__global__ static void CudaProcessAVFrameNV122RGB(int w, int h, int linesize, unsigned char * pYImage, unsigned char * pUVImage, uchar3 * pRGBImage)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  if(i<w && j<h)
  {
    //unsigned int u_idx = (j/2*w/2 + i/2)*2;
    unsigned int u_idx = (j/2*linesize/2 + i/2)*2;
    unsigned int v_idx = u_idx + 1;
    unsigned int pixel_idx = j*linesize+i;
    unsigned int channel = threadIdx.z;
    
    if ( channel == 1) {
        pRGBImage[w*j+i].x = clamp(
                                (pYImage[pixel_idx]-16) * 1.164 +
                                0 +
                                (pUVImage[v_idx]-128) * 1.596
                                , 0, 255);
    } else if ( channel == 2) {
        pRGBImage[w*j+i].y = clamp(
                                (pYImage[pixel_idx]-16) * 1.164 +
                                (pUVImage[u_idx]-128) * -0.392 +
                                (pUVImage[v_idx]-128) * -0.813
                                , 0, 255);
    } else if ( channel == 3) {
        pRGBImage[w*j+i].z = clamp(
                                (pYImage[pixel_idx]-16) * 1.164 +
                                (pUVImage[u_idx]-128) * 2.017 +
                                0
                                , 0, 255);
    }
  }
}


__global__ static void CudaProcessAVFrameNV122RGBA(int w, int h, int linesize, unsigned char * pYImage, unsigned char * pUVImage, uchar4 * pRGBAImage)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  if(i<w && j<h)
  {
    unsigned int u_idx = (j/2*linesize/2 + i/2)*2;
    unsigned int v_idx = u_idx + 1;
    unsigned int pixel_idx = j*linesize+i;
    unsigned int channel = threadIdx.z; 
    
    if ( channel == 1) {
        pRGBAImage[w*j+i].x = clamp(
                                (pYImage[pixel_idx]-16) * 1.164 +
                                0 +
                                (pUVImage[v_idx]-128) * 1.596
                                , 0, 255);
    } else if ( channel == 2) {
        pRGBAImage[w*j+i].y = clamp(
                                (pYImage[pixel_idx]-16) * 1.164 +
                                (pUVImage[u_idx]-128) * -0.392 +
                                (pUVImage[v_idx]-128) * -0.813
                                , 0, 255);
    } else if ( channel == 3) {
        pRGBAImage[w*j+i].z = clamp(
                                (pYImage[pixel_idx]-16) * 1.164 +
                                (pUVImage[u_idx]-128) * 2.017 +
                                0
                                , 0, 255);
    } else if ( channel == 4) {
        //pRGBAImage[w*j+i].w = 255;
        pRGBAImage[w*j+i].w = 0;
    }
  }
}

__global__ static void CudaProcessYUV420P2RGBA(int w, int h, int linesize, unsigned char * pYImage, unsigned char * pUImage, unsigned char * pVImage, uchar4 * pRGBAImage)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  if(i<w && j<h)
  {
    unsigned int uv_idx = j/2*linesize/2 + i/2;

    unsigned int pixel_idx = j*linesize+i;
    unsigned int channel = threadIdx.z;
    
    if ( channel == 1) {
        pRGBAImage[w*j+i].x = clamp(
                                (pYImage[pixel_idx]-16) * 1.164 +
                                0 +
                                (pVImage[uv_idx]-128) * 1.596
                                , 0, 255);
    } else if ( channel == 2) {
        pRGBAImage[w*j+i].y = clamp(
                                (pYImage[pixel_idx]-16) * 1.164 +
                                (pUImage[uv_idx]-128) * -0.392 +
                                (pVImage[uv_idx]-128) * -0.813
                                , 0, 255);
    } else if ( channel == 3) {
        pRGBAImage[w*j+i].z = clamp(
                                (pYImage[pixel_idx]-16) * 1.164 +
                                (pUImage[uv_idx]-128) * 2.017 +
                                0
                                , 0, 255);
    } else if ( channel == 4) {
        pRGBAImage[w*j+i].w = 255;
    }
  }
}

extern "C" 
{
  cudaError launch_CudaRGB2NV12Process(int w, int h, int align, CUdeviceptr pRGBImage, CUdeviceptr pNV12Image )
  {
    int linesize = std::ceil(((float) w)/align) * align;
    {
      dim3 dimBlock(16, 16, 1);
      dim3 dimGrid(((w)+dimBlock.x-1)/dimBlock.x, ((h)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessRGB2Y<<<dimGrid, dimBlock>>>(w, h, linesize, (uchar3 *)pRGBImage, (unsigned char *)pNV12Image);
    }
    {
      dim3 dimBlock(16, 16, 1);
      dim3 dimGrid(((w/2)+dimBlock.x-1)/dimBlock.x, ((h/2)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessRGB2UV<<<dimGrid, dimBlock>>>(w/2, h/2, linesize, (uchar3 *)pRGBImage, ((unsigned char *)pNV12Image)+linesize*h);
    }
    cudaError err = cudaGetLastError();                                
    return err;
  }

  cudaError launch_CudaNV12TORGBProcessDualChannel( int w, int h, int align, CUdeviceptr pYPlane, CUdeviceptr pUVPlane, CUdeviceptr pRGBImage) {
    {
      int linesize = std::ceil(((float) w)/align) * align;
      dim3 dimBlock(16, 16, 4);
      dim3 dimGrid(((w)+dimBlock.x-1)/dimBlock.x, ((h)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessAVFrameNV122RGB<<<dimGrid, dimBlock>>>(w, h, linesize, (unsigned char *)pYPlane, (unsigned char *)pUVPlane, (uchar3 *)pRGBImage);
    }
    cudaError err = cudaGetLastError();                                
    return err;
  }

  cudaError launch_CudaRGBA2NV12Process(int w, int h, int align, CUdeviceptr pRGBAImage, CUdeviceptr pNV12Image )
  {
    int linesize = std::ceil(((float) w)/align) * align;
    {
      dim3 dimBlock(16, 16, 1);
      dim3 dimGrid(((w)+dimBlock.x-1)/dimBlock.x, ((h)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessRGBA2Y<<<dimGrid, dimBlock>>>(w, h, linesize, (uchar4 *)pRGBAImage, (unsigned char *)pNV12Image);
    }
    {
      dim3 dimBlock(16, 16, 1);
      dim3 dimGrid(((w/2)+dimBlock.x-1)/dimBlock.x, ((h/2)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessRGBA2UV<<<dimGrid, dimBlock>>>(w/2, h/2, linesize, (uchar4 *)pRGBAImage, ((unsigned char *)pNV12Image)+linesize*h);
    }
    cudaError err = cudaGetLastError();
    return err;
  }

  cudaError launch_CudaNV12TORGBAProcessDualChannel( int w, int h, int align, CUdeviceptr pYPlane, CUdeviceptr pUVPlane, CUdeviceptr pRGBAImage) {
    {
      int linesize = std::ceil(((float) w)/align) * align;
      dim3 dimBlock(16, 16, 4);
      dim3 dimGrid(((w)+dimBlock.x-1)/dimBlock.x, ((h)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessAVFrameNV122RGBA<<<dimGrid, dimBlock>>>(w, h, linesize, (unsigned char *)pYPlane, (unsigned char *)pUVPlane, (uchar4 *)pRGBAImage);
    }
    cudaError err = cudaGetLastError();                                
    return err;
  }


  cudaError launch_CudaYUV420PTORGBAProcessTriChannel( int w, int h, int align, CUdeviceptr pYPlane, CUdeviceptr pUPlane, CUdeviceptr pVPlane, CUdeviceptr pRGBAImage) {
    {
      int linesize = std::ceil(((float) w)/align) * align;
      dim3 dimBlock(16, 16, 4);
      dim3 dimGrid(((w)+dimBlock.x-1)/dimBlock.x, ((h)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessYUV420P2RGBA<<<dimGrid, dimBlock>>>(w, h, linesize, (unsigned char *)pYPlane, (unsigned char *)pUPlane, (unsigned char *)pVPlane, (uchar4 *)pRGBAImage);
    }
    cudaError err = cudaGetLastError();                                
    return err;
  }


/*****************************not used any more**************************************/

  cudaError launch_CudaNV12TORGBAProcess(int w, int h, int align, CUdeviceptr pNV12Image, CUdeviceptr pRGBAImage )
  {
    int linesize = std::ceil(((float) w)/align) * align;
    {
      dim3 dimBlock(16, 16, 4); // might be a bad idea to use 1024 threads per SM;
      dim3 dimGrid(((w)+dimBlock.x-1)/dimBlock.x, ((h)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessNV122RGBA<<<dimGrid, dimBlock>>>(w, h, linesize, (unsigned char *)pNV12Image, (uchar4 *)pRGBAImage);
    }
    cudaError err = cudaGetLastError();                                
    return err;
  }

  cudaError launch_CudaNV12TORGBProcess(int w, int h, int align, CUdeviceptr pNV12Image, CUdeviceptr pRGBImage )
  {
    int linesize = std::ceil(((float) w)/align) * align;
    {
      dim3 dimBlock(16, 16, 4);
      dim3 dimGrid(((w)+dimBlock.x-1)/dimBlock.x, ((h)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessNV122RGB<<<dimGrid, dimBlock>>>(w, h, linesize, (unsigned char *)pNV12Image, (uchar3 *)pRGBImage);
    }
    cudaError err = cudaGetLastError();                                
    return err;
  }
}
