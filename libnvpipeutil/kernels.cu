#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>



__forceinline__  __device__ float clamp(float x, float a, float b)
{
  return max(a, min(b, x));
}

__forceinline__  __device__ float RGBA2Y(uchar4  argb)
{
  return clamp((0.257 * argb.x) + (0.504 * argb.y) + (0.098 * argb.z) + 16,0,255);
}

__forceinline__  __device__ float RGB2Y(uchar3  rgb)
{
  return clamp((0.257 * rgb.x) + (0.504 * rgb.y) + (0.098 * rgb.z) + 16,0,255);
}

__forceinline__  __device__ float ARGB2Y(uchar4  argb)
{
  return clamp((0.257 * argb.y) + (0.504 * argb.z) + (0.098 * argb.w) + 16,0,255);
}

__global__ static void CudaProcessARGB2Y(int w, int h, uchar4 * pARGBImage, unsigned char * pNV12ImageY)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  if(i<w && j<h)
  {
    uchar4 argb=pARGBImage[w*j+i];
    pNV12ImageY[w*j+i]= ARGB2Y(argb);
  }
}

__global__ static void CudaProcessRGB2Y(int w, int h, uchar3 * pRGBImage, unsigned char * pNV12ImageY)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  if(i<w && j<h)
  {
    uchar3 rgb=pRGBImage[w*j+i];
    pNV12ImageY[w*j+i]= RGB2Y(rgb);
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

__forceinline__  __device__ float RGB2U(uchar3  rgb)
{
  return clamp(-(0.148 * rgb.x) - (0.291 * rgb.y) + (0.439 * rgb.z)+128.0,0,255);
}

__forceinline__  __device__ float RGB2V(uchar3  rgb)
{
  return clamp((0.439 * rgb.x) - (0.368 * rgb.y) - (0.0701 * rgb.z)+128.0,0,255);
}


__global__ static void CudaProcessARGB2UV(int w, int h, uchar4 * pARGBImage, unsigned char * pNV12ImageUV)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  unsigned int fi = i*2;//full size image i
  unsigned int fj = j*2;//full size image j
  unsigned int fw = w*2;//full size image w
  unsigned int fh = h*2;//full size image h
  unsigned int u_idx = i*2 + 1 + j*w*2;
  unsigned int v_idx = i*2 + j*w*2;
  if(fi<fw-1 && fj<fh-1)
  {
    uchar4 argb1=pARGBImage[fj*fw+fi];
    uchar4 argb2=pARGBImage[fj*fw+fi+1];
    uchar4 argb3=pARGBImage[(fj+1)*fw+fi];
    uchar4 argb4=pARGBImage[(fj+1)*fw+fi+1];

    float U  = ARGB2U(argb1);
    float U2 = ARGB2U(argb2);
    float U3 = ARGB2U(argb3);
    float U4 = ARGB2U(argb4);

    float V =  ARGB2V(argb1);
    float V2 = ARGB2V(argb2);
    float V3 = ARGB2V(argb3);
    float V4 = ARGB2V(argb4);

    pNV12ImageUV[u_idx] = (U+U2+U3+U4)/4.0;
    pNV12ImageUV[v_idx] = (V+V2+V3+V4)/4.0;
  }
}

__global__ static void CudaProcessRGB2UV(int w, int h, uchar3 * pRGBImage, unsigned char * pNV12ImageUV)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  unsigned int fi = i*2;//full size image i
  unsigned int fj = j*2;//full size image j
  unsigned int fw = w*2;//full size image w
  unsigned int fh = h*2;//full size image h
  unsigned int u_idx = i*2 + 1 + j*w*2;
  unsigned int v_idx = i*2 + j*w*2;
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

__global__ static void CudaProcessNV122ARGB(int w, int h, unsigned char * pNV12Image, uchar4 * pARGBImage)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  if(i<w && j<h)
  {
    unsigned int v_idx = (j/2*w/2 + i/2)*2; // ( j/2*w/2 + i/2 ) * 2;
    unsigned int u_idx = v_idx + 1;
    
    unsigned int offset_idx = w*h;
    unsigned int pixel_idx = j*w+i;
    unsigned int channel = threadIdx.z;
    
    if ( channel == 0 ) {
        pARGBImage[w*j+i].x = 0;
    } else if ( channel == 1) {
        pARGBImage[w*j+i].y = clamp(
                                (pNV12Image[pixel_idx]-16) * 1.164 +
                                0 +
                                (pNV12Image[offset_idx+v_idx]-128) * 1.596
                                , 0, 255);
    } else if ( channel == 2) {
        pARGBImage[w*j+i].z = clamp(
                                (pNV12Image[pixel_idx]-16) * 1.164 +
                                (pNV12Image[offset_idx+u_idx]-128) * -0.392 +
                                (pNV12Image[offset_idx+v_idx]-128) * -0.813
                                , 0, 255);
    } else if ( channel == 3) {
        pARGBImage[w*j+i].w = clamp(
                                (pNV12Image[pixel_idx]-16) * 1.164 +
                                (pNV12Image[offset_idx+u_idx]-128) * 2.017 +
                                0
                                , 0, 255);
    }
  }
}

__global__ static void CudaProcessNV122RGB(int w, int h, unsigned char * pNV12Image, uchar3 * pRGBImage)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  if(i<w && j<h)
  {
    unsigned int v_idx = (j/2*w/2 + i/2)*2; // ( j/2*w/2 + i/2 ) * 2;
    unsigned int u_idx = v_idx + 1;

    unsigned int offset_idx = w*h;
    unsigned int pixel_idx = j*w+i;
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

__global__ static void CudaProcessAVFrameNV122RGB(int w, int h, unsigned char * pYImage, unsigned char * pUVImage, uchar3 * pRGBImage)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
  if(i<w && j<h)
  {
    unsigned int v_idx = (j/2*w/2 + i/2)*2; // ( j/2*w/2 + i/2 ) * 2;
    unsigned int u_idx = v_idx + 1;

    unsigned int pixel_idx = j*w+i;
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


extern "C" 
{
  // I really think the code is actually using RGBA instead of ARGB
  cudaError launch_CudaARGB2NV12Process(int w, int h, CUdeviceptr pARGBImage, CUdeviceptr pNV12Image )
  {
    {
      dim3 dimBlock(16, 16, 1);
      dim3 dimGrid(((w)+dimBlock.x-1)/dimBlock.x, ((h)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessARGB2Y<<<dimGrid, dimBlock>>>(w, h, (uchar4 *)pARGBImage, (unsigned char *)pNV12Image);
    }
    {
      dim3 dimBlock(16, 16, 1);
      dim3 dimGrid(((w/2)+dimBlock.x-1)/dimBlock.x, ((h/2)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessARGB2UV<<<dimGrid, dimBlock>>>(w/2, h/2, (uchar4 *)pARGBImage, ((unsigned char *)pNV12Image)+w*h);
    }
    cudaError err = cudaGetLastError();                                
    return err;
  }

  cudaError launch_CudaNV12TOARGBProcess(int w, int h, CUdeviceptr pNV12Image, CUdeviceptr pARGBImage )
  {
    {
      dim3 dimBlock(16, 16, 4); // might be a bad idea to use 1024 threads per SM;
      dim3 dimGrid(((w)+dimBlock.x-1)/dimBlock.x, ((h)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessNV122ARGB<<<dimGrid, dimBlock>>>(w, h, (unsigned char *)pNV12Image, (uchar4 *)pARGBImage);
    }
    cudaError err = cudaGetLastError();                                
    return err;
  }

  cudaError launch_CudaRGB2NV12Process(int w, int h, CUdeviceptr pRGBImage, CUdeviceptr pNV12Image )
  {
    {
      dim3 dimBlock(16, 16, 1);
      dim3 dimGrid(((w)+dimBlock.x-1)/dimBlock.x, ((h)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessRGB2Y<<<dimGrid, dimBlock>>>(w, h, (uchar3 *)pRGBImage, (unsigned char *)pNV12Image);
    }
    {
      dim3 dimBlock(16, 16, 1);
      dim3 dimGrid(((w/2)+dimBlock.x-1)/dimBlock.x, ((h/2)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessRGB2UV<<<dimGrid, dimBlock>>>(w/2, h/2, (uchar3 *)pRGBImage, ((unsigned char *)pNV12Image)+w*h);
    }
    cudaError err = cudaGetLastError();                                
    return err;
  }

  cudaError launch_CudaNV12TORGBProcess(int w, int h, CUdeviceptr pNV12Image, CUdeviceptr pRGBImage )
  {
    {
      dim3 dimBlock(16, 16, 4);
      dim3 dimGrid(((w)+dimBlock.x-1)/dimBlock.x, ((h)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessNV122RGB<<<dimGrid, dimBlock>>>(w, h, (unsigned char *)pNV12Image, (uchar3 *)pRGBImage);
    }
    cudaError err = cudaGetLastError();                                
    return err;
  }

  cudaError launch_CudaNV12TORGBProcessDualChannel( int w, int h, CUdeviceptr pYPlane, CUdeviceptr pUVPlane, CUdeviceptr pRGBImage) {
    {
      dim3 dimBlock(16, 16, 4);
      dim3 dimGrid(((w)+dimBlock.x-1)/dimBlock.x, ((h)+dimBlock.y-1)/dimBlock.y, 1);
      CudaProcessAVFrameNV122RGB<<<dimGrid, dimBlock>>>(w, h, (unsigned char *)pYPlane, (unsigned char *)pUVPlane, (uchar3 *)pRGBImage);
    }
    cudaError err = cudaGetLastError();                                
    return err;
  }


}
