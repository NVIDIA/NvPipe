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
#include "libnvpipeutil/format.h"
#include "libnvpipeutil/formatConversionCuda.h"

/***************************************************************
 *  format conversion
 * 
 **************************************************************/

int formatConversionReuseMemory( int w, int h, int align,
            void* source,
            void* destination,
            enum NVPipeImageFormatConversion conversionEnum,
            nvpipeMemGpu2 *mem_gpu2) 
{
    unsigned int * d_sourcePtr;
    unsigned int * d_destinationPtr;

    size_t sourceSize;
    size_t destinationSize;
    size_t largeBufferSize;
    size_t smallBufferSize;
    
    unsigned int ** smallBuffer;
    unsigned int ** largeBuffer;

    int ret = 0;
    conversionFunctionPtr funcPtr = NULL;

    int linesize = std::ceil(((float) w)/align) * align;

    switch ( conversionEnum ) {
    case NVPIPE_IMAGE_FORMAT_CONVERSION_NULL:
        return -1;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGB:
        sourceSize = sizeof(uint8_t)*w*h*3/2;
        destinationSize = sizeof(uint8_t)*linesize*h*3;
        smallBuffer = &d_sourcePtr;
        smallBufferSize = sourceSize;
        largeBuffer = &d_destinationPtr;
        largeBufferSize = destinationSize;
        funcPtr = &launch_CudaNV12TORGBProcess;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_RGB_TO_NV12:
        sourceSize = sizeof(uint8_t)*w*h*3;
        destinationSize = sizeof(uint8_t)*linesize*h*3/2;
        smallBuffer = &d_destinationPtr;
        smallBufferSize = destinationSize;
        largeBuffer = &d_sourcePtr;
        largeBufferSize = sourceSize;
        funcPtr = &launch_CudaRGB2NV12Process;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGBA:
        sourceSize = sizeof(uint8_t)*w*h*3/2;
        destinationSize = sizeof(uint8_t)*linesize*h*4;
        smallBuffer = &d_sourcePtr;
        smallBufferSize = sourceSize;
        largeBuffer = &d_destinationPtr;
        largeBufferSize = destinationSize;
        funcPtr = &launch_CudaNV12TORGBAProcess;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_RGBA_TO_NV12:
        sourceSize = sizeof(uint8_t)*w*h*4;
        destinationSize = sizeof(uint8_t)*linesize*h*3/2;
        smallBuffer = &d_destinationPtr;
        smallBufferSize = destinationSize;
        largeBuffer = &d_sourcePtr;
        largeBufferSize = sourceSize;
        funcPtr = &launch_CudaRGBA2NV12Process;
        break;
    default:
        return -1;
    }

    if (largeBufferSize > mem_gpu2->d_buffer_1_size_ ||
        smallBufferSize > mem_gpu2->d_buffer_2_size_) {
            allocateMemGpu2(mem_gpu2, largeBufferSize, smallBufferSize);
    }
    (*smallBuffer) = mem_gpu2->d_buffer_2_;
    (*largeBuffer) = mem_gpu2->d_buffer_1_;

    checkCudaErrors(
        cudaMemcpy( d_sourcePtr, source, sourceSize, 
                    cudaMemcpyHostToDevice ));

    ret =   (*funcPtr)(w, h, align,
            (CUdeviceptr) d_sourcePtr, (CUdeviceptr) d_destinationPtr);

    checkCudaErrors(
        cudaMemcpy( destination, d_destinationPtr, 
        destinationSize, cudaMemcpyDeviceToHost ));

    return ret;
}

int formatConversionAVFrameRGBReuseMemory( AVFrame *frame, int align,
                                void *buffer,
                                nvpipeMemGpu2 *mem_gpu2) {

    switch( frame->format ) {
    case AV_PIX_FMT_NV12:
        {
            unsigned int * d_YPtr;
            unsigned int * d_UVPtr;
            unsigned int * d_bufferPtr;
            
            int w = frame->width;
            int h = frame->height;

            int linesize = std::ceil(((float) w)/align) * align;
            size_t aligned_pixel_count = linesize * h * sizeof(uint8_t);
            size_t pixel_count = w * h * sizeof(uint8_t);

            if (pixel_count*3 > mem_gpu2->d_buffer_1_size_ ||
                pixel_count*3/2 > mem_gpu2->d_buffer_2_size_ ) {
                allocateMemGpu2(mem_gpu2,
                                pixel_count*3,
                                aligned_pixel_count*3/2);
            }

            d_bufferPtr = mem_gpu2->d_buffer_1_;
            d_YPtr = mem_gpu2->d_buffer_2_;

            //Alert!
            //  ugly coade ahead: 
            //  pixel_count/4 because CUDA offset is per word!
            d_UVPtr = mem_gpu2->d_buffer_2_ + aligned_pixel_count/4;

            checkCudaErrors(
                cudaMemcpy( d_YPtr, frame->data[0], aligned_pixel_count, 
                            cudaMemcpyHostToDevice ));
            checkCudaErrors(
                cudaMemcpy( d_UVPtr, frame->data[1], aligned_pixel_count/2, 
                            cudaMemcpyHostToDevice ));

            int ret =   launch_CudaNV12TORGBProcessDualChannel(
                    frame->width, frame->height, align,
                    (CUdeviceptr) d_YPtr,
                    (CUdeviceptr) d_UVPtr,
                    (CUdeviceptr) d_bufferPtr);

            checkCudaErrors(
                cudaMemcpy( buffer, d_bufferPtr, 
                pixel_count*3, cudaMemcpyDeviceToHost ));

            return ret;
            break;
        }
    case AV_PIX_FMT_RGBA:
        {
            printf("formatConversionAVFrameRGBReuseMemory AVFrame frame->format not supported yet\n");
            break;
        }

    case AV_PIX_FMT_ARGB:
        {
            printf("formatConversionAVFrameRGBReuseMemory AVFrame frame->format not supported yet\n");
            break;
        }
    default:
        {
            printf("formatConversionAVFrameRGBReuseMemory AVFrame frame->format default?\n");
            break;
        }
    }
    return 0;
}
int formatConversionAVFrameRGBAReuseMemory( AVFrame *frame, int align,
                                void *buffer,
                                nvpipeMemGpu2 *mem_gpu2) {

    switch( frame->format ) {
    case AV_PIX_FMT_NV12:
        {
            unsigned int * d_YPtr;
            unsigned int * d_UVPtr;
            unsigned int * d_bufferPtr;
            
            int w = frame->width;
            int h = frame->height;

            int linesize = std::ceil(((float) w)/align) * align;
            size_t aligned_pixel_count = linesize * h * sizeof(uint8_t);
            size_t pixel_count = w * h * sizeof(uint8_t);

            if (pixel_count*4 > mem_gpu2->d_buffer_1_size_ ||
                pixel_count*3/2 > mem_gpu2->d_buffer_2_size_ ) {
                allocateMemGpu2(mem_gpu2,
                                pixel_count*4,
                                aligned_pixel_count*3/2);
            }

            d_bufferPtr = mem_gpu2->d_buffer_1_;
            d_YPtr = mem_gpu2->d_buffer_2_;

            //Alert!
            //  ugly coade ahead: 
            //  pixel_count/4 because CUDA offset is per word!
            d_UVPtr = mem_gpu2->d_buffer_2_ + aligned_pixel_count/4;

            checkCudaErrors(
                cudaMemcpy( d_YPtr, frame->data[0], aligned_pixel_count,
                            cudaMemcpyHostToDevice ));
            checkCudaErrors(
                cudaMemcpy( d_UVPtr, frame->data[1], aligned_pixel_count/2,
                            cudaMemcpyHostToDevice ));
            
            int ret =   launch_CudaNV12TORGBAProcessDualChannel(
                    frame->width, frame->height, align,
                    (CUdeviceptr) d_YPtr,
                    (CUdeviceptr) d_UVPtr,
                    (CUdeviceptr) d_bufferPtr);

            checkCudaErrors(
                cudaMemcpy( buffer, d_bufferPtr,
                pixel_count*4, cudaMemcpyDeviceToHost ));

            return ret;
            break;
        }
    case AV_PIX_FMT_RGBA:
        {
            printf("formatConversionAVFrameRGBAReuseMemory AVFrame frame->format not supported yet\n");
            break;
        }

    case AV_PIX_FMT_ARGB:
        {
            printf("formatConversionAVFrameRGBAReuseMemory AVFrame frame->format not supported yet\n");
            break;
        }
    // hacking starts here:
    case AV_PIX_FMT_YUV420P:
        {
            unsigned int * d_YPtr;
            unsigned int * d_UPtr;
            unsigned int * d_VPtr;
            unsigned int * d_bufferPtr;
            
            int w = frame->width;
            int h = frame->height;

            int linesize = std::ceil(((float) w)/align) * align;
            size_t aligned_pixel_count = linesize * h * sizeof(uint8_t);
            size_t pixel_count = w * h * sizeof(uint8_t);

            if (pixel_count*4 > mem_gpu2->d_buffer_1_size_ ||
                pixel_count*3/2 > mem_gpu2->d_buffer_2_size_ ) {
                allocateMemGpu2(mem_gpu2,
                                pixel_count*4,
                                aligned_pixel_count*3/2);
            }

            d_bufferPtr = mem_gpu2->d_buffer_1_;
            d_YPtr = mem_gpu2->d_buffer_2_;

            //Alert!
            //  ugly coade ahead: 
            //  pixel_count/4 because CUDA offset is per word!
            d_UPtr = mem_gpu2->d_buffer_2_ + aligned_pixel_count/4;
            d_VPtr = d_UPtr + aligned_pixel_count/16;

            checkCudaErrors(
                cudaMemcpy( d_YPtr, frame->data[0], aligned_pixel_count, 
                            cudaMemcpyHostToDevice ));
            checkCudaErrors(
                cudaMemcpy( d_UPtr, frame->data[1], aligned_pixel_count/4, 
                            cudaMemcpyHostToDevice ));
            checkCudaErrors(
                cudaMemcpy( d_VPtr, frame->data[2], aligned_pixel_count/4, 
                            cudaMemcpyHostToDevice ));

            int ret =   launch_CudaYUV420PTORGBAProcessTriChannel(
                    frame->width, frame->height, align,
                    (CUdeviceptr) d_YPtr,
                    (CUdeviceptr) d_UPtr,
                    (CUdeviceptr) d_VPtr,
                    (CUdeviceptr) d_bufferPtr);

            checkCudaErrors(
                cudaMemcpy( buffer, d_bufferPtr, 
                pixel_count*4, cudaMemcpyDeviceToHost ));

            return ret;
            break;
        }
    default:
        {
            printf("formatConversionAVFrameRGBAReuseMemory AVFrame frame->format default: %d?\n", frame->format);
            break;
        }
    }
    return 0;
}

/***************************************************************
 *  GPU memory allocation
 * 
 **************************************************************/

void destroyMemGpu2(nvpipeMemGpu2 *mem_gpu) {
    if ( mem_gpu->d_buffer_1_ ) {
        mem_gpu->d_buffer_1_size_ = 0;
        checkCudaErrors(cudaFree(mem_gpu->d_buffer_1_));
    }
    if ( mem_gpu->d_buffer_2_ ) {
        mem_gpu->d_buffer_2_size_ = 0;
        checkCudaErrors(cudaFree(mem_gpu->d_buffer_2_));
    }
}

void allocateMemGpu2(   nvpipeMemGpu2 *mem_gpu,
                        size_t size_1, size_t size_2) {
    destroyMemGpu2(mem_gpu);

    checkCudaErrors(
        cudaMalloc( (void **) &(mem_gpu->d_buffer_1_), size_1));
    mem_gpu->d_buffer_1_size_ = size_1;

    checkCudaErrors(
        cudaMalloc( (void **) &(mem_gpu->d_buffer_2_), size_2));
    mem_gpu->d_buffer_2_size_ = size_2;
}

void initializeMemGpu2(nvpipeMemGpu2 *mem_gpu) {
    mem_gpu->d_buffer_1_size_= 0;
    mem_gpu->d_buffer_2_size_= 0;
    mem_gpu->d_buffer_1_ = NULL;
    mem_gpu->d_buffer_2_ = NULL;
}
