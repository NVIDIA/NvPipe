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
#include "nvpipe.h"
#include "libnvpipecodec/nvpipecodec.h"
#include "libnvpipecodec/nvpipecodec264.h"
#include "libnvpipeutil/formatConversionCuda.h"

// AJ profiling
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

nvpipe* nvpipe_create_instance( enum NVPipeCodecID id)
{
    nvpipe* codec;
    switch(id) {
    case NVPIPE_CODEC_ID_NULL:
        codec = (nvpipe*) calloc( sizeof(nvpipe), 1 );
        codec->type_ = id;
        codec->codec_ptr_ = NULL;
        break;

    case NVPIPE_CODEC_ID_H264_HARDWARE:
        {
        codec = (nvpipe*) calloc( sizeof(nvpipe), 1 );
        codec->type_ = NVPIPE_CODEC_ID_H264_HARDWARE;
        NvPipeCodec264* ptr = new NvPipeCodec264();
        ptr->setCodecImplementation(NV_CODEC);
        codec->codec_ptr_ = ptr;
        break;
        }
    case NVPIPE_CODEC_ID_H264_SOFTWARE:
        {
        codec = (nvpipe*) calloc( sizeof(nvpipe), 1 );
        codec->type_ = NVPIPE_CODEC_ID_H264_SOFTWARE;
        NvPipeCodec264* ptr = new NvPipeCodec264();
        ptr->setCodecImplementation(FFMPEG_LIBX);
        codec->codec_ptr_ = ptr;
        break;
        }
    default:
        printf("Unrecognised format enumerator id: %d\n", id);
    }

    return codec;
}

void nvpipe_destroy_instance( nvpipe *codec )
{
    if (codec == NULL)
        return;
    
    switch(codec->type_) {
    case NVPIPE_CODEC_ID_NULL:
        memset( codec, 0, sizeof(nvpipe) );
        free(codec);
        break;
    case NVPIPE_CODEC_ID_H264_HARDWARE:
    case NVPIPE_CODEC_ID_H264_SOFTWARE:
        delete (NvPipeCodec264*) codec->codec_ptr_;
        memset( codec, 0, sizeof(nvpipe) );
        free(codec);
        break;
    default:
        printf("Unrecognised format enumerator id: %d\n", codec->type_);
        memset( codec, 0, sizeof(nvpipe) );
        free(codec);
        break;
    }
}

int nvpipe_set_bitrate(nvpipe *codec, int64_t bitrate) {
    if (codec == NULL)
        return -1;

    NvPipeCodec *codec_ptr = static_cast<NvPipeCodec*> 
                                (codec->codec_ptr_);
    codec_ptr->setBitrate(bitrate);

    return 0;
}

int nvpipe_encode(  nvpipe *codec, 
                    void *input_buffer,
                    const size_t input_buffer_size,
                    void *output_buffer,
                    size_t* output_buffer_size,
                    const int width,
                    const int height,
                    enum NVPipeImageFormat format) {
    if (codec == NULL)
        return -1;

    // AJ temporary hacking
    if ( !(width == 1024 && height == 768)  &&
         !(width == 1280 && height == 720)  &&
         !(width == 1920 && height == 1080) &&
         !(width == 4096 && height == 2160)
        ) {
        printf("NvPipe::nvpipe_encode() resolution: %dx%d is not supported\n",
                width, height);
        *output_buffer_size = 0;
        return -1;
    }

    NvPipeCodec *codec_ptr = static_cast<NvPipeCodec*> 
                                (codec->codec_ptr_);

    // AJ profiling
    cudaProfilerStart();
    nvtxRangePushA("encodingSession");

    codec_ptr->setImageSize(width, height);
    codec_ptr->setInputFrameBuffer(input_buffer, input_buffer_size);
    codec_ptr->encode(output_buffer, *output_buffer_size, format);

    // AJ profiling
    nvtxRangePop();
    cudaProfilerStop();

    return 0;

}

int nvpipe_decode(  nvpipe *codec, 
                    void *input_buffer,
                    size_t input_buffer_size,
                    void *output_buffer,
                    size_t output_buffer_size,
                    int* width,
                    int* height,
                    enum NVPipeImageFormat format){
    if (codec == NULL)
        return -1;

    if ( input_buffer_size == 0 ) {
        printf("NvPipe::nvpipe_decode() empty input_buffer\n");
        return -1;
    }

    // AJ temporary hacking
    if ( !(*width == 1024 && *height == 768)  &&
         !(*width == 1280 && *height == 720)  &&
         !(*width == 1920 && *height == 1080) &&
         !(*width == 4096 && *height == 2160)
        ) {
        printf("NvPipe::nvpipe_decode() resolution: %dx%d is not supported\n",
                *width, *height);
        return -1;
    }

    NvPipeCodec *codec_ptr = static_cast<NvPipeCodec*> 
                                (codec->codec_ptr_);

    // AJ profiling
    cudaProfilerStart();
    nvtxRangePushA("decodingSession");

    codec_ptr->setImageSize(*width, *height);
    codec_ptr->setInputPacketBuffer(input_buffer, input_buffer_size);
    codec_ptr->decode(  output_buffer,
                        *width,
                        *height,
                        output_buffer_size,
                        format);
    // AJ profiling
    nvtxRangePop();
    cudaProfilerStop();

    return 0;
}

int formatConversion(   int w, int h,
            void* source,
            void* destination, 
            enum NVPipeImageFormatConversion conversionEnum)
{
    unsigned int * d_sourcePtr;
    unsigned int * d_destinationPtr;

    size_t sourceSize;
    size_t destinationSize;

    int ret = 0;
    conversionFunctionPtr funcPtr = NULL;

    switch ( conversionEnum ) {
    case NVPIPE_IMAGE_FORMAT_CONVERSION_NULL:
        return -1;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_RGB_TO_NV12:
        sourceSize = sizeof(uint8_t)*w*h*3;
        destinationSize = sizeof(uint8_t)*w*h*3/2;
        funcPtr = &launch_CudaRGB2NV12Process;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGB:
        sourceSize = sizeof(uint8_t)*w*h*3/2;
        destinationSize = sizeof(uint8_t)*w*h*3;
        funcPtr = &launch_CudaNV12TORGBProcess;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_RGBA_TO_NV12:
        sourceSize = sizeof(uint8_t)*w*h*4;
        destinationSize = sizeof(uint8_t)*w*h*3/2;
        funcPtr = &launch_CudaRGBA2NV12Process;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGBA:
        sourceSize = sizeof(uint8_t)*w*h*3/2;
        destinationSize = sizeof(uint8_t)*w*h*4;
        funcPtr = &launch_CudaNV12TORGBAProcess;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_ARGB_TO_NV12:
        sourceSize = sizeof(uint8_t)*w*h*4;
        destinationSize = sizeof(uint8_t)*w*h*3/2;
        funcPtr = &launch_CudaARGB2NV12Process;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_ARGB:
        sourceSize = sizeof(uint8_t)*w*h*3/2;
        destinationSize = sizeof(uint8_t)*w*h*4;
        funcPtr = &launch_CudaNV12TOARGBProcess;
        break;
    default:
        return -1;
    }

    checkCudaErrors(
        cudaMalloc( (void **) &d_sourcePtr, sourceSize ));
    checkCudaErrors(
        cudaMalloc( (void **) &d_destinationPtr, destinationSize ));
    checkCudaErrors(
        cudaMemcpy( d_sourcePtr, source, sourceSize, 
                    cudaMemcpyHostToDevice ));
    
    ret =   (*funcPtr)(w, h, 
            (CUdeviceptr) d_sourcePtr, (CUdeviceptr) d_destinationPtr);

    checkCudaErrors(
        cudaMemcpy( destination, d_destinationPtr, 
        destinationSize, cudaMemcpyDeviceToHost ));
    checkCudaErrors(cudaFree(d_sourcePtr));
    checkCudaErrors(cudaFree(d_destinationPtr));

    return ret;
}

int formatConversionReuseMemory(   int w, int h,
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

    switch ( conversionEnum ) {
    case NVPIPE_IMAGE_FORMAT_CONVERSION_NULL:
        return -1;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGB:
        sourceSize = sizeof(uint8_t)*w*h*3/2;
        destinationSize = sizeof(uint8_t)*w*h*3;
        smallBuffer = &d_sourcePtr;
        smallBufferSize = sourceSize;
        largeBuffer = &d_destinationPtr;
        largeBufferSize = destinationSize;
        funcPtr = &launch_CudaNV12TORGBProcess;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_RGB_TO_NV12:
        sourceSize = sizeof(uint8_t)*w*h*3;
        destinationSize = sizeof(uint8_t)*w*h*3/2;
        smallBuffer = &d_destinationPtr;
        smallBufferSize = destinationSize;
        largeBuffer = &d_sourcePtr;
        largeBufferSize = sourceSize;
        funcPtr = &launch_CudaRGB2NV12Process;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGBA:
        sourceSize = sizeof(uint8_t)*w*h*3/2;
        destinationSize = sizeof(uint8_t)*w*h*4;
        smallBuffer = &d_sourcePtr;
        smallBufferSize = sourceSize;
        largeBuffer = &d_destinationPtr;
        largeBufferSize = destinationSize;
        funcPtr = &launch_CudaNV12TORGBAProcess;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_RGBA_TO_NV12:
        sourceSize = sizeof(uint8_t)*w*h*4;
        destinationSize = sizeof(uint8_t)*w*h*3/2;
        smallBuffer = &d_destinationPtr;
        smallBufferSize = destinationSize;
        largeBuffer = &d_sourcePtr;
        largeBufferSize = sourceSize;
        funcPtr = &launch_CudaRGBA2NV12Process;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_ARGB:
        sourceSize = sizeof(uint8_t)*w*h*3/2;
        destinationSize = sizeof(uint8_t)*w*h*4;
        smallBuffer = &d_sourcePtr;
        smallBufferSize = sourceSize;
        largeBuffer = &d_destinationPtr;
        largeBufferSize = destinationSize;
        funcPtr = &launch_CudaNV12TOARGBProcess;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_ARGB_TO_NV12:
        sourceSize = sizeof(uint8_t)*w*h*4;
        destinationSize = sizeof(uint8_t)*w*h*3/2;
        smallBuffer = &d_destinationPtr;
        smallBufferSize = destinationSize;
        largeBuffer = &d_sourcePtr;
        largeBufferSize = sourceSize;
        funcPtr = &launch_CudaARGB2NV12Process;
        break;
    default:
        return -1;
    }
    
    if (largeBufferSize > mem_gpu2->d_buffer_1_size_ ||
        smallBufferSize > mem_gpu2->d_buffer_2_size_) {
            printf("mem reallocate!\n");
        destroyMemGpu2(mem_gpu2);
        checkCudaErrors(
            cudaMalloc( (void **) &(mem_gpu2->d_buffer_1_),
                        largeBufferSize));
        mem_gpu2->d_buffer_1_size_ = largeBufferSize;
        checkCudaErrors(
            cudaMalloc( (void **) &(mem_gpu2->d_buffer_2_),
                        smallBufferSize));
        mem_gpu2->d_buffer_2_size_ = smallBufferSize;
    }
    (*smallBuffer) = mem_gpu2->d_buffer_2_;
    (*largeBuffer) = mem_gpu2->d_buffer_1_;

    checkCudaErrors(
        cudaMemcpy( d_sourcePtr, source, sourceSize, 
                    cudaMemcpyHostToDevice ));

    ret =   (*funcPtr)(w, h, 
            (CUdeviceptr) d_sourcePtr, (CUdeviceptr) d_destinationPtr);

    checkCudaErrors(
        cudaMemcpy( destination, d_destinationPtr, 
        destinationSize, cudaMemcpyDeviceToHost ));

    return ret;
}

int formatConversionAVFrameRGBReuseMemory( AVFrame *frame, 
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

            size_t pixel_count = w * h * sizeof(uint8_t);

            if (pixel_count*3 > mem_gpu2->d_buffer_1_size_ ||
                pixel_count*3/2 > mem_gpu2->d_buffer_2_size_ ) {
                printf("mem reallocate!\n");
                allocateMemGpu2(mem_gpu2,
                                pixel_count*3,
                                pixel_count*3/2);
            }

            d_bufferPtr = mem_gpu2->d_buffer_1_;
            d_YPtr = mem_gpu2->d_buffer_2_;

            //Alert!
            //  ugly coade ahead: 
            //  pixel_count/4 because CUDA offset is per word!
            d_UVPtr = mem_gpu2->d_buffer_2_ + pixel_count/4;

            checkCudaErrors(
                cudaMemcpy( d_YPtr, frame->data[0], sizeof(uint8_t)*w*h, 
                            cudaMemcpyHostToDevice ));
            checkCudaErrors(
                cudaMemcpy( d_UVPtr, frame->data[1], sizeof(uint8_t)*w*h/2, 
                            cudaMemcpyHostToDevice ));

            int ret =   launch_CudaNV12TORGBProcessDualChannel(
                    frame->width, frame->height,
                    (CUdeviceptr) d_YPtr,
                    (CUdeviceptr) d_UVPtr,
                    (CUdeviceptr) d_bufferPtr);

            checkCudaErrors(
                cudaMemcpy( buffer, d_bufferPtr, 
                sizeof(uint8_t)*w*h*3, cudaMemcpyDeviceToHost ));

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

int formatConversionAVFrameRGB( AVFrame *frame, 
                                void *buffer) {

    switch( frame->format ) {
    case AV_PIX_FMT_NV12:
        {
            unsigned int * d_YPtr;
            unsigned int * d_UVPtr;
            unsigned int * d_bufferPtr;
            
            int w = frame->width;
            int h = frame->height;

            size_t pixel_count = w * h * sizeof(uint8_t);
            checkCudaErrors(
                cudaMalloc( (void **) &(d_bufferPtr), pixel_count*3));
            checkCudaErrors(
                cudaMalloc( (void **) &(d_YPtr), pixel_count));
            checkCudaErrors(
                cudaMalloc( (void **) &(d_UVPtr), pixel_count/2));

            checkCudaErrors(
                cudaMemcpy( d_YPtr, frame->data[0], sizeof(uint8_t)*w*h, 
                            cudaMemcpyHostToDevice ));
            checkCudaErrors(
                cudaMemcpy( d_UVPtr, frame->data[1], sizeof(uint8_t)*w*h/2, 
                            cudaMemcpyHostToDevice ));
            
            int ret =   launch_CudaNV12TORGBProcessDualChannel(
                    frame->width, frame->height,
                    (CUdeviceptr) d_YPtr,
                    (CUdeviceptr) d_UVPtr,
                    (CUdeviceptr) d_bufferPtr);

            checkCudaErrors(
                cudaMemcpy( buffer, d_bufferPtr, 
                sizeof(uint8_t)*w*h*3, cudaMemcpyDeviceToHost ));
            
            checkCudaErrors(cudaFree(d_bufferPtr));
            checkCudaErrors(cudaFree(d_YPtr));
            checkCudaErrors(cudaFree(d_UVPtr));
            
            return ret;
            break;
        }
    case AV_PIX_FMT_ARGB:
        {
            printf("formatConversionAVFrameRGB AVFrame frame->format not supported yet\n");
            break;
        }
    default:
        {
            printf("formatConversionAVFrameRGB AVFrame frame->format = default?\n");
            break;
        }
    }
    return 0;
}

int formatConversionAVFrameRGBAReuseMemory( AVFrame *frame, 
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

            size_t pixel_count = w * h * sizeof(uint8_t);

            if (pixel_count*4 > mem_gpu2->d_buffer_1_size_ ||
                pixel_count*3/2 > mem_gpu2->d_buffer_2_size_ ) {
                printf("mem reallocate!\n");
                allocateMemGpu2(mem_gpu2,
                                pixel_count*4,
                                pixel_count*3/2);
            }

            d_bufferPtr = mem_gpu2->d_buffer_1_;
            d_YPtr = mem_gpu2->d_buffer_2_;

            //Alert!
            //  ugly coade ahead: 
            //  pixel_count/4 because CUDA offset is per word!
            d_UVPtr = mem_gpu2->d_buffer_2_ + pixel_count/4;

            checkCudaErrors(
                cudaMemcpy( d_YPtr, frame->data[0], sizeof(uint8_t)*w*h, 
                            cudaMemcpyHostToDevice ));
            checkCudaErrors(
                cudaMemcpy( d_UVPtr, frame->data[1], sizeof(uint8_t)*w*h/2, 
                            cudaMemcpyHostToDevice ));
            
            int ret =   launch_CudaNV12TORGBAProcessDualChannel(
                    frame->width, frame->height,
                    (CUdeviceptr) d_YPtr,
                    (CUdeviceptr) d_UVPtr,
                    (CUdeviceptr) d_bufferPtr);

            checkCudaErrors(
                cudaMemcpy( buffer, d_bufferPtr, 
                sizeof(uint8_t)*w*h*4, cudaMemcpyDeviceToHost ));

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
            /*
            printf("yuv420p\n");
            printf("size: %d\n", frame->linesize[0]);
            printf("size: %d\n", frame->linesize[1]);
            printf("size: %d\n", frame->linesize[2]);
            printf("size: %d\n", frame->linesize[3]);
            printf("data: %p\n", frame->data[0]);
            printf("data: %p\n", frame->data[1]);
            printf("data: %p\n", frame->data[2]);
            printf("data: %p\n", frame->data[3]);
            */
            unsigned int * d_YPtr;
            unsigned int * d_UPtr;
            unsigned int * d_VPtr;
            unsigned int * d_bufferPtr;
            
            int w = frame->width;
            int h = frame->height;

            size_t pixel_count = w * h * sizeof(uint8_t);

            if (pixel_count*4 > mem_gpu2->d_buffer_1_size_ ||
                pixel_count*3/2 > mem_gpu2->d_buffer_2_size_ ) {
                printf("mem reallocate!\n");
                allocateMemGpu2(mem_gpu2,
                                pixel_count*4,
                                pixel_count*3/2);
            }

            d_bufferPtr = mem_gpu2->d_buffer_1_;
            d_YPtr = mem_gpu2->d_buffer_2_;

            //Alert!
            //  ugly coade ahead: 
            //  pixel_count/4 because CUDA offset is per word!
            d_UPtr = mem_gpu2->d_buffer_2_ + pixel_count/4;
            d_VPtr = d_UPtr + pixel_count/16;

            checkCudaErrors(
                cudaMemcpy( d_YPtr, frame->data[0], sizeof(uint8_t)*w*h, 
                            cudaMemcpyHostToDevice ));
            checkCudaErrors(
                cudaMemcpy( d_UPtr, frame->data[1], sizeof(uint8_t)*w*h/4, 
                            cudaMemcpyHostToDevice ));
            checkCudaErrors(
                cudaMemcpy( d_VPtr, frame->data[2], sizeof(uint8_t)*w*h/4, 
                            cudaMemcpyHostToDevice ));

            int ret =   launch_CudaYUV420PTORGBAProcessTriChannel(
                    frame->width, frame->height,
                    (CUdeviceptr) d_YPtr,
                    (CUdeviceptr) d_UPtr,
                    (CUdeviceptr) d_VPtr,
                    (CUdeviceptr) d_bufferPtr);

            checkCudaErrors(
                cudaMemcpy( buffer, d_bufferPtr, 
                sizeof(uint8_t)*w*h*4, cudaMemcpyDeviceToHost ));

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

int formatConversionAVFrameRGBA( AVFrame *frame, 
                                void *buffer) {

    switch( frame->format ) {
    case AV_PIX_FMT_NV12:
        {
            unsigned int * d_YPtr;
            unsigned int * d_UVPtr;
            unsigned int * d_bufferPtr;
            
            int w = frame->width;
            int h = frame->height;

            size_t pixel_count = w * h * sizeof(uint8_t);
            checkCudaErrors(
                cudaMalloc( (void **) &(d_bufferPtr), pixel_count*4));
            checkCudaErrors(
                cudaMalloc( (void **) &(d_YPtr), pixel_count));
            checkCudaErrors(
                cudaMalloc( (void **) &(d_UVPtr), pixel_count/2));

            checkCudaErrors(
                cudaMemcpy( d_YPtr, frame->data[0], sizeof(uint8_t)*w*h, 
                            cudaMemcpyHostToDevice ));
            checkCudaErrors(
                cudaMemcpy( d_UVPtr, frame->data[1], sizeof(uint8_t)*w*h/2, 
                            cudaMemcpyHostToDevice ));
            
            int ret =   launch_CudaNV12TORGBAProcessDualChannel(
                    frame->width, frame->height,
                    (CUdeviceptr) d_YPtr,
                    (CUdeviceptr) d_UVPtr,
                    (CUdeviceptr) d_bufferPtr);

            checkCudaErrors(
                cudaMemcpy( buffer, d_bufferPtr, 
                sizeof(uint8_t)*w*h*4, cudaMemcpyDeviceToHost ));

            checkCudaErrors(cudaFree(d_bufferPtr));
            checkCudaErrors(cudaFree(d_YPtr));
            checkCudaErrors(cudaFree(d_UVPtr));
            return ret;
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
            printf("yuv420p\n");
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
