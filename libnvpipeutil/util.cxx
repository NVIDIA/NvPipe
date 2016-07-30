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

#include "nvpipe.h"

#include "libnvpipecodec/nvpipecodec.h"

#include "libnvpipecodec/nvpipecodec264.h"

#include "libnvpipeutil/image_format_conversion.h"

nvpipe* nvpipe_create_instance( enum NVPipeCodecID id )
{
    nvpipe* codec;
    switch(id) {
        
    case NVPIPE_CODEC_ID_NULL:
        codec = (nvpipe*) calloc( sizeof(nvpipe), 1 );
        codec->type_ = id;
        codec->codec_ptr_ = NULL;
        break;

    case NVPIPE_CODEC_ID_H264:
        codec = (nvpipe*) calloc( sizeof(nvpipe), 1 );
        codec->type_ = NVPIPE_CODEC_ID_H264;
        codec->codec_ptr_ = new NvPipeCodec264();
        break;

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
    case NVPIPE_CODEC_ID_H264:
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

int nvpipe_encode(  nvpipe *codec, 
                    void *input_buffer,
                    const size_t input_buffer_size,
                    void *output_buffer,
                    size_t* output_buffer_size,
                    const int width,
                    const int height,
                    enum NVPipeImageFormat format) 
{

    if (codec == NULL)
        return -1;
    
    NvPipeCodec *codec_ptr = static_cast<NvPipeCodec*> 
                                (codec->codec_ptr_);

    codec_ptr->setImageSize(width, height, format);
    codec_ptr->setInputFrameBuffer(input_buffer, input_buffer_size);
    codec_ptr->encode(output_buffer, *output_buffer_size);

    return 0;

}

int nvpipe_decode(  nvpipe *codec, 
                    void *input_buffer,
                    size_t input_buffer_size,
                    void *output_buffer,
                    size_t output_buffer_size,
                    int* width,
                    int* height,
                    enum NVPipeImageFormat format)
{
    if (codec == NULL)
        return -1;

    NvPipeCodec *codec_ptr = static_cast<NvPipeCodec*> 
                                (codec->codec_ptr_);

    codec_ptr->setImageSize(*width, *height, format);
    codec_ptr->setInputPacketBuffer(input_buffer, input_buffer_size);
    codec_ptr->decode(output_buffer, *width, *height, output_buffer_size);

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
    case NVPIPE_IMAGE_FORMAT_CONVERSION_ARGB_TO_NV12:
        sourceSize = sizeof(uint8_t)*w*h*4;
        destinationSize = sizeof(uint8_t)*w*h*3/2;
        funcPtr = &launch_CudaARGB2NV12Process;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_RGB_TO_NV12:
        sourceSize = sizeof(uint8_t)*w*h*3;
        destinationSize = sizeof(uint8_t)*w*h*3/2;
        funcPtr = &launch_CudaRGB2NV12Process;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_ARGB:
        sourceSize = sizeof(uint8_t)*w*h*3/2;
        destinationSize = sizeof(uint8_t)*w*h*4;
        funcPtr = &launch_CudaNV12TOARGBProcess;
        break;
    case NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGB:
        sourceSize = sizeof(uint8_t)*w*h*3/2;
        destinationSize = sizeof(uint8_t)*w*h*3;
        funcPtr = &launch_CudaNV12TORGBProcess;
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

int formatConversionAVFrameRGB(AVFrame *frame, void *buffer) {
    
    switch( frame->format ) {
    case AV_PIX_FMT_NV12:
        {
            unsigned int * d_YPtr;
            unsigned int * d_UVPtr;
            unsigned int * d_bufferPtr;
            
            int w = frame->width;
            int h = frame->height;
            
            checkCudaErrors(
                cudaMalloc( (void **) &d_YPtr, sizeof(uint8_t)*w*h));
            checkCudaErrors(
                cudaMalloc( (void **) &d_UVPtr, sizeof(uint8_t)*w*h/2));
            checkCudaErrors(
                cudaMalloc( (void **) &d_bufferPtr, sizeof(uint8_t)*w*h*3));
            
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
                
            checkCudaErrors(cudaFree(d_YPtr));
            checkCudaErrors(cudaFree(d_UVPtr));
            checkCudaErrors(cudaFree(d_bufferPtr));

            return ret;
            break;
        }
    case AV_PIX_FMT_ARGB:
        {
            printf("not supported yet\n");
            break;
        }
    default:
        {
            printf("default?\n");
            break;
        }
    }
    return 0;
}


