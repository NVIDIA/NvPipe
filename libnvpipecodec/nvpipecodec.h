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

#include "libnvpipeutil/format.h"
#include <cstdlib>


class NvPipeCodec {
public:
    virtual void setImageSize(int width, int height, enum NVPipeImageFormat format);
    
    virtual void setInputFrameBuffer(void* frame_buffer, size_t buffer_size);

    virtual void setInputPacketBuffer(void* packet_buffer, size_t buffer_size);

    virtual int encode(void* frame, size_t &size)=0;
    
    virtual int decode(void* packet, int &width, int &height, size_t &size)=0; 

protected:
    NvPipeCodec();
    virtual ~NvPipeCodec();

    int width_;
    int height_;
    enum NVPipeImageFormat format_;

    void* frame_;
    size_t frame_buffer_size_;

    void* packet_;
    size_t packet_buffer_size_;

private:

};
