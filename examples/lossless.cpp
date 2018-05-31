/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <NvPipe.h>

#include <iostream>
#include <iomanip>
#include <vector>

#include "utils.h"




void test(const uint8_t* data, NvPipe_Format format, uint32_t width, uint32_t height)
{
    uint64_t dataSize = width * height;
    uint64_t dataPitch = width;
    if (format == NVPIPE_UINT4)
    {
        dataSize /= 2;
        dataPitch /= 2;
    }
    else if (format == NVPIPE_UINT16)
    {
        dataSize *= 2;
        dataPitch *= 2;
    }
    else if (format == NVPIPE_UINT32)
    {
        dataSize *= 4;
        dataPitch *= 4;
    }


    Timer timer;

    // Encode
    NvPipe* encoder = NvPipe_CreateEncoder(format, NVPIPE_H264, NVPIPE_LOSSLESS, 0, 0, width, height);
    if (!encoder)
    {
        std::cerr << "Failed to create encoder: " << NvPipe_GetError(NULL) << std::endl;
        return;
    }

    std::vector<uint8_t> buffer(dataSize * 2);
    timer.reset();
    uint64_t size = NvPipe_Encode(encoder, data, dataPitch, buffer.data(), buffer.size(), width, height, false);
    double encodeMs = timer.getElapsedMilliseconds();
    if (0 == size)
    {
        std::cerr << "Encode error: " << NvPipe_GetError(encoder) << std::endl;
        return;
    }

    NvPipe_Destroy(encoder);

    // Decode
    NvPipe* decoder = NvPipe_CreateDecoder(format, NVPIPE_H264, width, height);
    if (!decoder)
    {
        std::cerr << "Failed to create decoder: " << NvPipe_GetError(NULL) << std::endl;
        return;
    }

    std::vector<uint8_t> result(dataSize);
    timer.reset();
    uint64_t r = NvPipe_Decode(decoder, buffer.data(), size, result.data(), width, height);
    double decodeMs = timer.getElapsedMilliseconds();
    if (0 == r)
    {
        std::cerr << "Decode error: " << NvPipe_GetError(decoder) << std::endl;
        return;
    }

    NvPipe_Destroy(decoder);


    // Check
    auto bitpattern = [](uint8_t v) -> std::string
    {
        std::string s;

        for (uint32_t i = 1; i <= 128; i *= 2)
        {
            if (v & i)
                s = "1" + s;
            else
                s = "0" + s;

            if (i == 8)
                s = " " + s;
        }

        return s;
    };

    bool ok = true;
    std::string mismatch;

    for (uint32_t i = 0; i < dataSize; ++i)
    {
        if (data[i] != result[i])
        {
            mismatch = "Byte " + std::to_string(i) + ": " + bitpattern(data[i]) + " -> " + bitpattern(result[i]);
            ok = false;
            break;
        }
    }

    // Print result
    if (format == NVPIPE_UINT4)
        std::cout << " - [as UINT4] ";
    else if (format == NVPIPE_UINT8)
        std::cout << " - [as UINT8] ";
    else if (format == NVPIPE_UINT16)
        std::cout << " - [as UINT16] ";
    else if (format == NVPIPE_UINT32)
        std::cout << " - [as UINT32] ";

    std::cout << std::fixed << std::setprecision(1) << " Size: " << size * 0.001 << " KB, Encode: " << encodeMs << " ms, Decode: " << decodeMs << " ms - ";

    if (ok)
        std::cout << "OK" << std::endl;
    else
        std::cout << "MISMATCH [" << mismatch << "]" << std::endl;
}


int main(int argc, char* argv[])
{
    std::cout << "NvPipe example application: Tests lossless compression of a grayscale integer frame." << std::endl << std::endl;

    uint32_t width = 1024;
    uint32_t height = 1024;

    // UINT 8 test
    {
        std::vector<uint8_t> image(width * height);
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
                image[y * width + x] = (255.0f * x * y) / (width * height) * (y % 100 < 50);

        std::cout << std::fixed << std::setprecision(1) << "Input: " << width << " x " << height << " UINT8 (Raw size: " << (width * height)  * 0.001 << " KB)" << std::endl;
        test(image.data(), NVPIPE_UINT4, width * 2, height);
        test(image.data(), NVPIPE_UINT8, width, height);
        test(image.data(), NVPIPE_UINT16, width / 2, height);
        test(image.data(), NVPIPE_UINT32, width / 4, height);
    }

    std::cout << std::endl;


    // UINT32 test
    {
        std::vector<uint32_t> image(width * height);
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
                image[y * width + x] = (4294967295.0f * x * y) / (width * height) * (y % 100 < 50);

        std::cout << std::fixed << std::setprecision(1) << "Input: " << width << " x " << height << " UINT32 (Raw size: " << (width * height * 4)  * 0.001 << " KB)" << std::endl;
//        test((uint8_t*) image.data(), NVPIPE_UINT4, width * 8, height);
        test((uint8_t*) image.data(), NVPIPE_UINT8, width * 4, height);
        test((uint8_t*) image.data(), NVPIPE_UINT16, width * 2, height);
        test((uint8_t*) image.data(), NVPIPE_UINT32, width, height);
    }


    return 0;
}
