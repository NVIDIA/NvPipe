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

#include "utils.h"

#include <iostream>
#include <vector>
#include <fstream>

int main(int argc, char* argv[])
{
    std::cout << "NvPipe example application: Encodes to a file / decodes from a file." << std::endl << "Useful for testing if only encoding or decoding is enabled." << std::endl << std::endl;

    const uint32_t width = 3840;
    const uint32_t height = 2160;

    const NvPipe_Codec codec = NVPIPE_H264;
    const float bitrateMbps = 32;
    const uint32_t targetFPS = 90;


    std::cout << "Resolution: " << width << " x " << height << std::endl;
    std::cout << "Codec: " << (codec == NVPIPE_H264 ? "H.264" : "HEVC") << std::endl;
    std::cout << "Bitrate: " << bitrateMbps << " Mbps @ " << targetFPS << " Hz" << std::endl;

    std::cout << "Resolution: " << width << " x " << height << std::endl;

    Timer timer;

    std::vector<uint8_t> rgba(width * height * 4);
    std::vector<uint8_t> compressed(rgba.size());


    // Encoding
#ifdef NVPIPE_WITH_ENCODER
    // Construct dummy frame
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
            rgba[4 * (y * width + x) + 1] = (255.0f * x* y) / (width * height) * (y % 100 < 50);

    savePPM(rgba.data(), width, height, "file-input.ppm");


    NvPipe* encoder = NvPipe_CreateEncoder(NVPIPE_RGBA32, codec, NVPIPE_LOSSY, bitrateMbps * 1000 * 1000, targetFPS, width, height);
    if (!encoder)
        std::cerr << "Failed to create encoder: " << NvPipe_GetError(NULL) << std::endl;

    std::ofstream out("stream.bin", std::ios::out | std::ios::binary);

    std::cout << std::endl << "Encoding..." << std::endl;

    for (uint32_t i = 0; i < 10; ++i)
    {
        timer.reset();
        uint64_t size = NvPipe_Encode(encoder, rgba.data(), width * 4, compressed.data(), compressed.size(), width, height, false);
        double encodeMs = timer.getElapsedMilliseconds();

        if (0 == size)
            std::cerr << "Encode error: " << NvPipe_GetError(encoder) << std::endl;

        out.write((char*) &size, sizeof(uint64_t));
        out.write((char*) compressed.data(), size);

        std::cout << i << ": " << encodeMs << " ms" << std::endl;
    }

    out.close();

    NvPipe_Destroy(encoder);
#endif


    // Decoding
#ifdef NVPIPE_WITH_DECODER
    NvPipe* decoder = NvPipe_CreateDecoder(NVPIPE_RGBA32, codec, width, height);
    if (!decoder)
        std::cerr << "Failed to create decoder: " << NvPipe_GetError(NULL) << std::endl;

    std::ifstream in("stream.bin", std::ios::in | std::ios::binary);
    if (!in)
    {
        std::cerr << std::endl;
        std::cerr << "Error: Failed to open input file \"stream.bin\"." << std::endl;
        std::cerr << "The file can be created using this example with" << std::endl;
        std::cerr << "NvPipe encoding enabled." << std::endl;
        return 1;
    }

    std::cout << std::endl << "Decoding..." << std::endl;

    for (uint32_t i = 0; i < 10; ++i)
    {
        uint64_t size;
        in.read((char*) &size, sizeof(uint64_t));
        in.read((char*) compressed.data(), size);


        // Decode
        timer.reset();
        uint64_t r = NvPipe_Decode(decoder, compressed.data(), size, rgba.data(), width, height);
        double decodeMs = timer.getElapsedMilliseconds();

        if (r == size)
            std::cerr << "Decode error: " << NvPipe_GetError(decoder) << std::endl;

        std::cout << i << ": " << decodeMs << " ms" << std::endl;

        if (i == 0)
            savePPM(rgba.data(), width, height, "file-output.ppm");
    }

    in.close();

    NvPipe_Destroy(decoder);
#endif

    return 0;
}
