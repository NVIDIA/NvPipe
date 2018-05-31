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

#include <cuda_runtime_api.h>


int main(int argc, char* argv[])
{
    std::cout << "NvPipe example application: Comparison of using host/device memory." << std::endl << std::endl;

    const uint32_t width = 3840;
    const uint32_t height = 2160;

    const NvPipe_Codec codec = NVPIPE_H264;
    const NvPipe_Compression compression = NVPIPE_LOSSY;
    const float bitrateMbps = 32;
    const uint32_t targetFPS = 90;


    std::cout << "Resolution: " << width << " x " << height << std::endl;
    std::cout << "Codec: " << (codec == NVPIPE_H264 ? "H.264" : "HEVC") << std::endl;
    if (compression == NVPIPE_LOSSY)
        std::cout << "Bitrate: " << bitrateMbps << " Mbps @ " << targetFPS << " Hz" << std::endl;


    // Construct dummy frame
    std::vector<uint8_t> rgba(width * height * 4);
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
            rgba[4 * (y * width + x) + 0] = (255.0f * x * y) / (width * height) * (y % 100 < 50);

    savePPM(rgba.data(), width, height, "memory-input.ppm");

    std::cout << "Resolution: " << width << " x " << height << std::endl;


    std::vector<uint8_t> compressed(rgba.size());
    std::vector<uint8_t> decompressed(rgba.size());

    Timer timer;


    // Host memory benchmark
    {
        std::cout << std::endl << "--- Encode from host memory / Decode to host memory ---" << std::endl;
        std::cout << "Frame | Encode (ms) | Decode (ms) | Size (KB)" << std::endl;

        // Create encoder
        NvPipe* encoder = NvPipe_CreateEncoder(NVPIPE_RGBA32, codec, compression, bitrateMbps * 1000 * 1000, targetFPS, width, height);
        if (!encoder)
            std::cerr << "Failed to create encoder: " << NvPipe_GetError(NULL) << std::endl;

        // Create decoder
        NvPipe* decoder = NvPipe_CreateDecoder(NVPIPE_RGBA32, codec, width, height);
        if (!decoder)
            std::cerr << "Failed to create decoder: " << NvPipe_GetError(NULL) << std::endl;

        // A few frames ...
        for (uint32_t i = 0; i < 10; ++i)
        {
            // Encode
            timer.reset();
            uint64_t size = NvPipe_Encode(encoder, rgba.data(), width * 4, compressed.data(), compressed.size(), width, height, false);
            double encodeMs = timer.getElapsedMilliseconds();

            if (0 == size)
                std::cerr << "Encode error: " << NvPipe_GetError(encoder) << std::endl;

            // Decode
            timer.reset();
            uint64_t r = NvPipe_Decode(decoder, compressed.data(), size, decompressed.data(), width, height);
            double decodeMs = timer.getElapsedMilliseconds();

            if (0 == r)
                std::cerr << "Decode error: " << NvPipe_GetError(decoder) << std::endl;

            double sizeKB = size / 1000.0;
            std::cout << std::fixed << std::setprecision(1) << std::setw(5) << i << " | " << std::setw(11) << encodeMs << " | " <<  std::setw(11) << decodeMs << " | " <<  std::setw(8) << sizeKB << std::endl;

            if (i == 9)
                savePPM(decompressed.data(), width, height, "memory-output.ppm");
        }

        // Clean up
        NvPipe_Destroy(encoder);
        NvPipe_Destroy(decoder);
    }

    // Device memory benchmark
    {
        std::cout << std::endl << "--- Encode from device memory / Decode to device memory ---" << std::endl;
        std::cout << "Frame | Encode (ms) | Decode (ms) | Size (KB)" << std::endl;

        // Create encoder
        NvPipe* encoder = NvPipe_CreateEncoder(NVPIPE_RGBA32, codec, compression, bitrateMbps * 1000 * 1000, targetFPS, width, height);
        if (!encoder)
            std::cerr << "Failed to create encoder: " << NvPipe_GetError(NULL) << std::endl;

        // Create decoder
        NvPipe* decoder = NvPipe_CreateDecoder(NVPIPE_RGBA32, codec, width, height);
        if (!decoder)
            std::cerr << "Failed to create decoder: " << NvPipe_GetError(NULL) << std::endl;

        // Allocate device memory and copy input
        void* rgbaDevice;
        cudaMalloc(&rgbaDevice, rgba.size());
        cudaMemcpy(rgbaDevice, rgba.data(), rgba.size(), cudaMemcpyHostToDevice);

        void* decompressedDevice;
        cudaMalloc(&decompressedDevice, rgba.size());

        for (uint32_t i = 0; i < 10; ++i)
        {
            // Encode
            timer.reset();
            uint64_t size = NvPipe_Encode(encoder, rgbaDevice, width * 4, compressed.data(), compressed.size(), width, height, false);
            double encodeMs = timer.getElapsedMilliseconds();

            if (0 == size)
                std::cerr << "Encode error: " << NvPipe_GetError(encoder) << std::endl;

            // Decode
            timer.reset();
            uint64_t r = NvPipe_Decode(decoder, compressed.data(), size, decompressedDevice, width, height);
            double decodeMs = timer.getElapsedMilliseconds();

            if (0 == r)
                std::cerr << "Decode error: " << NvPipe_GetError(decoder) << std::endl;

            double sizeKB = size / 1000.0;
            std::cout << std::fixed << std::setprecision(1) << std::setw(5) << i << " | " << std::setw(11) << encodeMs << " | " <<  std::setw(11) << decodeMs << " | " <<  std::setw(8) << sizeKB << std::endl;
        }

        cudaFree(rgbaDevice);
        cudaFree(decompressedDevice);

        // Clean up
        NvPipe_Destroy(encoder);
        NvPipe_Destroy(decoder);
    }

    return 0;
}
