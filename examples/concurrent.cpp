/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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


#include <thread>
#include <mutex>
#include <string>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include <NvPipe.h>

#include "utils.h"


void run(void* rgba, uint32_t width, uint32_t height, uint32_t index, std::mutex* printMutex)
{
    const NvPipe_Codec codec = NVPIPE_H264;
    const NvPipe_Compression compression = NVPIPE_LOSSY;
    const float bitrateMbps = 4;
    const uint32_t targetFPS = 30;
    const uint32_t numFrames = 500;

    std::vector<uint8_t> compressed(width * height * 4);

    void* decompressedDevice;
    cudaMalloc(&decompressedDevice, width * height * 4);

    // Create encoder
    NvPipe* encoder = NvPipe_CreateEncoder(NVPIPE_RGBA32, codec, compression, bitrateMbps * 1000 * 1000, targetFPS, width, height);
    if (!encoder)
    {
        std::lock_guard<std::mutex> lock(*printMutex);
        std::cerr << "[" << index << "] Failed to create encoder: " << NvPipe_GetError(NULL) << std::endl;
        return;
    }

    // Create decoder
    NvPipe* decoder = NvPipe_CreateDecoder(NVPIPE_RGBA32, codec, width, height);
    if (!decoder)
    {
        std::lock_guard<std::mutex> lock(*printMutex);
        std::cerr << "[" << index << "] Failed to create decoder: " << NvPipe_GetError(NULL) << std::endl;
        return;
    }

    Timer timer;
    double encodeMs = 0.0;
    double decodeMs = 0.0;

    for (uint32_t i = 0; i < numFrames; ++i)
    {
        // Encode
        timer.reset();
        uint64_t size = NvPipe_Encode(encoder, rgba, width * 4, compressed.data(), compressed.size(), width, height, false);
        encodeMs += timer.getElapsedMilliseconds();

        if (0 == size)
        {
            std::lock_guard<std::mutex> lock(*printMutex);
            std::cerr << "[" << index << "] Encode error: " << NvPipe_GetError(encoder) << std::endl;
            return;
        }

        // Decode
        timer.reset();
        uint64_t r = NvPipe_Decode(decoder, compressed.data(), size, decompressedDevice, width, height);
        decodeMs += timer.getElapsedMilliseconds();

        if (0 == r)
        {
            std::lock_guard<std::mutex> lock(*printMutex);
            std::cerr << "[" << index << "] Decode error: " << NvPipe_GetError(decoder) << std::endl;
            return;
        }
    }

    encodeMs /= numFrames;
    decodeMs /= numFrames;

    // Clean up
    NvPipe_Destroy(encoder);
    NvPipe_Destroy(decoder);

    cudaFree(decompressedDevice);

    // Print results
    std::lock_guard<std::mutex> lock(*printMutex);
    std::cout << std::fixed << std::setprecision(1) << "[" << index << "] Average encode: " << encodeMs << " ms / Average decode: " << decodeMs << " ms" << std::endl;
}


int main(int argc, char* argv[])
{
    std::cout << "NvPipe example application: Concurrent encode/decode sessions." << std::endl;

    const uint32_t width = 1920;
    const uint32_t height = 1080;

    const uint32_t numWorkersMax = 4;


    // Allocate dummy frame
    std::vector<uint8_t> rgba(width * height * 4);
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
            rgba[4 * (y * width + x) + 0] = (255.0f * x * y) / (width * height) * (y % 100 < 50);

    // Allocate device memory and copy input
    void* rgbaDevice;
    cudaMalloc(&rgbaDevice, rgba.size());
    cudaMemcpy(rgbaDevice, rgba.data(), rgba.size(), cudaMemcpyHostToDevice);

    // Multiple runs with increasing worker count
    for (uint32_t numWorkers = 1; numWorkers <= numWorkersMax; ++numWorkers)
    {
        std::cout << std::endl;
        std::cout << "--- " << numWorkers << " threads ---" << std::endl;

        std::mutex printMutex;
        std::vector<std::thread> workers;

        for (uint32_t i = 0; i < numWorkers; ++i)
            workers.push_back(std::thread(run, rgbaDevice, width, height, i, &printMutex));

        for (std::thread& t : workers)
            if (t.joinable())
                t.join();
    }

    // Clean up
    cudaFree(rgbaDevice);

    return 0;
}
