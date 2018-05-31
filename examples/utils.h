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

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>


void savePPM(uint8_t* rgba, uint32_t width, uint32_t height, const std::string& path)
{
    // Convert to RGB
    std::vector<uint8_t> rgb(width * height * 3);
    for (uint32_t i = 0; i < width * height; ++i)
        for (uint32_t j = 0; j < 3; ++j)
            rgb[3 * i + j] = rgba[4 * i + j];

    // Write PPM
    std::ofstream outFile;
    outFile.open(path.c_str(), std::ios::binary);

    outFile << "P6" << "\n"
            << width << " " << height << "\n"
            << "255\n";

    outFile.write((char*) rgb.data(), rgb.size());
}


class Timer
{
public:
    Timer()
    {
        this->reset();
    }

    void reset()
    {
        this->time  = std::chrono::high_resolution_clock::now();
    }

    double getElapsedSeconds() const
    {
        return 1.0e-6 * std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - this->time).count();
    }

    double getElapsedMilliseconds() const
    {
        return 1.0e-3 * std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - this->time).count();
    }

private:
    std::chrono::high_resolution_clock::time_point time;
};



