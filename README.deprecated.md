Introduction
============

NvPipe is a simple and lightweight C API library for low-latency video compression.
It provides easy-to-use access to NVIDIA's hardware-accelerated H.264 and HEVC video codecs and is a great choice to drastically lower the bandwidth required for your
networked interactive server/client application.

Designed for both remote rendering solutions and general compression of arbitrary image data, NvPipe accepts frames in various formats and supports access to host memory, CUDA device memory, OpenGL textures and OpenGL pixel buffer objects.

Supported formats are 32 bit RGBA frames (8 bit per channel; alpha is not supported by the underlying video codecs and is ignored) and unsigned integer grayscale frames with 4 bit, 8 bit, 16 bit or 32 bit per pixel.

Besides conventional lossy video compression based on target bitrate and framerate, also fully lossless compression is available enabling exact bit pattern reconstruction.

Please note that NvPipe acts as a lightweight synchronous convenience layer around the [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk) and doesn't offer all high-performance capabilities.
If you're looking for ultimate encode/decode performance, you may want to consider using NvCodec directly.


Usage
============

The library is specifically designed to be easily integratable into existing
low-latency streaming applications.  NvPipe does not take over any of the
network communication aspects, allowing your application to dictate the
server/client scenario it is used in.

A sample encoding scenario:

```c++
#include <NvPipe.h>
...

uint32_t width = ..., height = ...; // Image resolution
uint8_t* rgba = ...; // Image data in device or host memory
uint8_t* buffer = ...; // Buffer for compressed output in host memory

// Create encoder
NvPipe* encoder = NvPipe_CreateEncoder(NVPIPE_RGBA32, NVPIPE_H264, NVPIPE_LOSSY, 32 * 1000 * 1000, 90, width, height); // 32 Mbps @ 90 Hz

while (frameAvailable)
{
    // Encode next frame
    uint64_t compressedSize = NvPipe_Encode(encoder, rgba, width * 4, buffer, bufferSize, width, height, false);

    // Send the frame size and compressed stream to the consuming side
    send(socket, &compressedSize, sizeof(uint64_t), ...);
    send(socket, buffer, compressedSize, ...);
    send(socket, ...) // Other meta data
}

// Destroy encode session
NvPipe_Destroy(encoder);
```

The corresponding decoding scenario:

```c++
#include <NvPipe.h>
...

uint32_t width = ..., height = ...; // Image resolution
uint8_t* rgba = ...; // Image destination in device or host memory
uint8_t* buffer = ...; // Buffer for incoming packets

// Create decoder
NvPipe* decoder = NvPipe_CreateDecoder(NVPIPE_RGBA32, NVPIPE_H264, width, height);

while (frameAvailable)
{
    // Receive data
    uint64_t compressedSize;
    receive(socket, &compressedSize, sizeof(uint64_t), ...);
    receive(socket, buffer, compressedSize, ...);
    receive(socket, ...);

    // Decode frame
    NvPipe_Decode(decoder, buffer, compressedSize, rgba, width, height);

    // Use frame (blit/save/...)
    ...
}

// Destroy decode session
NvPipe_Destroy(decoder);
```



Installation
============
NvPipe requires a C++ compiler and uses the standard CMake compilation and installation procedure.

Download the Video Codec SDK from https://developer.nvidia.com/nvidia-video-codec-sdk (tested with 9.1.23).

```bash
mkdir build && cd build
cmake PATH_TO_NVPIPE -DNV_VIDEO_CODEC_SDK=PATH_TO_SDK
make
make install
```

It is possible to compile the encoding or decoding interface only using the `NVPIPE_WITH_ENCODER` and `NVPIPE_WITH_DECODER` options (default: `ON`).

The OpenGL interface is optional and can be disabled using the `NVPIPE_WITH_OPENGL` option (default: `ON`).

The compilation of the included sample applications can be controlled via the `NVPIPE_BUILD_EXAMPLES` CMake option (default: `ON`).

Only shared libraries are supported.


##### Compiling on Windows using Visual Studio 2017 #####

On Windows, NvPipe can be compiled using Visual Studio's built-in CMake support.

Just place a `CMakeSettings.json` file with the following contents (adjust paths accordingly) next to NvPipe's `CMakeLists.txt`:

```
{
  "configurations": [
    {
      "name": "x64-Release",
      "generator": "Ninja",
      "configurationType": "Release",
      "inheritEnvironments": [
        "msvc_x64_x64"
      ],
      "buildRoot": "C:\\.build\\NvPipe\\${name}",
      "installRoot": "C:\\.install\\NvPipe\\${name}",
      "cmakeCommandArgs": "",
      "buildCommandArgs": "-v",
      "ctestCommandArgs": "",
      "variables": [
        {
          "name": "GLEW_INCLUDE_DIR",
          "value": "C:\\PATH\\TO\\glew-2.1.0\\include"
        },
        {
          "name": "GLEW_LIBRARY_RELEASE",
          "value": "C:\\PATH\\TO\\glew-2.1.0\\lib\\Release\\x64\\glew32.lib"
        }
      ]
    }
  ]
}
```


Examples
=====

Two example applications are included that perform encoding and decoding of a sequence of frames, measure performance, and dump image files before and after compression.
Additionally, an example of integer frame compression verifies the bitwise correctness of lossless encoding and decoding.

The `memory` example compares the performance of input and output based on host memory vs. CUDA device memory.
As illustrated in the following example output, device memory can be directly accessed by the video codec hardware and is thus faster, whereas host memory entails additional bus transfers.
```bash
$ ./nvpExampleMemory
NvPipe example application: Comparison of using host/device memory.

Resolution: 3840 x 2160
Codec: H.264
Bitrate: 32 Mbps @ 90 Hz
Resolution: 3840 x 2160

--- Encode from host memory / Decode to host memory ---
Frame | Encode (ms) | Decode (ms) | Size (KB)
    0 |        57.8 |        42.2 |     31.0
    1 |        15.4 |        13.3 |     12.1
    2 |        16.6 |        13.5 |      5.5
    3 |        16.6 |        13.6 |      8.3
    4 |        16.9 |        13.8 |      3.9
    5 |        17.1 |        13.8 |      3.5
    6 |        16.9 |        13.8 |      3.5
    7 |        17.0 |        13.8 |      3.5
    8 |        17.0 |        13.8 |      3.5
    9 |        16.9 |        14.3 |      3.5

--- Encode from device memory / Decode to device memory ---
Frame | Encode (ms) | Decode (ms) | Size (KB)
    0 |        45.9 |        35.0 |     31.0
    1 |        10.5 |         6.9 |     12.1
    2 |        10.2 |         6.8 |      5.5
    3 |        10.1 |         6.8 |      8.3
    4 |        10.2 |         6.8 |      3.9
    5 |        10.2 |         6.8 |      3.5
    6 |        10.1 |         6.9 |      3.5
    7 |        10.2 |         6.8 |      3.5
    8 |        10.1 |         6.9 |      3.5
    9 |        10.1 |         6.8 |      3.5
```

As indicated by the size column, the first frame is an I-frame and thus requires more bandwidth. The subsequent frames however are more lightweight P-frames, which only describe differences to previous frames.


The `egl` example application demonstrates the usage of NvPipe in a server/client remote rendering scenario. An offscreen OpenGL framebuffer is created through EGL which is [ideally suited for remote rendering on headless nodes without X server](https://devblogs.nvidia.com/egl-eye-opengl-visualization-without-x-server/). The rendered frame is encoded by directly accessing the framebuffer's color attachment. After decoding, a fullscreen texture is used to draw the frame to the default framebuffer.
The following example output shows that performance is similar to CUDA device memory access as illustrated above.
```bash
$ ./nvpExampleEGL
NvPipe example application: Render to offscreen framebuffer using EGL,
encode framebuffer, decode to display texture.

Resolution: 3840 x 2160
Codec: H.264
Bitrate: 32 Mbps @ 90 Hz

Frame | Encode (ms) | Decode (ms) | Size (KB)
    0 |        54.4 |        36.4 |     38.5
    1 |        11.8 |         7.5 |     29.4
    2 |        11.9 |         7.4 |     28.5
    3 |        11.8 |         7.5 |     28.0
    4 |        11.9 |         7.4 |     27.8
    5 |        11.9 |         7.5 |     27.8
    6 |        11.8 |         7.5 |     27.7
    7 |        11.9 |         7.5 |     27.4
    8 |        11.9 |         7.4 |     27.4
    9 |        11.8 |         7.5 |     27.6
```

Note that the overall compressed sizes differ in the two examples as they use different images for input.

The `lossless` example demonstrates the usage of NvPipe for lossless integer data compression. An 8 bit unsigned integer frame is created, compressed, and the result verified for bitwise correctness after decompression.
The following example output shows that while correctness is always guaranteed due to lossless compression, the chosen pixel format has crucial impact on the compressed output size.
```bash
$ ./nvpExampleLossless
NvPipe example application: Tests lossless compression of a grayscale integer frame.

Input: 1024 x 1024 UINT8 (Raw size: 1048.6 KB)
 - [as UINT4]  Size: 181.3 KB, Encode: 14.8 ms, Decode: 17.1 ms - OK
 - [as UINT8]  Size: 45.2 KB, Encode: 14.8 ms, Decode: 15.5 ms - OK
 - [as UINT16]  Size: 57.7 KB, Encode: 16.4 ms, Decode: 11.4 ms - OK
```

The ideal pixel format is highly dependent on the structure of your input data. Keep in mind that video codecs are optimized for spatial and temporal coherence. For instance, the 8 bit pixel data  in the example above interpreted as 4 bit pixels results in poor compression due to high frequency noise from the encoder's perspective.



Supported Platforms
===================

NvPipe is supported on both Linux and Windows. OS X support is not plausible in the short term.

Please refer to the hardware capability matrices of the [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk) for more details on feature availability and driver requirements.

Note that NvPipe does not support the Jetson platform, on which the video hardware should be accessed through the [NvMedia API](https://docs.nvidia.com/drive/nvvib_docs/NVIDIA%20DRIVE%20Linux%20SDK%20Development%20Guide/baggage/group__nvmedia__top.html).



Publication
===================

NvPipe was successfully used in the EGPGV 2018 best paper *Hardware-Accelerated Multi-Tile Streaming for Realtime Remote Visualization*, which is available [here](https://hpcvis.org/publications/Streaming-EGPGV2018.pdf).



Feedback
===================

Feedback and pull requests welcome! After starting a PR, remember to sign the CLA.

We would love to hear more about your use cases of NvPipe!
