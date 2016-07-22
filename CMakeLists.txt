# Copyright (c) 2016 NVIDIA Corporation.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual
# property and proprietary rights in and to this software,
# related documentation and any modifications thereto.  Any use,
# reproduction, disclosure or distribution of this software and
# related documentation without an express license agreement from
# NVIDIA Corporation is strictly prohibited.
#
# TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE
# IS PROVIDED *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL
# WARRANTIES, EITHER EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED
# TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE
# LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL
# DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
# LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
# INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF
# OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGES
cmake_minimum_required(VERSION 3.0)
project(nvpipe)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${MI_INCLUDE})
include_directories(${CMAKE_SOURCE_DIR})

# AJ library
set(NVPIPE "nvpipe")

add_library(${NVPIPE}
    ${CMAKE_SOURCE_DIR}/libnvpipecodec/nvpipecodec.cxx
    ${CMAKE_SOURCE_DIR}/libnvpipecodec/nvpipecodec264.cxx
    ${CMAKE_SOURCE_DIR}/libnvpipeutil/util.cxx
)

# using pkg-config to get ffmpeg
find_package(PkgConfig)
if (PKG_CONFIG_FOUND)

  pkg_check_modules(AVDEVICE "libavdevice")
  if (AVDEVICE_FOUND)    
    target_link_libraries(${NVPIPE} ${AVDEVICE_LIBRARIES})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${AVDEVICE_CFLAGS}")
  endif(AVDEVICE_FOUND)

  pkg_check_modules(AVFILTER "libavfilter")
  if (AVFILTER_FOUND)    
    target_link_libraries(${NVPIPE} ${AVFILTER_LIBRARIES})
  endif(AVFILTER_FOUND)

  pkg_check_modules(AVFORMAT "libavformat")
  if (AVFORMAT_FOUND)    
    target_link_libraries(${NVPIPE} ${AVFORMAT_LIBRARIES})
  endif(AVFORMAT_FOUND)

  pkg_check_modules(AVCODEC "libavcodec")
  if (AVCODEC_FOUND)    
    target_link_libraries(${NVPIPE} ${AVCODEC_LIBRARIES})
  endif(AVCODEC_FOUND)

  pkg_check_modules(AVSWRESAMPLE "libswresample")
  if (AVSWRESAMPLE_FOUND)    
    target_link_libraries(${NVPIPE} ${AVSWRESAMPLE_LIBRARIES})
  endif(AVSWRESAMPLE_FOUND)

  pkg_check_modules(AVSWSCALE "libswscale")
  if (AVSWSCALE_FOUND)    
    target_link_libraries(${NVPIPE} ${AVSWSCALE_LIBRARIES})
  endif(AVSWSCALE_FOUND)

  pkg_check_modules(AVUTIL "libavutil")
  if (AVUTIL_FOUND)    
    target_link_libraries(${NVPIPE} ${AVUTIL_LIBRARIES})
  endif(AVUTIL_FOUND)

endif(PKG_CONFIG_FOUND)

# some extra lib needed
target_link_libraries (${NVPIPE} "-lm -lX11 -lva -lva-drm -lva-x11 -lvdpau")

add_executable(example ${CMAKE_SOURCE_DIR}/doc/example/test.c)

target_link_libraries(example ${NVPIPE})
