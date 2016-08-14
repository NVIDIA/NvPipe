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
#project(nvpipe)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -g")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -pthread")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

#include_directories(${CMAKE_SOURCE_DIR})
#link_directories($ENV{LD_LIBRARY_PATH})

# AJ library
set(NVPIPE "nvpipe")
#link_directories($ENV{LD_LIBRARY_PATH})
#find_package(CUDA)

# some extra lib needed
add_executable(test_app ${CMAKE_SOURCE_DIR}/doc/example/test.c)
link_directories($ENV{CUDA_LIBRARY_PATH})

target_link_libraries(test_app ${NVPIPE})

#add_executable(benchmark 
cuda_add_executable(benchmark 
                ${CMAKE_SOURCE_DIR}/doc/example/benchmark.cxx
                ${CMAKE_SOURCE_DIR}/doc/example/file_io.cxx)

target_link_libraries(benchmark ${NVPIPE})
target_link_libraries(benchmark "-lnvToolsExt")
