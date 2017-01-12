# Copyright (c) 2016 NVIDIA Corporation.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this software,
# related documentation and any modifications thereto.  Any use,
# reproduction, disclosure or distribution of this software and
# related documentation without an express license agreement from
# NVIDIA CORPORATION is strictly prohibited.
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

if(USE_FFMPEG)
  add_executable(test_app ${CMAKE_SOURCE_DIR}/doc/example/test.c)
  link_directories($ENV{CUDA_LIBRARY_PATH})

  target_link_libraries(test_app ${NVPIPE})
endif(USE_FFMPEG)
