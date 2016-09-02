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

# some extra lib needed
add_executable(test_app ${CMAKE_SOURCE_DIR}/doc/example/test.c)
link_directories($ENV{CUDA_LIBRARY_PATH})

target_link_libraries(test_app ${NVPIPE})

add_executable(client_app ${CMAKE_SOURCE_DIR}/doc/example/client.c)
target_link_libraries(client_app ${NVPIPE})
add_executable(server_app ${CMAKE_SOURCE_DIR}/doc/example/server.c)
target_link_libraries(server_app ${NVPIPE})
#add_executable(benchmark 
#cuda_add_executable(benchmark 
#                ${CMAKE_SOURCE_DIR}/doc/example/benchmark.cxx
#                ${CMAKE_SOURCE_DIR}/doc/example/file_io.cxx)

#target_link_libraries(benchmark ${NVPIPE})
#target_link_libraries(benchmark "-lnvToolsExt")

#add_executable(benchmark 
#cuda_add_executable(benchmark_v2
#                ${CMAKE_SOURCE_DIR}/doc/example/benchmarkRGBA.cxx
#                ${CMAKE_SOURCE_DIR}/doc/example/file_io.cxx)

#target_link_libraries(benchmark_v2 ${NVPIPE})
#target_link_libraries(benchmark_v2 "-lnvToolsExt")
