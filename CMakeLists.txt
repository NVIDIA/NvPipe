# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
cmake_minimum_required(VERSION 3.0)

set(NVPIPE "nvpipe")

project(${NVPIPE})

# Configuration
set(config_install_dir "lib/cmake/${PROJECT_NAME}")
set(include_install_dir "include")
set(project_config "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include_directories(BEFORE "${PROJECT_BINARY_DIR}")

find_package(CUDA)
list(APPEND CUDA_NVCC_FLAGS "--std c++11")

include_directories(${CMAKE_SOURCE_DIR} ${CUDA_SDK_ROOT_DIR}/common/inc)
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
link_directories("${CMAKE_INSTALL_PREFIX}/lib")

cuda_compile_ptx(ptxprograms
	convert.cu
)

option(USE_FFMPEG "Enable ffmpeg-based backend." OFF)
set(NvPipe_FFMPEG 0)

# using pkg-config to get ffmpeg
find_package(PkgConfig)
if(USE_FFMPEG AND PKG_CONFIG_FOUND)
    pkg_check_modules(AVFORMAT REQUIRED "libavformat")
    set(nvpipe_libs ${nvpipe_libs} ${AVFORMAT_LIBRARIES})
    include_directories(${AVFORMAT_INCLUDE_DIRS})
    link_directories(${AVFORMAT_LIBRARY_DIRS})

    pkg_check_modules(AVCODEC REQUIRED "libavcodec")
    set(nvpipe_libs ${nvpipe_libs} ${AVCODEC_LIBRARIES})
    include_directories(${AVCODEC_INCLUDE_DIRS})
    link_directories(${AVCODEC_LIBRARY_DIRS})

    pkg_check_modules(AVUTIL REQUIRED "libavutil")
    set(nvpipe_libs ${nvpipe_libs} ${AVUTIL_LIBRARIES})
    include_directories(${AVUTIL_INCLUDE_DIRS})
    link_directories(${AVUTIL_LIBRARY_DIRS})

    set(FFMPEG_SOURCES
      ${CMAKE_SOURCE_DIR}/ffmpeg.cxx
      ${CMAKE_SOURCE_DIR}/codec/abstract.cxx
      ${CMAKE_SOURCE_DIR}/codec/hw.cxx
      ${CMAKE_SOURCE_DIR}/util/format.cxx
      ${CMAKE_SOURCE_DIR}/util/kernels.cu
    )
    set(NvPipe_FFMPEG 1)
endif(USE_FFMPEG AND PKG_CONFIG_FOUND)

cuda_add_library(${NVPIPE} SHARED
  ${FFMPEG_SOURCES}
	${CMAKE_SOURCE_DIR}/debug.c
	${CMAKE_SOURCE_DIR}/decode.c
	${CMAKE_SOURCE_DIR}/encode.c
	${CMAKE_SOURCE_DIR}/error.c
	${CMAKE_SOURCE_DIR}/yuv.c
	${CMAKE_SOURCE_DIR}/api.c
	${ptxprograms}
	SHARED
)

# x86-64 libraries always need to be PIC.
set_property(TARGET ${NVPIPE} PROPERTY POSITION_INDEPENDENT_CODE ON)

# Linking with C for supercomputer-friendliness, but we still have some C++ in
# the ffmpeg backend.
if(NOT NvPipe_FFMPEG)
	set_target_properties(${NVPIPE} PROPERTIES LINKER_LANGUAGE C)
endif()
# The direct nvEnc/cuvid backends use c99.
set_property(TARGET ${NVPIPE} PROPERTY C_STANDARD 99)
set_property(TARGET ${NVPIPE} PROPERTY CXX_STANDARD 11)

target_link_libraries(${NVPIPE} PRIVATE
  ${nvpipe_libs} cuda nvToolsExt nvcuvid dl pthread
)

add_subdirectory(doc/example)

configure_file(
    "cmake/Config.cmake.in"
    "${project_config}" @ONLY
)
set(NvPipe_DECODE 1)
set(NvPipe_ENCODE 1)
set(NvPipe_PREFIX ${CMAKE_INSTALL_PREFIX})
configure_file(
	"${PROJECT_SOURCE_DIR}/config.nvp.h.in"
	"${PROJECT_BINARY_DIR}/config.nvp.h"
)

install(FILES nvpipe.h DESTINATION include)
install(TARGETS ${NVPIPE} DESTINATION lib)
install(FILES "${project_config}" DESTINATION lib/cmake/${NVPIPE})
install(FILES ${PROJECT_BINARY_DIR}/convert.ptx DESTINATION share/nvpipe)
install(FILES ${PROJECT_BINARY_DIR}/config.nvp.h DESTINATION include)
