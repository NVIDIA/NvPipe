# Copyright (c) 2016 NVIDIA Corporation.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this software,
# related documentation and any modifications thereto.  Any use,
# reproduction, disclosure or distribution of this software and
# related documentation without an express license agreement from
# NVIDIA CORPORATION is strictly prohibited.
cmake_minimum_required(VERSION 3.0)

set(NVPIPE "nvpipe")

project(${NVPIPE})

# Configuration
set(config_install_dir "lib/cmake/${PROJECT_NAME}")
set(include_install_dir "include")
set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
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
    pkg_check_modules(AVDEVICE "libavdevice")
    if (AVDEVICE_FOUND)
        set(nvpipe_libs ${AVDEVICE_LIBRARIES})
        include_directories(${AVDEVICE_INCLUDE_DIRS})
        link_directories(${AVDEVICE_LIBRARY_DIRS})
    else (AVDEVICE_FOUND)
        message( FATAL_ERROR "pkgcfg cannot find libavdevice.pc")
    endif(AVDEVICE_FOUND)
    pkg_check_modules(AVFILTER "libavfilter")
    if (AVFILTER_FOUND)
        set(nvpipe_libs ${nvpipe_libs} ${AVFILTER_LIBRARIES})
        include_directories(${AVFILTER_INCLUDE_DIRS})
        link_directories(${AVFILTER_LIBRARY_DIRS})
    else (AVFILTER_FOUND)
        message( FATAL_ERROR "pkgcfg cannot find libavfilter.pc")
    endif(AVFILTER_FOUND)
    pkg_check_modules(AVFORMAT "libavformat")
    if (AVFORMAT_FOUND)
        set(nvpipe_libs ${nvpipe_libs} ${AVFORMAT_LIBRARIES})
        include_directories(${AVFORMAT_INCLUDE_DIRS})
        link_directories(${AVFORMAT_LIBRARY_DIRS})
    else (AVFORMAT_FOUND)
        message( FATAL_ERROR "pkgcfg cannot find libavformat.pc")
    endif(AVFORMAT_FOUND)
    pkg_check_modules(AVCODEC "libavcodec")
    if (AVCODEC_FOUND)
        set(nvpipe_libs ${nvpipe_libs} ${AVCODEC_LIBRARIES})
        include_directories(${AVCODEC_INCLUDE_DIRS})
        link_directories(${AVCODEC_LIBRARY_DIRS})
    else (AVCODEC_FOUND)
        message( FATAL_ERROR "pkgcfg cannot find libavcodec.pc")
    endif(AVCODEC_FOUND)

    pkg_check_modules(AVUTIL "libavutil")
    if (AVUTIL_FOUND)
        set(nvpipe_libs ${nvpipe_libs} ${AVUTIL_LIBRARIES})
        include_directories(${AVUTIL_INCLUDE_DIRS})
        link_directories(${AVUTIL_LIBRARY_DIRS})
    else (AVUTIL_FOUND)
        message( FATAL_ERROR "pkgcfg cannot find libavutil.pc")
    endif(AVUTIL_FOUND)
    set(FFMPEG_SOURCES
      ${CMAKE_SOURCE_DIR}/ffmpeg.cxx
      ${CMAKE_SOURCE_DIR}/codec/abstract.cxx
      ${CMAKE_SOURCE_DIR}/codec/hw.cxx
      ${CMAKE_SOURCE_DIR}/util/format.cxx
      ${CMAKE_SOURCE_DIR}/util/kernels.cu
    )
    set(NvPipe_FFMPEG 1)
endif(USE_FFMPEG AND PKG_CONFIG_FOUND)

cuda_add_library(${NVPIPE}
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
