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

set(NVPIPE "nvpipe")

project(${NVPIPE})

# Configuration
set(config_install_dir "lib/cmake/${PROJECT_NAME}")
set(include_install_dir "include")
set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")

configure_file(
    "cmake/Config.cmake.in"
    "${project_config}" @ONLY)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -std=c99 -pthread -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -pthread -fPIC")

find_package(CUDA)

option(NVPIPE_USE_LOCAL_FFMPEG "" OFF)
option(NVPIPE_USE_LOCAL_CUDA "" OFF)

include_directories(${CMAKE_SOURCE_DIR} ${CUDA_SDK_ROOT_DIR}/common/inc)
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")

cuda_add_library(${NVPIPE}
    ${CMAKE_SOURCE_DIR}/libnvpipecodec/nvpipecodec.cxx
    ${CMAKE_SOURCE_DIR}/libnvpipecodec/nvpipecodec264.cxx
    ${CMAKE_SOURCE_DIR}/libnvpipeutil/util.cxx
    ${CMAKE_SOURCE_DIR}/libnvpipeutil/kernels.cu
    SHARED
)

# using pkg-config to get ffmpeg
if(NVPIPE_USE_LOCAL_FFMPEG)
  set(NVPIPE_FFMPEG_DIRECTORY "" CACHE STRING "set root directory for ffmpeg")
  if("${NVPIPE_FFMPEG_DIRECTORY}" STREQUAL "")
    message("do something")
  else("${NVPIPE_FFMPEG_DIRECTORY}" STREQUAL "")
    message("oops")
  endif("${NVPIPE_FFMPEG_DIRECTORY}" STREQUAL "")
else(NVPIPE_USE_LOCAL_FFMPEG)
  find_package(PkgConfig)
  if (PKG_CONFIG_FOUND)
    pkg_check_modules(AVDEVICE "libavdevice")
    if (AVDEVICE_FOUND)    
      target_link_libraries(${NVPIPE} ${AVDEVICE_LIBRARIES})
      include_directories(${AVDEVICE_INCLUDE_DIRS})
      link_directories(${AVDEVICE_LIBRARY_DIRS})
      #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${AVDEVICE_CFLAGS}")
    endif(AVDEVICE_FOUND)
    pkg_check_modules(AVFILTER "libavfilter")
    if (AVFILTER_FOUND)    
      target_link_libraries(${NVPIPE} ${AVFILTER_LIBRARIES})
      include_directories(${AVFILTER_INCLUDE_DIRS})
      link_directories(${AVFILTER_LIBRARY_DIRS})
    endif(AVFILTER_FOUND)
    pkg_check_modules(AVFORMAT "libavformat")
    if (AVFORMAT_FOUND)    
      target_link_libraries(${NVPIPE} ${AVFORMAT_LIBRARIES})
      include_directories(${AVFORMAT_INCLUDE_DIRS})
      link_directories(${AVFORMAT_LIBRARY_DIRS})
    endif(AVFORMAT_FOUND)
    pkg_check_modules(AVCODEC "libavcodec")
    if (AVCODEC_FOUND)    
      target_link_libraries(${NVPIPE} ${AVCODEC_LIBRARIES})
      include_directories(${AVCODEC_INCLUDE_DIRS})
      link_directories(${AVCODEC_LIBRARY_DIRS})
    endif(AVCODEC_FOUND)
    pkg_check_modules(AVSWRESAMPLE "libswresample")
    if (AVSWRESAMPLE_FOUND)    
      target_link_libraries(${NVPIPE} ${AVSWRESAMPLE_LIBRARIES})
      include_directories(${AVSWRESAMPLE_INCLUDE_DIRS})
      link_directories(${AVSWRESAMPLE_LIBRARY_DIRS})
    endif(AVSWRESAMPLE_FOUND)
    pkg_check_modules(AVSWSCALE "libswscale")
    if (AVSWSCALE_FOUND)    
      target_link_libraries(${NVPIPE} ${AVSWSCALE_LIBRARIES})
      include_directories(${AVSWSCALE_INCLUDE_DIRS})
      link_directories(${AVSWSCALE_LIBRARY_DIRS})
    endif(AVSWSCALE_FOUND)
    pkg_check_modules(AVUTIL "libavutil")
    if (AVUTIL_FOUND)    
      target_link_libraries(${NVPIPE} ${AVUTIL_LIBRARIES})
      include_directories(${AVUTIL_INCLUDE_DIRS})
      link_directories(${AVUTIL_LIBRARY_DIRS})
    endif(AVUTIL_FOUND)
  endif(PKG_CONFIG_FOUND)
endif(NVPIPE_USE_LOCAL_FFMPEG)

# some extra lib needed
target_link_libraries (${NVPIPE} "-lm -lX11 -lva -lva-drm -lva-x11")
target_link_libraries(${NVPIPE} "-lnvToolsExt")

# build example application
add_subdirectory("./doc/example")

install(FILES nvpipe.h DESTINATION include)
install(FILES libnvpipecodec/nvpipecodec264.h libnvpipecodec/nvpipecodec.h 
        DESTINATION include/libnvpipecodec/)
install(FILES libnvpipeutil/format.h libnvpipeutil/formatConversionCuda.h
        DESTINATION include/libnvpipeutil/)
install(TARGETS ${NVPIPE} DESTINATION lib)
install(FILES "${project_config}" DESTINATION lib/cmake/${NVPIPE})
