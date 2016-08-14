add_library(nvpipe STATIC IMPORTED)

find_library(NVPIPE_LIBRARY_PATH nvpipe HINTS "/usr/local/lib/")

set(NVPIPE_LIBRARIES nvpipe)

find_package(CUDA)
  set(NVPIPE_LIBRARIES ${NVPIPE_LIBRARIES} ${CUDA_LIBRARIES})

find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
  pkg_check_modules(AVDEVICE "libavdevice")
  if (AVDEVICE_FOUND)    
    set(NVPIPE_LIBRARIES ${NVPIPE_LIBRARIES} ${AVDEVICE_LIBRARIES})
  endif(AVDEVICE_FOUND)

  pkg_check_modules(AVFILTER "libavfilter")
  if (AVFILTER_FOUND)    
    set(NVPIPE_LIBRARIES ${NVPIPE_LIBRARIES} ${AVFILTER_LIBRARIES})
  endif(AVFILTER_FOUND)

  pkg_check_modules(AVFORMAT "libavformat")
  if (AVFORMAT_FOUND)    
    set(NVPIPE_LIBRARIES ${NVPIPE_LIBRARIES} ${AVFORMAT_LIBRARIES})
  endif(AVFORMAT_FOUND)

  pkg_check_modules(AVCODEC "libavcodec")
  if (AVCODEC_FOUND)    
    set(NVPIPE_LIBRARIES ${NVPIPE_LIBRARIES} ${AVCODEC_LIBRARIES})
  endif(AVCODEC_FOUND)

  pkg_check_modules(AVSWRESAMPLE "libswresample")
  if (AVSWRESAMPLE_FOUND)    
    set(NVPIPE_LIBRARIES ${NVPIPE_LIBRARIES} ${AVSWRESAMPLE_LIBRARIES})
  endif(AVSWRESAMPLE_FOUND)

  pkg_check_modules(AVSWSCALE "libswscale")
  if (AVSWSCALE_FOUND)    
    set(NVPIPE_LIBRARIES ${NVPIPE_LIBRARIES} ${AVSWSCALE_LIBRARIES})
  endif(AVSWSCALE_FOUND)

  pkg_check_modules(AVUTIL "libavutil")
  if (AVUTIL_FOUND)    
    set(NVPIPE_LIBRARIES ${NVPIPE_LIBRARIES} ${AVUTIL_LIBRARIES})
  endif(AVUTIL_FOUND)

endif(PKG_CONFIG_FOUND)

# some extra lib needed
# target_link_libraries (${NVPIPE} "-lm -lX11 -lva -lva-drm -lva-x11 -lvdpau")
set_target_properties(nvpipe PROPERTIES IMPORTED_LOCATION "${NVPIPE_LIBRARY_PATH}")
