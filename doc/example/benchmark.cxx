/*
 * Copyright (c) 2016 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual
 * property and proprietary rights in and to this software,
 * related documentation and any modifications thereto.  Any use,
 * reproduction, disclosure or distribution of this software and
 * related documentation without an express license agreement from
 * NVIDIA CORPORATION is strictly prohibited.
 *
 */
#include <cstdio>
#include <cinttypes>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include "nvpipe.h"
#include "doc/example/file_io.h"

#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
//#include <nvToolsExtCuda.h>
//#include <nvToolsExtCudaRt.h>
//#include <cudaProfiler.h>


void PrintHelp() {
    printf("Usage: \n");
    printf("-e\n");
    printf("    run encoding benchmark\n");
    printf("-d\n");
    printf("    run decoding benchmark\n");
    printf("-i [string/filename]\n");
    printf("    specify input flag:\n");
    printf("      encoder\n");
    printf("          'cont' for continuous changing data\n");
    printf("          'rand' for randomly changing data\n");
    printf("      decoder\n");
    printf("      filename of packet file\n");
    printf("-w [number]\n");
    printf("    specify width of the image (only for encoder)\n");
    printf("-h [number]\n");
    printf("    specify height of the image (only for encoder)\n");
    printf("-f [number]\n");
    printf("    specify total # of frame\n");
    printf("-r [number]\n");
    printf("    specify repeating times\n");
    printf("-p\n");
    printf("    save image to files\n");
    printf("-v\n");
    printf("    discard first frame in profiler\n");
    printf("-o [filename]\n");
    printf("    specify output filename\n");
    printf("      encoder will generate filename.pkt\n");
    printf("      decoder will generate a series of filename#.pgm\n");
    printf("-b [number]\n");
    printf("    specify encoding bitrate\n");
}

int64_t ConvertStrTo64(const char *s) {
  int64_t i;
  char c ;
  int scanned = sscanf(s, "%" SCNd64 "%c", &i, &c);
  if (scanned == 1) return i;
  if (scanned > 1) {
    // TBD about extra data found
    return i;
    }
  // TBD failed to scan;  
  return 0;  
}


int main( int argc, char* argv[] ) {
    bool encoding_flag = false;
    bool decoding_flag = false;

    int frame_number = 100;
    int width = 1280;
    int height = 720;
    int repeat = 1;
    bool save_image_flag = false;
    bool discard_initial_timer = false;
    int64_t bitrate = 0;
    char *output_file = NULL;

    char *input_flag = NULL;

    int c;
    while ((c = getopt (argc, argv, "depvi:f:o:w:h:b:r:")) != -1) {
        switch (c) {
        case 'd':
            decoding_flag = true;
            break;
        case 'p':
            save_image_flag = true;
            break;
        case 'e':
            encoding_flag = true;
            break;
        case 'i':
            input_flag = optarg;
            break;
        case 'f':
            frame_number = std::stoi(optarg);
            break;
        case 'w':
            width = std::stoi(optarg);
            break;
        case 'h':
            height = std::stoi(optarg);
            break;
        case 'o':
            output_file = optarg;
            break;
        case 'b':
            bitrate = ConvertStrTo64(optarg);
            break;
        case 'r':
            repeat = std::stoi(optarg);
            break;
        case 'v':
            discard_initial_timer = true;
            break;
        case '?':
            if (isprint (optopt))
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf (stderr,
                        "Unknown option character `\\x%x'.\n",
                        optopt);
            PrintHelp();
            return -1;
        default:
            PrintHelp();
            return -1;
        }
    }

    if ( !(encoding_flag|decoding_flag) ) {
        printf ("[ERROR] specify encoding and/or decoding...\n\n");
        PrintHelp();
        return -1;
    }

    if ( encoding_flag ) {
        printf ("encoding benchmark test...\n");
        printf ("   image resolution: %dx%d\n", width, height);
        printf ("   frame #: %d\n", frame_number);
    } else {
        printf ("decoding benchmark test...\n");
    }

    if ( input_flag )
        printf ("   input flag: %s\n", input_flag);

    printf ("   output_file: %s\n", output_file);

    printf ("*************************\n");
    printf ("\n   experiment starts...\n");

    nvpipe* codec = nvpipe_create_instance(NVPIPE_CODEC_ID_H264_HARDWARE);

    size_t buffer_size = sizeof(uint8_t) * 
                            width * height *
                            frame_number * 3;

    uint8_t* img_buffer = (uint8_t*)pa_alloc(buffer_size);

    uint8_t* pkt_buffer = (uint8_t*)pa_alloc(buffer_size * repeat / 100);

    MemoryStack img_stack(img_buffer, buffer_size);
    MemoryStack pkt_stack(pkt_buffer, buffer_size/8);

    size_t image_size = width * height * 3 * sizeof(uint8_t);
    uint8_t *ptr;
    size_t current_packet_size;

    char file_name[50];

    if ( encoding_flag ) {

        size_t total_packet_size = 0;

        if ( input_flag ) {
            img_stack.loadBufferFromFileList(   input_flag,
                                                PLAIN_DATA,
                                                frame_number);
        } else {
            for ( int i = 0; i < frame_number; i++ ) {
                ptr = img_stack.getBufferHandle();
                for(size_t y=0;y<height;y++) {
                    for(size_t x=0;x<width;x++) {
                        int index = y * width + x;
                        ptr[index*3] = (x + y + i*5);
                        ptr[index*3+1] = 128 + y + i *3;
                        ptr[index*3+2] = 64 + x;
                    }
                }
                img_stack.pushBuffer(image_size);
            }
        }

        if ( output_file && save_image_flag ) {
            sprintf(file_name, "%s_encoded", output_file);
            img_stack.writeBufferToFileList(file_name,
                                            RGB_PICTURE,
                                            width,
                                            height);
        }
        ptr = pkt_stack.getBufferHandle();

        if ( bitrate > 0 ) {
            nvpipe_set_bitrate(codec, bitrate);
        }

        /************************************************
         *  encoding timing!
         ************************************************/
        int iteration = 0;
        int index = 0;
        //cudaProfilerStart();
        bool flag = !discard_initial_timer;
        for ( int i = 0; i < frame_number; i++ ) {
            current_packet_size = pkt_stack.getRemainingSpace();
            
            //if ( flag )
                //nvtxRangePushA("encodingSession");
            
            if ( (iteration & 1) == 0 ) {
                index = i;
            } else {
                index = frame_number - i - 1;
            }
            
            //printf ("%d, %d\n", i, index);
            
            nvpipe_encode(  codec,
                            img_stack.getBufferHandle(index), image_size,
                            pkt_stack.getBufferHandle(), &current_packet_size,
                            width, height,
                            NVPIPE_IMAGE_FORMAT_RGB);
            pkt_stack.pushBuffer(current_packet_size);
            //if ( flag )
                //nvtxRangePop();
            //else
                //flag = true;

            if ( i == frame_number - 2 && iteration < repeat-1 ) {
                iteration++;
                i = -1;
            }
        }
        //cudaProfilerStop();
        /***encoding timing ended************************/

        printf ("   image frame count: %d, memory usage: size: %zu\n", 
                    img_stack.getSize(),
                    img_stack.getUsedSpace());
        printf ("   packet size: %d, memory usage: %zu\n", 
                pkt_stack.getSize(), 
                pkt_stack.getUsedSpace() );
        double ratio = pkt_stack.getUsedSpace();
        ratio *= img_stack.getSize();
        ratio /= img_stack.getUsedSpace();
        ratio /= pkt_stack.getSize();
        printf ("   compression ratio: %lf\n", ratio);

        //for ( int i = 0; i < pkt_stack.getSize(); i++ ) {
        //    printf("      packet index: %d, size: %zu\n",
        //            i, pkt_stack.getBufferSize(i));
        //}
        if ( output_file ) {
            sprintf(file_name, "%s.264", output_file);
            pkt_stack.writeBufferToFile(file_name);
        }

    }

    if ( decoding_flag ) {
        int index = 0;
        bool flag = !discard_initial_timer;
        int decode_width = width;
        int decode_height = height;
        if ( !encoding_flag ) {
            // load from file, write it later maybe
        }

        /************************************************
         *  decoding timing!
         ************************************************/
        //cudaProfilerStart();
        for ( int i = 0; i < pkt_stack.getSize(); i++ ) {
            current_packet_size = pkt_stack.getBufferSize(i);
            //if ( flag ) {
                //nvtxRangePushA("decodingSession");
            //}
            
            index = i % frame_number;
            nvpipe_decode(  codec,
                            pkt_stack.getBufferHandle(i), current_packet_size,
                            img_stack.getBufferHandle(index), image_size,
                            &decode_width, &decode_height,
                            NVPIPE_IMAGE_FORMAT_RGB);
            //if ( flag )
                //nvtxRangePop();
            //else
                //flag = true;

            
        }
        //cudaProfilerStop();
        /***decoding timing ended************************/

        if ( output_file && save_image_flag ) {
            sprintf(file_name, "%s_decoded", output_file);
            img_stack.writeBufferToFileList(file_name,
                                            RGB_PICTURE,
                                            decode_width,
                                            decode_height);
        }
    }

    free(img_buffer);
    free(pkt_buffer);
    return 0;
}
