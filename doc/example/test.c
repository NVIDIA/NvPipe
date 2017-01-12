/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <stdio.h>
#include <inttypes.h>
#include "nvpipe.h"
#include "util/format.h"

void SaveBufferRGBA(uint8_t *data, int width, int height, char *str) {
  FILE *pFile;
  
  // Open file
  pFile=fopen(str, "wb");
  if(pFile==NULL)
    return;
  
  // Write header
  fprintf(pFile, "P6\n%d %d\n255\n", width, height);

  uint8_t *row = malloc( sizeof(uint8_t) * width * 3 );

  // Write pixel data
  for(int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      int index = x + y*width;
      row[x*3] = data[index*4];
      row[x*3+1] = data[index*4+1];
      row[x*3+2] = data[index*4+2];
    }
    fwrite(row, 1, width*3, pFile);
  }
  
  free (row);
  // Close file
  fclose(pFile);
}


void SaveBufferRGB(uint8_t *data, int width, int height, char *str) {
  FILE *pFile;
  
  // Open file
  pFile=fopen(str, "wb");
  if(pFile==NULL)
    return;
  
  // Write header
  fprintf(pFile, "P6\n%d %d\n255\n", width, height);

  uint8_t *row = malloc( sizeof(uint8_t) * width * 3 );

  // Write pixel data
  for(int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      int index = x + y*width;
      row[x*3] = data[index*3];
      row[x*3+1] = data[index*3+1];
      row[x*3+2] = data[index*3+2];
    }
    fwrite(row, 1, width*3, pFile);
  }
  
  free (row);
  // Close file
  fclose(pFile);
}

void SaveBufferBit(uint8_t *data, int length, char *str) {
  FILE *pFile;

  // Open file
  pFile=fopen(str, "ab");
  if(pFile==NULL)
    return;

  fwrite(data, 1, length, pFile);
  
  // Close file
  fclose(pFile);
}

int main( int argc, char* argv[] ) {
  (void)argc; (void)argv;
    nvpipe* enc = nvpipe_create_encoder(NVPIPE_H264_NVFFMPEG, 0);
    nvpipe* dec = nvpipe_create_decoder(NVPIPE_H264_NVFFMPEG);
    
  //  int width = 1920;
   // int height = 1080;
    size_t width = 400;
    size_t height = 400;

    size_t buffer_size = sizeof(uint8_t)*width*height*4;
    void* img_buffer = malloc(buffer_size);
    uint8_t* img_ptr0 = img_buffer;
    void* pkt_buffer = malloc(buffer_size);
    size_t pkt_buffer_size = buffer_size;
    char image_filename[30];
    int channel=4;

    for (int i = 0; i < 10; i++ ) {
        pkt_buffer_size = buffer_size;
        for(size_t y=0;y<height;y++) {
            for(size_t x=0;x<width;x++) {
                int index = y * width + x;
                    img_ptr0[index*channel] = x+y+i*5;//x+y;
                    img_ptr0[index*channel+1] = x+i*10;//x;
                    img_ptr0[index*channel+2] = 0;
                    img_ptr0[index*channel+3] = 255;
            }
        }

        
            if ( nvpipe_encode(enc, img_buffer, buffer_size, pkt_buffer, &pkt_buffer_size, width, height, NVPIPE_RGBA) == 0 ) {
                if ( nvpipe_decode(dec, pkt_buffer, pkt_buffer_size, img_buffer, width, height) == 0 ) {
                    sprintf(image_filename, "decoded_%d.pgm", i);
                    SaveBufferRGBA(img_buffer, width, height, image_filename);
                } else {
                    printf("something went wrong\n");
                }
            } else {
                printf("what happened?\n");
            }
    }

    nvpipe_destroy(enc);
    nvpipe_destroy(dec);
    free(img_buffer);
    free(pkt_buffer);
    return 0;
}
