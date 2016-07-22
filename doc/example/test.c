#include <stdio.h>
#include <inttypes.h>
#include "nvpipe.h"

void SaveBuffer(uint8_t *data, int width, int height) {
  FILE *pFile;
  char szFilename[32];
  int  y;

  // Open file
  sprintf(szFilename, "AJBufferframe.ppm");
  pFile=fopen(szFilename, "wb");
  if(pFile==NULL)
	return;

  // Write header
  fprintf(pFile, "P6\n%d %d\n255\n", width, height);

  // Write pixel data
  for(y=0; y<height; y++)
	fwrite(data+y*width*3, 1, width*3, pFile);
  
  // Close file
  fclose(pFile);
}


int main( int argc, char* argv[] ) {
    printf("hello world!\n");

    nvpipe* codec = nvpipe_create_instance(NVPIPE_CODEC_ID_H264);


    printf("AJ 2\n");
    int width=640;
    int height=480;
    size_t buffer_size = sizeof(uint8_t)*width*height*3/2;
    printf("%zu is the size;", buffer_size);

    void* img_buffer = malloc(buffer_size);
    size_t img_buffer_size = buffer_size;
    uint8_t* img_ptr0 = img_buffer;
    uint8_t* img_ptr1 = img_ptr0 + (width*height);
    uint8_t* img_ptr2 = img_ptr0 + (width*height) / 4;
    
    int gap0 = width;
    int gap1 = width/2;
    int gap2 = width/2;
    
    for(size_t y=0;y<height;y++) {
        for(size_t x=0;x<width;x++) {
                img_ptr0[y * gap0 + x] = x + y;
        }
    }

        /* Cb and Cr */
        for(size_t y=0;y<height/2;y++) {
            for(size_t x=0;x<width/2;x++) {
                img_ptr1[y * gap1 + x] = 128 + y;
                img_ptr2[y * gap2 + x] = 64 + x;
        }
    }
    
    printf("aj 2\n");


    void* pkt_buffer = malloc(buffer_size);
    size_t pkt_buffer_size = buffer_size;
    
    printf("AJ 3\n");
    nvpipe_encode(codec, img_buffer, buffer_size, pkt_buffer, &pkt_buffer_size, width, height, NVPIPE_IMAGE_FORMAT_RGB);
    
    nvpipe_decode(codec, pkt_buffer, buffer_size, img_buffer, &img_buffer_size, &width, &height, NVPIPE_IMAGE_FORMAT_RGB);

    SaveBuffer(img_buffer, width, height);

    nvpipe_destroy_instance(codec);
    
    free(img_buffer);
    free(pkt_buffer);

    return 0;
}
