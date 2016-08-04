#include <stdio.h>
#include <inttypes.h>
#include "nvpipe.h"

void SaveBufferNV12(uint8_t *data, int width, int height, char *str) {
  FILE *pFile;
  
  pFile=fopen(str, "wb");
  if(pFile==NULL)
	return;

  // Write header
  fprintf(pFile, "P5\n%d %d\n255\n", width, height);

  // Write pixel data
  for(int y=0; y<height; y++)
	fwrite(data+y*width, 1, width, pFile);
  
  // Close file
  fclose(pFile);
}

void SaveBufferARGB(uint8_t *data, int width, int height, char *str) {
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
      row[x*3] = data[index*4+1];
      row[x*3+1] = data[index*4+2];
      row[x*3+2] = data[index*4+3];
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
    fwrite(data, 1, width*3, pFile);
    data += width*3;
  }
  // Close file
  fclose(pFile);
}


static void*
pa_alloc(size_t sz) {
	void* rv;
	const int err = posix_memalign(&rv, 4096, sz);
	if(0 != err) {
		fprintf(stderr, "%zu-byte allocation failed.\n", sz);
		return NULL;
	}
	return rv;
}


void SaveBufferYUV420P(uint8_t *data, int width, int height, char *str) {
  FILE *pFile;
  char szFilename[32];
  int  y;

  // Open file
  sprintf(szFilename, str);
  pFile=fopen(szFilename, "wb");
  if(pFile==NULL)
	return;

  // Write header
  fprintf(pFile, "P5\n%d %d\n255\n", width, height);

  // Write pixel data
  for(y=0; y<height; y++)
	fwrite(data+y*width, 1, width, pFile);
  
  // Close file
  fclose(pFile);
}


void SaveBufferBit(uint8_t *data, int length, char *str) {
  FILE *pFile;
  char szFilename[32];
  int  y;

  // Open file
  sprintf(szFilename, str);
  pFile=fopen(szFilename, "wb");
  if(pFile==NULL)
	return;

  fwrite(data, 1, length, pFile);
  
  // Close file
  fclose(pFile);
}


static void pgm_save(unsigned char *buf, int wrap, int xsize, int ysize,
                     char *filename)
{
    FILE *f;
    int i;
    f = fopen(filename,"w");
    fprintf(f, "P5\n%d %d\n%d\n", xsize, ysize, 255);
    for (i = 0; i < ysize; i++)
        fwrite(buf + i * wrap, 1, xsize, f);
    fclose(f);
}


int main( int argc, char* argv[] ) {

    /*
    int width = 640;
    int height = 480;
    uint8_t* rgb_img_ptr = malloc(sizeof(uint8_t)*width*height*4);
    uint8_t* nv12_img_ptr = malloc(sizeof(uint8_t)*width*height*3/2);

    for ( int i = 0; i < 10; i++ ) {
        for ( int y = 0; y < height; y++ ) {
            for ( int x = 0; x < width; x++ ) {
                int index = x + y * width;
                rgb_img_ptr[index*4] = 0;
                rgb_img_ptr[index*4+1] = x*2+y+i*12;//255;
                rgb_img_ptr[index*4+2] = x+y+12;
                rgb_img_ptr[index*4+3] = x+y*2+i*2;
            }
        }
        SaveBufferARGB(rgb_img_ptr, width, height, "original_rgb.pgm");
        formatConversion(width, height, rgb_img_ptr, 
                        nv12_img_ptr, NVPIPE_IMAGE_FORMAT_CONVERSION_ARGB_TO_NV12);
        
        SaveBufferNV12(nv12_img_ptr, width, height, "nv12.pgm");
        
        //formatConversion(width, height, nv12_img_ptr, 
        //                rgb_img_ptr, NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_ARGB);
        formatConversion(width, height, nv12_img_ptr, 
                        rgb_img_ptr, NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGB);
        
        SaveBufferRGB(rgb_img_ptr, width, height, "decoded_rgb.pgm");
    }
    free (rgb_img_ptr);
    free (nv12_img_ptr);
    return 0;
    */

    /*
    nvpipe* codec = nvpipe_create_instance(NVPIPE_CODEC_ID_H264);

    int width=640;
    int height=480;
    size_t buffer_size = sizeof(uint8_t)*width*height*3/2;
    printf("%zu is the size;", buffer_size);
    void* img_buffer = pa_alloc(buffer_size);
    size_t img_buffer_size = buffer_size;
    uint8_t* img_ptr0 = img_buffer;
    uint8_t* img_ptr1 = img_ptr0 + (width*height);
    uint8_t* img_ptr2 = img_ptr1 + (width*height) / 4;
    
    int gap0 = width;
    int gap1 = width/2;
    int gap2 = width/4;
    

    void* pkt_buffer = pa_alloc(buffer_size);
    size_t pkt_buffer_size = buffer_size;

    for ( int i = 0; i < 20; i++ ) {
        
        
        pkt_buffer_size = buffer_size;
        img_buffer_size = buffer_size;

        if ( i < 10 ) {
            width = 640;
            height = 480;
        } else {
            width = 320;
            height = 240;
        }

        for(size_t y=0;y<height;y++) {
            for(size_t x=0;x<width;x++) {
                    img_ptr0[y * width + x] = (x + y + i*10);
            }
        }
 
        for(size_t y=0;y<height/2;y++) {
            for(size_t x=0;x<width/2;x++) {
                img_ptr1[y * gap1 + x] = 128 + y;
                img_ptr2[y * gap2 + x] = 64 + x;
            }
    }*/

    nvpipe* codec = nvpipe_create_instance(NVPIPE_CODEC_ID_H264);
    nvpipe* codec2 = nvpipe_create_instance(NVPIPE_CODEC_ID_H264);
    int width=640;
    int height=480;
    size_t buffer_size = sizeof(uint8_t)*width*height*3;
    void* img_buffer = malloc(buffer_size);
    //void* img_buffer = pa_alloc(buffer_size);
    size_t img_buffer_size = buffer_size;
    uint8_t* img_ptr0 = img_buffer;

    //void* pkt_buffer = pa_alloc(buffer_size);
    void* pkt_buffer = malloc(buffer_size);
    size_t pkt_buffer_size = buffer_size;

    for ( int i = 0; i < 10; i++ ) {

        pkt_buffer_size = buffer_size;
        img_buffer_size = buffer_size;

        //if ( i > 0 ) {
        if ( 1 ) {
            width = 640;
            height = 480;
        } else {
            width = 320;
            height = 240;
        }

        for(size_t y=0;y<height;y++) {
            for(size_t x=0;x<width;x++) {
                int index = y * width + x;
                img_ptr0[index*3] = (x + y + i*10);
                img_ptr0[index*3+1] = 128 + y + i *15;
                img_ptr0[index*3+2] = 64 + x;
            }
        }

        char str[15];
        
        int num = i % 100;
        
        sprintf(str, "encoded_file%d.pgm", num);
        SaveBufferRGB(img_buffer, width, height, str);
        nvpipe_encode(codec, img_buffer, buffer_size, pkt_buffer, &pkt_buffer_size, width, height, NVPIPE_IMAGE_FORMAT_RGB);
        printf( "frame: %d, packet size: %zu\n", i, pkt_buffer_size);
        if (nvpipe_decode(codec, pkt_buffer, pkt_buffer_size, img_buffer, &img_buffer_size, &width, &height, NVPIPE_IMAGE_FORMAT_RGB) == 0 ) {
        //if (nvpipe_decode(codec, pkt_buffer, pkt_buffer_size, img_buffer, &img_buffer_size, &width, &height, NVPIPE_IMAGE_FORMAT_NV12) == 0 ) {
            sprintf(str, "decoded_file%d.pgm", i);
            SaveBufferRGB(img_buffer, width, height, str);
            //SaveBufferNV12(img_buffer, width, height, str);
            
            //formatConversion(width, height, img_buffer, pkt_buffer, NVPIPE_IMAGE_FORMAT_CONVERSION_NV12_TO_RGB);
            //sprintf(str, "decoded_file_conv%d.pgm", i);
            //SaveBufferRGB(pkt_buffer, width, height, str);

        } else {
            printf("decoding frame not written to file: %d\n", i);
        }

    }

    nvpipe_destroy_instance(codec);
    nvpipe_destroy_instance(codec2);

    free(img_buffer);
    free(pkt_buffer);

    return 0;
}
