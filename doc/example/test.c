#include <stdio.h>
#include <inttypes.h>
#include "nvpipe.h"

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


void SaveBuffer(uint8_t *data, int width, int height, char *str) {
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
    printf("hello world!\n");

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
        
        width = 640;
        height = 480;
    
        for(size_t y=0;y<height;y++) {
            for(size_t x=0;x<width;x++) {
                    img_ptr0[y * width + x] = (x + y + i*10);
            }
        }

            /* Cb and Cr */
            for(size_t y=0;y<height/2;y++) {
                for(size_t x=0;x<width/2;x++) {
                    img_ptr1[y * gap1 + x] = 128 + y;
                    img_ptr2[y * gap2 + x] = 64 + x;
            }
        }
        
        char str[15];
        sprintf(str, "encoded_file%d.pgm", i);
        SaveBuffer(img_buffer, width, height, str);
        
        nvpipe_encode(codec, img_buffer, buffer_size, pkt_buffer, &pkt_buffer_size, width, height, NVPIPE_IMAGE_FORMAT_YUV420P);
        
        if (nvpipe_decode(codec, pkt_buffer, pkt_buffer_size, img_buffer, &img_buffer_size, &width, &height, NVPIPE_IMAGE_FORMAT_YUV420P) != 0 ) {
            sprintf(str, "decoded_file%d.pgm", i);
            SaveBuffer(img_buffer, width, height, str);
            printf("Next\n");
        }

    }
    nvpipe_destroy_instance(codec);
    
    free(img_buffer);
    free(pkt_buffer);

    return 0;
}
