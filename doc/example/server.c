#if __STDC_VERSION__ >= 199901L
#define _XOPEN_SOURCE 600
#else
#define _XOPEN_SOURCE 500
#endif /* __STDC_VERSION__ */

/* A simple server in the internet domain using TCP
   The port number is passed as an argument */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>

#include <inttypes.h>
#include <getopt.h>
#include "nvpipe.h"
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

#include <time.h>


static int pid = 0;

void error(const char *msg)
{
    perror(msg);
    exit(1);
}

#define BILLION 1000000000L

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

void doStuff(const int sockfd)
{
    nvpipe* codec = nvpipe_create_instance(NVPIPE_CODEC_ID_H264_HARDWARE);
    int width = 1920;
    int height = 1080;
    //int width = 100;
    //int height = 50;
    int count = 500;
    int sync = 0;

    size_t buffer_size = sizeof(uint8_t)*width*height*4;
    void* img_buffer = malloc(buffer_size);
    size_t img_buffer_size = buffer_size;
    uint8_t* img_ptr0 = img_buffer;
    void* pkt_buffer = malloc(buffer_size);
    size_t pkt_buffer_size = buffer_size;
    char image_filename[20];
    
    uint64_t diff; 
    struct timespec start_clock, end_clock;
    struct timespec start_cpu, end_cpu;
    clock_t begin_time;

    for (int i = 0; i < count; i++ ) {

        if ( i == 50 ) {
            clock_gettime(CLOCK_MONOTONIC, &start_clock);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_cpu);
            begin_time = clock();
        }
        if ( i == 450 ) {
            clock_gettime(CLOCK_MONOTONIC, &end_clock);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_cpu);
            begin_time = clock() - begin_time;
        }
            
        
        //if ( i != 0) nvtxRangePushA("receiving end");
        if (sync) {
            pkt_buffer_size = read(sockfd,pkt_buffer, buffer_size);
            write(sockfd, &pkt_buffer_size, sizeof(pkt_buffer_size));
        } else {
            read(sockfd,&pkt_buffer_size,sizeof(pkt_buffer_size));
            size_t cur_pkt_size = pkt_buffer_size;
            //printf("pkt_buffer_size: %zu\n", pkt_buffer_size);
            while ( cur_pkt_size > 0 ) {
                size_t pkt_size_cur = read( sockfd,
                                            pkt_buffer + pkt_buffer_size - cur_pkt_size ,
                                            cur_pkt_size);
                //printf("    cur_pkt_buffer_size: %zu\n", pkt_size_cur);
                cur_pkt_size -= pkt_size_cur;
            }
        }
        //printf("pkt_buffer_size: %zu\n", pkt_buffer_size);

        if ( !nvpipe_decode(codec, pkt_buffer, pkt_buffer_size, img_buffer, img_buffer_size, &width, &height, NVPIPE_IMAGE_FORMAT_RGBA) ) {
            //if ( i <= 5 ) {
            if ( 1 ) {
                sprintf(image_filename, "decoded_%d.pgm", pid);
                SaveBufferRGBA(img_buffer, width, height, image_filename);
            }
        } else {
            printf("something went wrong\n");
        }
        //if ( i != 0) nvtxRangePop();
    }
    float number = begin_time;
    printf ("%f",  number / CLOCKS_PER_SEC);

    diff = BILLION * (end_clock.tv_sec - start_clock.tv_sec) + 
            end_clock.tv_nsec - start_clock.tv_nsec;
    printf("clock elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);

    diff = BILLION * (end_cpu.tv_sec - start_cpu.tv_sec) + 
            end_cpu.tv_nsec - start_cpu.tv_nsec;
    printf("cpu elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);

    nvpipe_destroy_instance(codec);
    free(img_buffer);
    free(pkt_buffer);
    return 0;

}


void doStuff_send(const int newsockfd) {

    nvpipe* codec = nvpipe_create_instance(NVPIPE_CODEC_ID_H264_HARDWARE);
    int width = 1920;
    int height = 1080;
    int count = 500;
    int sync = 0;
        
    size_t buffer_size = sizeof(uint8_t)*width*height*4;
    void* img_buffer = malloc(buffer_size);
    size_t img_buffer_size = buffer_size;
    uint8_t* img_ptr0 = img_buffer;
    void* pkt_buffer = malloc(buffer_size);
    size_t pkt_buffer_size = buffer_size;
    char image_filename[20];
    
    for (int i = 0; i < count; i++ ) {
        //if ( i != 0) nvtxRangePushA("sending end");
        pkt_buffer_size = buffer_size;
        for(size_t y=0;y<height;y++) {
            for(size_t x=0;x<width;x++) {
                int index = y * width + x;
                img_ptr0[index*4] = x+y+i*5;//x+y;
                img_ptr0[index*4+1] = x+i*10;//x;
                img_ptr0[index*4+2] = 0;
                img_ptr0[index*4+3] = 255;
            }
        }

        //sprintf(image_filename, "original_%d.pgm", i);
        //SaveBufferRGBA(img_buffer, width, height, image_filename);

        if ( !nvpipe_encode(codec, img_buffer, buffer_size, pkt_buffer, &pkt_buffer_size, width, height, NVPIPE_IMAGE_FORMAT_RGBA) ) {
            printf("pkt_buffer_size: %zu\n", pkt_buffer_size);
            if ( sync ) {
                write(newsockfd,pkt_buffer,pkt_buffer_size);
                read(newsockfd,&pkt_buffer_size,sizeof(pkt_buffer_size));
            } else {
                write(newsockfd,&pkt_buffer_size,sizeof(pkt_buffer_size));
                write(newsockfd,pkt_buffer,pkt_buffer_size);
            }
        } else {
            printf("what happened?\n");
        }
        //if ( i != 0) nvtxRangePop();
    }

    nvpipe_destroy_instance(codec);
    free(img_buffer);
    free(pkt_buffer);
    return 0;

}

int main(int argc, char *argv[])
{
     int sockfd, newsockfd, portno;
     socklen_t clilen;
     
     struct sockaddr_in serv_addr, cli_addr;
     if (argc < 2) {
         fprintf(stderr,"ERROR, no port provided\n");
         exit(1);
     }
     sockfd = socket(AF_INET, SOCK_STREAM, 0);
     if (sockfd < 0) 
        error("ERROR opening socket");
     bzero((char *) &serv_addr, sizeof(serv_addr));
     portno = atoi(argv[1]);
     pid = portno;
     serv_addr.sin_family = AF_INET;
     serv_addr.sin_addr.s_addr = INADDR_ANY;
     serv_addr.sin_port = htons(portno);
     if (bind(sockfd, (struct sockaddr *) &serv_addr,
              sizeof(serv_addr)) < 0) 
              error("ERROR on binding");
     listen(sockfd,5);
     clilen = sizeof(cli_addr);
     newsockfd = accept(sockfd, 
                 (struct sockaddr *) &cli_addr, 
                 &clilen);
     if (newsockfd < 0) 
          error("ERROR on accept");

     //doStuff(newsockfd);
     doStuff_send(newsockfd);

     close(newsockfd);
     close(sockfd);
     return 0; 
}

