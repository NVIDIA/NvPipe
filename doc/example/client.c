#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 

#include <inttypes.h>
#include <getopt.h>
#include "nvpipe.h"
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>


void error(const char *msg)
{
    perror(msg);
    exit(0);
}

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

void doStuff(const int newsockfd)
{
    nvpipe* codec = nvpipe_create_instance(NVPIPE_CODEC_ID_H264_HARDWARE);
    int width = 100;
    int height = 50;
    int count = 50;
    int sync = 1;
        
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
    int sockfd, portno;
    struct sockaddr_in serv_addr;
    struct hostent *server;

    if (argc < 3) {
       fprintf(stderr,"usage %s hostname port\n", argv[0]);
       exit(0);
    }
    portno = atoi(argv[2]);
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
        error("ERROR opening socket");
    server = gethostbyname(argv[1]);
    if (server == NULL) {
        fprintf(stderr,"ERROR, no such host\n");
        exit(0);
    }
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr_list[0], 
         (char *)&serv_addr.sin_addr.s_addr,
         server->h_length);
    serv_addr.sin_port = htons(portno);
    if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0) 
        error("ERROR connecting");

    doStuff(sockfd);    

    close(sockfd);
    return 0;
}
