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


void error(const char *msg)
{
    perror(msg);
    exit(1);
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

void doStuff(const int sockfd)
{
    nvpipe* codec = nvpipe_create_instance(NVPIPE_CODEC_ID_H264_HARDWARE);
    //int width = 1920;
    //int height = 1080;
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
        //if ( i != 0) nvtxRangePushA("receiving end");
        if (sync) {
            pkt_buffer_size = read(sockfd,pkt_buffer, buffer_size);
            write(sockfd, &pkt_buffer_size, sizeof(pkt_buffer_size));
        } else {
            read(sockfd,&pkt_buffer_size,sizeof(pkt_buffer_size));
            read(sockfd,pkt_buffer,pkt_buffer_size);
        }
        printf("pkt_buffer_size: %zu\n", pkt_buffer_size);

        if ( !nvpipe_decode(codec, pkt_buffer, pkt_buffer_size, img_buffer, img_buffer_size, &width, &height, NVPIPE_IMAGE_FORMAT_RGBA) ) {
            //sprintf(image_filename, "decoded_%d.pgm", i);
            //SaveBufferRGBA(img_buffer, width, height, image_filename);
        } else {
            printf("something went wrong\n");
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

     doStuff(newsockfd);

     close(newsockfd);
     close(sockfd);
     return 0; 
}

