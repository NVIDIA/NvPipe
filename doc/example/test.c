#include <stdio.h>
#include "nvpipe.h"

int main( int argc, char* argv[] ) {
    nvpipecodec* codec = nvpipe_create_encoder_nvenc();
    printf("hello world!\n");
    return 0;
}
