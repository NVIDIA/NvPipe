#include <stdio.h>
#include "nvpipe.h"

int main( int argc, char* argv[] ) {
    printf("hello world!\n");

    nvpipe* codec = nvpipe_create_instance(NVPIPE_CODEC_ID_H264);

    int width, height, buffer_size;

    nvpipe_encode(codec, NULL, NULL, width, height, &buffer_size);
    nvpipe_decode(codec, NULL, NULL, &width, &height, buffer_size);

    nvpipe_destroy_instance(codec);

    return 0;
}
