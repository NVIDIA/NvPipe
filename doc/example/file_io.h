#pragma once

#include <cstdio>
#include <cinttypes>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <vector>

void* pa_alloc(size_t sz);

void SaveBufferRGB(uint8_t *data, int width, int height, const char *str);

void SaveBufferBit(uint8_t *data, size_t length, const char *str);

size_t ReadFromFile(std::string file_name, void *data, size_t size);

typedef struct _memoryObj {
    size_t size_;
    uint8_t *ptr_;
} memoryObj;

enum Buffer_Type {
    PLAIN_DATA,
    RGB_PICTURE,
    PACKET_DATA
};

class MemoryStack {
public:
    MemoryStack(uint8_t *buffer, size_t buffer_size);
    void initialize(uint8_t *buffer, size_t buffer_size);
    int getSize();
    size_t getRemainingSpace();
    size_t getUsedSpace();

    void writeBufferToFile(std::string file_name);
    void writeBufferToFileList( std::string file_name,
                                enum Buffer_Type buffer_type,
                                int width = 0, int height = 0);
    void loadBufferFromFileList( std::string file_name,
                                    enum Buffer_Type buffer_type,
                                    int length);

    uint8_t* getBufferHandle();
    uint8_t* getBufferHandle( int index );
    size_t getBufferSize( int index );
    uint8_t* pushBuffer(size_t size); // return next buffer handle
    uint8_t* popBuffer();

protected:

private:
    size_t buffer_size_;
    uint8_t *buffer_;
    std::vector<memoryObj> stackItemVector_;
    size_t used_buffer_;
};
