#include "doc/example/file_io.h"

typedef std::vector<memoryObj>::iterator mem_obj_iter;

void* pa_alloc(size_t sz) {
    void* rv;
    const int err = posix_memalign(&rv, 4096, sz);
    if(0 != err) {
        fprintf(stderr, "%zu-byte allocation failed.\n", sz);
        return NULL;
    }
    return rv;
}

void SaveBufferRGB(uint8_t *data, int width, int height, const char *str) {
    FILE *pFile;

    // Open file
    pFile=fopen(str, "wb");
    if(pFile==NULL)
        return;

    // Write header
    fprintf(pFile, "P6\n%d %d\n255\n", width, height);

    // Write pixel data
    for(int y=0; y<height; y++) {
        fwrite(data, 1, width*3, pFile);
        data += width*3;
    }
    // Close file
    fclose(pFile);
}

void SaveBufferBit(uint8_t *data, size_t length, const char *str) {
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

MemoryStack::MemoryStack(uint8_t *buffer, size_t buffer_size) {
    initialize(buffer, buffer_size);
}

void MemoryStack::initialize(uint8_t *buffer, size_t buffer_size) {
    buffer_ = buffer;
    buffer_size_ = buffer_size;
    used_buffer_ = 0;
    stackItemVector_.clear();

    memoryObj mem_obj;
    mem_obj.ptr_ = buffer;
    mem_obj.size_ = 0;
    stackItemVector_.push_back(mem_obj);
}

uint8_t* MemoryStack::getBufferHandle() {
    return stackItemVector_.back().ptr_;
}

uint8_t* MemoryStack::getBufferHandle( int index ) {
    if ( index >= stackItemVector_.size() || index < 0 )
        return NULL;
    return stackItemVector_[index].ptr_;
}

int MemoryStack::getSize() {
    return stackItemVector_.size()-1;
}

size_t MemoryStack::getBufferSize( int index ) {
    if ( index >= stackItemVector_.size()-1 || index < 0 )
        return 0;
    return stackItemVector_[index].size_;
}

size_t MemoryStack::getRemainingSpace() {
    return buffer_size_ - used_buffer_;
}

size_t MemoryStack::getUsedSpace() {
    return used_buffer_;
}

uint8_t* MemoryStack::pushBuffer(size_t size) {

    if ( used_buffer_ + size > buffer_size_ ) {
        return NULL;
    }

    stackItemVector_.back().size_ = size;
    
    memoryObj mem_obj;
    mem_obj.ptr_ = stackItemVector_.back().ptr_ + size;
    mem_obj.size_ = 0;
    used_buffer_ += size;
    stackItemVector_.push_back(mem_obj);
    return stackItemVector_.back().ptr_;
}

uint8_t* MemoryStack::popBuffer() {
    int size = stackItemVector_.size();
    if ( size > 1 ) {
        stackItemVector_.pop_back();
        used_buffer_ -= stackItemVector_.back().size_;
        return stackItemVector_.back().ptr_;
    } else {
        return NULL;
    }
}

void MemoryStack::writeBufferToFile(std::string file_name) {
    SaveBufferBit( buffer_, used_buffer_, file_name.c_str() );
}

void MemoryStack::writeBufferToFileList( std::string file_name,
                            enum Buffer_Type buffer_type,
                            int width, int height) {
    char str[50];
    for ( int i = 0; i < getSize(); i++ ) {
        switch(buffer_type) {
        case RGB_PICTURE:
            sprintf(str, "%s_%d.pgm", file_name.c_str(), i);
            SaveBufferRGB(  stackItemVector_[i].ptr_, 
                            width, height, 
                            str);
            break;
        case PACKET_DATA:
            sprintf(str, "%s_%d.pkt", file_name.c_str(), i);
            SaveBufferBit(  stackItemVector_[i].ptr_,
                            stackItemVector_[i].size_,
                            str);
            break;
        }
    }
}
