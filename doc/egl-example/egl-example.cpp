/* Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <EGL/egl.h>
#include <GL/glew.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstring>
#include <string>
#include <iostream>
#include <fstream>

#include "nvpipe.h"


void captureFramebufferPPM(GLuint framebuffer, uint32_t width, uint32_t height, const std::string& path)
{
    // For verification...

    size_t numBytes = width * height * 3;
    uint8_t* rgb = new uint8_t[numBytes];

    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgb);

    std::ofstream outFile;
    outFile.open(path.c_str(), std::ios::binary);

    outFile << "P6" << "\n"
            << width << " " << height << "\n"
            << "255\n";

    outFile.write((char*) rgb, numBytes);

    delete[] rgb;
}


int main(int argc, char* argv[])
{
    /*
     * General demo setup: Init EGL and OpenGL context
     */
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    EGLint major, minor;
    eglInitialize(display, &major, &minor);

    const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };

    EGLint numConfigs;
    EGLConfig config;
    eglChooseConfig(display, configAttribs, &config, 1, &numConfigs);

    const uint32_t width = 1920;
    const uint32_t height = 1080;

    const EGLint pbufferAttribs[] = {
        EGL_WIDTH, width,
        EGL_HEIGHT, height,
        EGL_NONE,
    };

    EGLSurface surface = eglCreatePbufferSurface(display, config, pbufferAttribs);

    eglBindAPI(EGL_OPENGL_API);
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
    eglMakeCurrent(display, surface, surface, context);
    glewInit();



    /*
     * Server: Create render framebuffer and register for CUDA interop
     */
    GLuint serverColorTex;
    glGenTextures(1, &serverColorTex);
    glBindTexture(GL_TEXTURE_2D, serverColorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    GLuint serverDepthTex;
    glGenTextures(1, &serverDepthTex);
    glBindTexture(GL_TEXTURE_2D, serverDepthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

    GLuint serverFBO;
    glGenFramebuffers(1, &serverFBO);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, serverFBO);
    glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, serverColorTex, 0);
    glFramebufferTexture(GL_DRAW_FRAMEBUFFER,  GL_DEPTH_ATTACHMENT, serverDepthTex, 0);

    if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cerr << "Failed to create framebuffer" << std::endl;
        return 1;
    }

    cudaGraphicsResource_t serverGraphicsResource;
    cudaGraphicsGLRegisterImage(&serverGraphicsResource, serverColorTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);



    /*
     * Client: Create OpenGL texture for display
     */
    const GLchar* clientVertexShader =
            "#version 330\n"
            "void main() {}";

    const GLchar* clientGeometryShader =
            "#version 330 core\n"
            "layout(points) in;"
            "layout(triangle_strip, max_vertices = 4) out;"
            "out vec2 texcoord;"
            "void main() {"
            "gl_Position = vec4( 1.0, 1.0, 0.0, 1.0 ); texcoord = vec2( 1.0, 1.0 ); EmitVertex();"
            "gl_Position = vec4(-1.0, 1.0, 0.0, 1.0 ); texcoord = vec2( 0.0, 1.0 ); EmitVertex();"
            "gl_Position = vec4( 1.0,-1.0, 0.0, 1.0 ); texcoord = vec2( 1.0, 0.0 ); EmitVertex();"
            "gl_Position = vec4(-1.0,-1.0, 0.0, 1.0 ); texcoord = vec2( 0.0, 0.0 ); EmitVertex();"
            "EndPrimitive();"
            "}";

    const GLchar* clientFragmentShader =
            "#version 330\n"
            "uniform sampler2D tex;"
            "in vec2 texcoord;"
            "out vec4 color;"
            "void main() {"
            "	color = texture(tex, texcoord);"
            "}";

    GLuint clientVertexShaderHandle = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(clientVertexShaderHandle, 1, &clientVertexShader, 0);
    glCompileShader(clientVertexShaderHandle);

    GLuint clientGeometryShaderHandle = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(clientGeometryShaderHandle, 1, &clientGeometryShader, 0);
    glCompileShader(clientGeometryShaderHandle);

    GLuint clientFragmentShaderHandle = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(clientFragmentShaderHandle, 1, &clientFragmentShader, 0);
    glCompileShader(clientFragmentShaderHandle);

    GLuint clientFullscreenQuadProgram = glCreateProgram();
    glAttachShader(clientFullscreenQuadProgram, clientVertexShaderHandle);
    glAttachShader(clientFullscreenQuadProgram, clientGeometryShaderHandle);
    glAttachShader(clientFullscreenQuadProgram, clientFragmentShaderHandle);
    glLinkProgram(clientFullscreenQuadProgram);

    GLuint clientFullscreenTextureLocation = glGetUniformLocation(clientFullscreenQuadProgram, "tex");

    GLuint clientFullscreenVAO;
    glGenVertexArrays(1, &clientFullscreenVAO);

    GLuint clientColorTex;
    glGenTextures(1, &clientColorTex);
    glBindTexture(GL_TEXTURE_2D, clientColorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0); // must use RGBA(8) here for CUDA-GL interop
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    cudaGraphicsResource_t clientGraphicsResource;
    cudaGraphicsGLRegisterImage(&clientGraphicsResource, clientColorTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);



    /*
     * Server: Init encoder
     */
    const uint64_t bitrate = width * height * 30 * 4 * 0.07; // Kush gauge
    nvpipe* encoder = nvpipe_create_encoder(NVPIPE_H264_NV, bitrate);

    size_t serverDeviceBufferSize = width * height * 4;
    void* serverDeviceBuffer;
    if (cudaMalloc(&serverDeviceBuffer, serverDeviceBufferSize) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory" << std::endl;
        return 1;
    }

    size_t serverSendBufferSize = serverDeviceBufferSize; // Reserve enough space for encoded output
    uint8_t* serverSendBuffer = new uint8_t[serverSendBufferSize];



    /*
     * Client: Init decoder
     */
    nvpipe* decoder = nvpipe_create_decoder(NVPIPE_H264_NV);

    size_t clientDeviceBufferSize = width * height * 4;
    void* clientDeviceBuffer;
    if (cudaMalloc(&clientDeviceBuffer, clientDeviceBufferSize) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory" << std::endl;
        return 1;
    }

    size_t clientReceiveBufferSize = clientDeviceBufferSize; // Reserve enough space for input
    uint8_t* clientReceiveBuffer = new uint8_t[clientReceiveBufferSize];



    /*
     * Server: Render scene
     */
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, serverFBO);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Nothing to see here; just some oldschool immediate mode.. urgh
    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBegin(GL_TRIANGLES);
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, -0.9f, 0.0f);
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(-0.9f,0.9f, 0.0f);
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.9f, 0.9f, 0.0f);
    glEnd();

    // Export rendered image for verification
    captureFramebufferPPM(serverFBO, width, height, "1-server.ppm");



    /*
     * Server: Grab frame and encode
     */
    cudaGraphicsMapResources(1, &serverGraphicsResource);
    cudaArray_t serverArray;
    cudaGraphicsSubResourceGetMappedArray(&serverArray, serverGraphicsResource, 0, 0);
    cudaMemcpy2DFromArray(serverDeviceBuffer, width * 4, serverArray, 0, 0, width * 4, height, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &serverGraphicsResource);

    size_t numBytes = serverSendBufferSize;
    nvp_err_t encodeStatus = nvpipe_encode(encoder, serverDeviceBuffer, width * height * 4, serverSendBuffer, &numBytes, width, height, NVPIPE_RGBA);
    if (encodeStatus != NVPIPE_SUCCESS)
    {
        std::cerr << "Encode failed: " << std::string(nvpipe_strerror(encodeStatus)) << std::endl;
        return 1;
    }



    /*
     * Network transfer (e.g. over TCP socket)
     */
    // ... Send buffer size, buffer data, width, height, etc. (whatever you need)
    memcpy(clientReceiveBuffer, serverSendBuffer, numBytes); // Dummy



    /*
     * Client: Decode to OpenGL texture
     */
    nvp_err_t decodeStatus = nvpipe_decode(decoder, clientReceiveBuffer, numBytes, clientDeviceBuffer, width, height, NVPIPE_RGBA);
    if (decodeStatus != NVPIPE_SUCCESS)
    {
        std::cerr << "Decode failed: " << std::string(nvpipe_strerror(decodeStatus)) << std::endl;
        return 1;
    }

    cudaGraphicsMapResources(1, &clientGraphicsResource);
    cudaArray_t clientArray;
    cudaGraphicsSubResourceGetMappedArray(&clientArray, clientGraphicsResource, 0, 0);
    cudaMemcpy2DToArray(clientArray, 0, 0, clientDeviceBuffer, width * 4, width * 4, height, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &clientGraphicsResource);



    /*
     * Client: Display decoded frame
     */
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Only for verification

    glUseProgram(clientFullscreenQuadProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, clientColorTex);
    glUniform1i(clientFullscreenTextureLocation, 0);
    glBindVertexArray(clientFullscreenVAO);
    glDrawArrays(GL_POINTS, 0, 1);

    // Export rendered image for verification
    captureFramebufferPPM(0, width, height, "2-client.ppm");



    /*
     * Clean up
     */
    cudaFree(serverDeviceBuffer);
    delete[] serverSendBuffer;
    nvpipe_destroy(encoder);

    delete[] clientReceiveBuffer;
    cudaFree(clientDeviceBuffer);
    nvpipe_destroy(decoder);

    eglTerminate(display);

    return 0;
}
