/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <NvPipe.h>

#include "utils.h"

#include <iostream>
#include <vector>

#include <EGL/egl.h>
#include <GL/glew.h>


void captureFramebufferPPM(GLuint framebuffer, uint32_t width, uint32_t height, const std::string& path)
{
    std::vector<uint8_t> rgba(width * height * 4);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());

    savePPM(rgba.data(), width, height, path);
}


int main(int argc, char* argv[])
{
    std::cout << "NvPipe example application: Render to offscreen framebuffer using EGL," << std::endl << "encode framebuffer, decode to display texture." << std::endl << std::endl;

    const uint32_t width = 3840;
    const uint32_t height = 2160;

    const NvPipe_Codec codec = NVPIPE_H264;
    const float bitrateMbps = 32;
    const uint32_t targetFPS = 90;


    std::cout << "Resolution: " << width << " x " << height << std::endl;
    std::cout << "Codec: " << (codec == NVPIPE_H264 ? "H.264" : "HEVC") << std::endl;
    std::cout << "Bitrate: " << bitrateMbps << " Mbps @ " << targetFPS << " Hz" << std::endl;



    // Init EGL/OpenGL
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


    // Create framebuffer for offscreen rendering
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


    // Create shader and texture for fullscreen display of decompressed frame
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


    // Create encoder
    NvPipe* encoder = NvPipe_CreateEncoder(NVPIPE_RGBA32, codec, NVPIPE_LOSSY, bitrateMbps * 1000 * 1000, targetFPS, width, height);
    if (!encoder)
        std::cerr << "Failed to create encoder: " << NvPipe_GetError(NULL) << std::endl;

    // Create decoder
    NvPipe* decoder = NvPipe_CreateDecoder(NVPIPE_RGBA32, codec, width, height);
    if (!decoder)
        std::cerr << "Failed to create decoder: " << NvPipe_GetError(NULL) << std::endl;


    Timer timer;
    std::cout << std::endl << "Frame | Encode (ms) | Decode (ms) | Size (KB)" << std::endl;

    for (uint32_t i = 0; i < 10; ++i)
    {
        // Render dummy scene (Nothing to see here; just some oldschool immediate mode.. urgh)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, serverFBO);
        glUseProgram(0);

        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glViewport(0, 0, width, height);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glRotatef((float) i, 0.0f, 0.0f, 1.0f);

        glBegin(GL_TRIANGLES);
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, -0.9f, 0.0f);
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(-0.9f,0.9f, 0.0f);
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0.9f, 0.9f, 0.0f);
        glEnd();

        glFinish(); // Make sure rendering is complete before grabbing frame

        captureFramebufferPPM(serverFBO, width, height, "egl-input" + std::to_string(i) + ".ppm");

        // Encode
        std::vector<uint8_t> compressed(width * height * 4);

        timer.reset();
        uint64_t size = NvPipe_EncodeTexture(encoder, serverColorTex, GL_TEXTURE_2D, compressed.data(), compressed.size(), width, height, false);
        double encodeMs = timer.getElapsedMilliseconds();

        if (0 == size)
            std::cerr << "Encode error: " << NvPipe_GetError(encoder) << std::endl;


        // Decode
        timer.reset();
        uint64_t r = NvPipe_DecodeTexture(decoder, compressed.data(), size, clientColorTex, GL_TEXTURE_2D, width, height);
        double decodeMs = timer.getElapsedMilliseconds();

        if (0 == r)
            std::cerr << "Decode error: " << NvPipe_GetError(decoder) << std::endl;

        double sizeKB = size / 1000.0;
        std::cout << std::fixed << std::setprecision(1) << std::setw(5) << i << " | " << std::setw(11) << encodeMs << " | " <<  std::setw(11) << decodeMs << " | " <<  std::setw(8) << sizeKB << std::endl;



        // Display decoded frame
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glUseProgram(clientFullscreenQuadProgram);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, clientColorTex);
        glUniform1i(clientFullscreenTextureLocation, 0);
        glBindVertexArray(clientFullscreenVAO);
        glDrawArrays(GL_POINTS, 0, 1);

        captureFramebufferPPM(0, width, height, "egl-output" + std::to_string(i) + ".ppm");
    }

    // Clean up
    NvPipe_Destroy(encoder);
    NvPipe_Destroy(decoder);

    eglTerminate(display);

    return 0;
}
