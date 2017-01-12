/*
 * Copyright (c) 2016 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual
 * property and proprietary rights in and to this software,
 * related documentation and any modifications thereto.  Any use,
 * reproduction, disclosure or distribution of this software and
 * related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE
 * IS PROVIDED *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL
 * WARRANTIES, EITHER EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED
 * TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE
 * LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL
 * DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
 * LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
 * INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF
 * OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGES
 */
#ifndef NVPIPE_DEBUG_H
#define NVPIPE_DEBUG_H
/** Simple debug channel implementation.  Example usage:
 *
 *   DECLARE_CHANNEL(stuff);
 *   TRACE(stuff, "is happening!");
 *   ERR(stuff, "something really bad happened.");
 *   WARN(stuff, "i think something's wrong?");
 *
 * The user can enable/disable the above channel by setting the NVPIPE_VERBOSE
 * environment variable:
 *
 *   export NVPIPE_VERBOSE="stuff=+err,-warn,+trace" */
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

enum _nvDbgChannelClass {
	Err=0,
	Warn,
	Trace,
	Fixme,
};

struct nvdbgchannel {
	unsigned flags;
	char name[32];
};

#define DEFAULT_CHFLAGS \
	(1U << Err) | (1U << Warn) | (1U << Fixme)
/* creates a new debug channel.  debug channels are private to implementation,
 * and must not be declared in header files. */
#define DECLARE_CHANNEL(ch) \
	static struct nvdbgchannel nv_chn_##ch = { DEFAULT_CHFLAGS, #ch }; \
	__attribute__((constructor(200))) static void \
	ch_init_##ch() { \
		const char* dbg_ = getenv("NVPIPE_VERBOSE"); \
		nv_parse_options(&nv_chn_##ch, dbg_); \
	}

#define TRACE(ch, args...) \
	nv_dbg(Trace, &nv_chn_##ch, __FUNCTION__, args)
#define ERR(ch, args...) \
	nv_dbg(Err, &nv_chn_##ch, __FUNCTION__, args)
#define WARN(ch, args...) \
	nv_dbg(Warn, &nv_chn_##ch, __FUNCTION__, args)
#define FIXME(ch, args...) \
	nv_dbg(Fixme, &nv_chn_##ch, __FUNCTION__, args)

/* for internal use only. */
void nv_dbg(enum _nvDbgChannelClass, const struct nvdbgchannel*,
              const char* func, const char* format, ...)
              __attribute__((format(printf, 4, 5)));
void nv_parse_options(struct nvdbgchannel*, const char* opt);

#ifdef __cplusplus
}
#endif

#endif
