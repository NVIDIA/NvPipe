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
#define _POSIX_C_SOURCE 201112L
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>
#include <strings.h>
#include <sys/types.h>
#include <unistd.h>
#include "debug.h"

static long pid = -1;
static bool color_enabled = false;

__attribute__((constructor(101))) static void
fp_dbg_init() {
	pid = (long)getpid();
	color_enabled = isatty(STDOUT_FILENO) == 1;
#if 0
	fprintf(stderr, "debugging setup: pid %ld, %s color\n", pid,
	        color_enabled ? "" : "no");
#endif
}

static bool
dbgchannel_enabled(const struct nvdbgchannel *chn,
                   enum _nvDbgChannelClass c) {
	return (chn->flags & (1U << c)) > 0;
}

/* ANSI escape codes for colors. */
static const char* C_NORM   = "\033[00m";
static const char* C_RED    = "\033[01;31m";
static const char* C_YELLOW = "\033[01;33m";
static const char* C_LBLUE  = "\033[01;36m";
static const char* C_WHITE = "\033[01;27m";
/* Might want these in the future ... */
#if 0
static const char* C_DGRAY = "\033[01;30m";
static const char* C_GREEN = "\033[01;32m";
static const char* C_MAG   = "\033[01;35m";
#endif

static const char*
color(const enum _nvDbgChannelClass cls) {
	if(!color_enabled) {
		return "";
	}
	switch (cls) {
	case Trace:
		return C_WHITE;
	case Warn:
		return C_YELLOW;
	case Err:
		return C_RED;
	case Fixme:
		return C_LBLUE;
	}
	assert(false);
	return C_NORM;
}

void
nv_dbg(enum _nvDbgChannelClass type, const struct nvdbgchannel *channel,
       const char* func, const char* format, ...) {
	va_list args;
	va_start(args, format);
	if(dbgchannel_enabled(channel, type)) {
		const char* fixit = type == Fixme ? "-FIXME" : "";
		printf("%s[%ld](%s%s) ", color(type), pid, func, fixit);
		(void)vprintf(format, args);
		printf("%s\n", color_enabled ? C_NORM : "");
	}
	va_end(args);
}

/* maps a string name to a class.  there should be a one-to-one mapping from
 * every entry in 'enum _nvDbgChannelClass' to this. */
static enum _nvDbgChannelClass
name_class(const char* name) {
	if(strncasecmp(name, "err", 3) == 0) {
		return Err;
	}
	if(strncasecmp(name, "warn", 4) == 0) {
		return Warn;
	}
	if(strncasecmp(name, "trace", 5) == 0) {
		return Trace;
	}
	if(strncasecmp(name, "fixme", 5) == 0) {
		return Fixme;
	}
	assert(false);
	/* hack.  what do we do if they give us a class that isn't defined?  well,
	 * since we use this to find the flag's position by bit-shifting, let's just
	 * do something we know will shift off the end of our flag sizes.  that way,
	 * undefined classes are just silently ignored. */
	return 64;
}

/* parses options of the form "chname=+a,-b,+c;chname2=+d,-c". */
void
nv_parse_options(struct nvdbgchannel *ch, const char* opt) {
	_Static_assert(sizeof(enum _nvDbgChannelClass) <= sizeof(unsigned),
	               "to make sure we can't shift beyond flags");
	/* special case: if the environment variable is simply "1", then turn
	 * everything on. */
	if(opt && strcmp(opt, "1") == 0) {
		ch->flags = (1U << Err) | (1U << Warn) | (1U << Fixme) | (1U << Trace);
		return;
	}
	/* outer loop iterates over channels.  channel names are separated by ';' */
	for(const char* chan = opt; chan && chan != (const char*)0x1;
	    chan = strchr(chan, ';') + 1) {
		/* extract a substring to make parsing easier. */
		char* chopts = strdup(chan);
		{ /* if there's another channel after, cut the string there. */
			char* nextopt = strchr(chopts, ';');
			if(nextopt) {
				*nextopt = '\0';
			}
		}
		if(strncmp(chopts, ch->name, strlen(ch->name)) == 0) {
			/* matched our channel name.  now we want to parse the list of options,
			 * separated by commas, e.g.: "+x,-y,+blah,+abc" */
			for(char* olist = strchr(chopts, '=') + 1;
			    olist && olist != (const char*)0x1; olist = strchr(olist, ',') + 1) {
				/* the "+1" gets rid of the minus or plus */
				enum _nvDbgChannelClass cls = name_class(olist + 1);
				/* temporarily null out the subsequent options, for printing. */
				char* optend = strchr(olist, ',');
				if(optend) {
					*optend = '\0';
				}
				if(*olist == '+') {
					fprintf(stderr, "[%ld] %s: enabling %s\n", pid, ch->name, olist + 1);
					ch->flags |= (1U << (uint16_t) cls);
				} else if(*olist == '-') {
					fprintf(stderr, "[%ld] %s: disabling %s\n", pid, ch->name, olist + 1);
					ch->flags &= ~(1U << (uint16_t) cls);
				}
				/* 'de-null' it. */
				if(optend) {
					*optend = ',';
				}
			}
		}
		free(chopts);
	}
}
