#include "util.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

_Noreturn void fatal(const char *fmt, ...)
{
    va_list args;

    fputs("cgrad: fatal: ", stderr);
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    /* abort() rather than exit(): ASan and debuggers then show where we died. */
    abort();
}

/* A request for 0 bytes is bumped to 1: malloc(0) may legally return NULL,
 * which would be indistinguishable from out-of-memory. */

void *xmalloc(size_t size)
{
    void *p = malloc(size ? size : 1);
    if (p == NULL)
        fatal("out of memory (malloc of %zu bytes)", size);
    return p;
}

void *xcalloc(size_t count, size_t size)
{
    /* calloc itself checks count * size for overflow and returns NULL. */
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (p == NULL)
        fatal("out of memory (calloc of %zu x %zu bytes)", count, size);
    return p;
}

void *xrealloc(void *ptr, size_t size)
{
    void *p = realloc(ptr, size ? size : 1);
    if (p == NULL)
        fatal("out of memory (realloc to %zu bytes)", size);
    return p;
}
