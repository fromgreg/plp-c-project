/* util: fatal-error helper and allocation wrappers that never return NULL. */
#ifndef CGRAD_UTIL_H
#define CGRAD_UTIL_H

#include <stddef.h>

/* Print "cgrad: fatal: <message>" to stderr and abort. printf-style format. */
_Noreturn void fatal(const char *fmt, ...);

/* malloc/calloc/realloc that abort via fatal() instead of returning NULL.
 * Ownership is the same as the standard functions: the caller frees with free(). */
void *xmalloc(size_t size);
void *xcalloc(size_t count, size_t size);
void *xrealloc(void *ptr, size_t size);

#endif /* CGRAD_UTIL_H */
