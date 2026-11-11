/* Tests for util: allocation wrappers. The abort-on-NULL path cannot be
 * triggered in-process without taking the runner down with it. */
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "util.h"

static void xmalloc_usable(void)
{
    size_t n = 64;
    unsigned char *p = xmalloc(n);
    memset(p, 0xAB, n);
    CHECK(p[0] == 0xAB && p[n - 1] == 0xAB);
    free(p);
}

static void xmalloc_zero_bytes(void)
{
    void *p = xmalloc(0);
    CHECK(p != NULL);
    free(p);
}

static void xcalloc_zeroes(void)
{
    size_t n = 100;
    int *p = xcalloc(n, sizeof *p);
    int all_zero = 1;
    for (size_t i = 0; i < n; i++)
        all_zero = all_zero && p[i] == 0;
    CHECK(all_zero);
    free(p);
}

static void xrealloc_preserves_prefix(void)
{
    int *p = xmalloc(4 * sizeof *p);
    for (int i = 0; i < 4; i++)
        p[i] = i * 10;

    p = xrealloc(p, 1000 * sizeof *p);
    CHECK(p[0] == 0 && p[1] == 10 && p[2] == 20 && p[3] == 30);
    free(p);
}

static const TestCase tests[] = {
    {"xmalloc_usable", xmalloc_usable},
    {"xmalloc_zero_bytes", xmalloc_zero_bytes},
    {"xcalloc_zeroes", xcalloc_zeroes},
    {"xrealloc_preserves_prefix", xrealloc_preserves_prefix},
};

const TestSuite util_suite = {"util", tests, ARRAY_LEN(tests)};
