/* Test runner.
 *
 *   test_runner            run every test
 *   test_runner <filter>   run tests whose suite or test name contains <filter>
 *
 * Exit status is non-zero if a test failed or the filter matched nothing, so
 * make and CI notice. */
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "test.h"

static const TestSuite *const suites[] = {
    &util_suite,
    &rng_suite,
    &tensor_suite,
};

/* Failures recorded by the test that is currently running. */
static int current_failures;

void test_fail(const char *file, int line, const char *fmt, ...)
{
    va_list args;

    current_failures++;
    fprintf(stderr, "    %s:%d: ", file, line);
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
}

static int matches(const char *filter, const TestSuite *suite, const TestCase *tc)
{
    return filter == NULL || strstr(suite->name, filter) != NULL
        || strstr(tc->name, filter) != NULL;
}

int main(int argc, char **argv)
{
    const char *filter = argc > 1 ? argv[1] : NULL;
    int run = 0, failed = 0;

    for (size_t s = 0; s < ARRAY_LEN(suites); s++) {
        const TestSuite *suite = suites[s];
        for (size_t i = 0; i < suite->count; i++) {
            const TestCase *tc = &suite->tests[i];
            if (!matches(filter, suite, tc))
                continue;

            current_failures = 0;
            tc->fn();
            run++;
            if (current_failures > 0)
                failed++;
            /* Flush so PASS/FAIL lines interleave correctly with the
             * failure details that test_fail writes to stderr. */
            printf("%s %s/%s\n", current_failures ? "FAIL" : "PASS",
                   suite->name, tc->name);
            fflush(stdout);
        }
    }

    printf("%d tests run, %d failed\n", run, failed);
    if (run == 0) {
        fprintf(stderr, "no test matches filter '%s'\n", filter ? filter : "");
        return 1;
    }
    return failed > 0;
}
