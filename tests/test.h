/* test: minimal hand-rolled test framework.
 *
 * A test is a void(void) function that uses CHECK / CHECK_NEAR. A failed check
 * records file:line and the expression, then the test keeps running, so one
 * run reports every failing check. Each test_<module>.c exports one TestSuite;
 * test_main.c lists the suites and runs them. */
#ifndef CGRAD_TEST_H
#define CGRAD_TEST_H

#include <math.h>
#include <stddef.h>

typedef void (*TestFn)(void);

typedef struct {
    const char *name;
    TestFn fn;
} TestCase;

typedef struct {
    const char *name;
    const TestCase *tests;
    size_t count;
} TestSuite;

/* Number of elements of a true array (not a pointer!). */
#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

/* Record a failure of the currently running test (defined in test_main.c). */
void test_fail(const char *file, int line, const char *fmt, ...);

/* do { } while (0) turns the macro into a single statement, so it is safe in an
 * unbraced if/else and still needs a trailing semicolon. #cond turns the
 * expression into a string literal for the failure message. */
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond))                                                    \
            test_fail(__FILE__, __LINE__, "CHECK(%s)", #cond);          \
    } while (0)

/* Each argument is evaluated exactly once. Written as !(diff <= tol) so a NaN
 * on either side fails (every comparison with NaN is false). */
#define CHECK_NEAR(a, b, tol)                                           \
    do {                                                                \
        double check_a_ = (a), check_b_ = (b), check_tol_ = (tol);      \
        if (!(fabs(check_a_ - check_b_) <= check_tol_))                 \
            test_fail(__FILE__, __LINE__,                               \
                      "CHECK_NEAR(%s, %s): %.17g vs %.17g (tol %g)",    \
                      #a, #b, check_a_, check_b_, check_tol_);          \
    } while (0)

/* One suite per module, defined in tests/test_<module>.c. */
extern const TestSuite util_suite;
extern const TestSuite rng_suite;
extern const TestSuite tensor_suite;

#endif /* CGRAD_TEST_H */
