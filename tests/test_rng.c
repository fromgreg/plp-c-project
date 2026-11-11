/* Tests for rng: reproducibility, reference values, and distribution sanity.
 * Fixed seeds make the statistical checks deterministic, not flaky. */
#include <math.h>
#include <stdint.h>

#include "rng.h"
#include "test.h"

enum { N_SAMPLES = 100000 };

static void reference_values_seed_1(void)
{
    /* Computed independently (Python) from the published xorshift64* algorithm. */
    Rng rng;
    rng_seed(&rng, 1);
    CHECK(rng_next_u64(&rng) == UINT64_C(0x47E4CE4B896CDD1D));
    CHECK(rng_next_u64(&rng) == UINT64_C(0xABCFA6A8E079651D));
    CHECK(rng_next_u64(&rng) == UINT64_C(0xB9D10D8FEB731F57));
}

static void same_seed_same_sequence(void)
{
    Rng a, b;
    rng_seed(&a, 42);
    rng_seed(&b, 42);
    int equal = 1;
    for (int i = 0; i < 1000; i++)
        equal = equal && rng_next_u64(&a) == rng_next_u64(&b);
    CHECK(equal);
}

static void different_seeds_differ(void)
{
    Rng a, b;
    rng_seed(&a, 1);
    rng_seed(&b, 2);
    CHECK(rng_next_u64(&a) != rng_next_u64(&b));
}

static void seed_zero_not_stuck(void)
{
    Rng rng;
    rng_seed(&rng, 0);
    uint64_t first = rng_next_u64(&rng);
    CHECK(first != 0);
    CHECK(rng_next_u64(&rng) != first);
}

static void uniform_range_and_mean(void)
{
    Rng rng;
    rng_seed(&rng, 7);
    int in_range = 1;
    double sum = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double u = rng_uniform(&rng);
        in_range = in_range && u >= 0.0 && u < 1.0;
        sum += u;
    }
    CHECK(in_range);
    CHECK_NEAR(sum / (double)N_SAMPLES, 0.5, 0.01);
}

static void normal_mean_and_std(void)
{
    Rng rng;
    rng_seed(&rng, 7);
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double z = rng_normal(&rng, 2.0, 3.0);
        sum += z;
        sum_sq += z * z;
    }
    double mean = sum / (double)N_SAMPLES;
    double var = sum_sq / (double)N_SAMPLES - mean * mean;
    CHECK_NEAR(mean, 2.0, 0.05);
    CHECK_NEAR(sqrt(var), 3.0, 0.05);
}

static const TestCase tests[] = {
    {"reference_values_seed_1", reference_values_seed_1},
    {"same_seed_same_sequence", same_seed_same_sequence},
    {"different_seeds_differ", different_seeds_differ},
    {"seed_zero_not_stuck", seed_zero_not_stuck},
    {"uniform_range_and_mean", uniform_range_and_mean},
    {"normal_mean_and_std", normal_mean_and_std},
};

const TestSuite rng_suite = {"rng", tests, ARRAY_LEN(tests)};
