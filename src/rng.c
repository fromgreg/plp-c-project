#include "rng.h"

#include <math.h>

/* M_PI is POSIX, not ISO C, so define 2*pi ourselves. */
static const double TWO_PI = 6.283185307179586476925286766559;

/* Arbitrary non-zero replacement for seed 0 (the golden-ratio constant). */
static const uint64_t ZERO_SEED_REPLACEMENT = 0x9E3779B97F4A7C15u;

void rng_seed(Rng *rng, uint64_t seed)
{
    rng->state = seed ? seed : ZERO_SEED_REPLACEMENT;
}

/* xorshift64* (Vigna, 2016): a xorshift step followed by a multiplication,
 * which scrambles the weak low bits of plain xorshift. */
uint64_t rng_next_u64(Rng *rng)
{
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 0x2545F4914F6CDD1Du;
}

double rng_uniform(Rng *rng)
{
    /* A double has a 53-bit mantissa: keep the top 53 bits and scale by 2^-53.
     * Every result k * 2^-53 is exact and the maximum is 1 - 2^-53 < 1. */
    return (double)(rng_next_u64(rng) >> 11) * 0x1.0p-53;
}

/* Box-Muller transform: two independent uniforms -> one standard normal.
 * The second normal it could produce is discarded to keep Rng stateless
 * beyond the xorshift state. */
double rng_normal(Rng *rng, double mean, double std)
{
    double u1 = 1.0 - rng_uniform(rng); /* in (0, 1], so log(u1) is finite */
    double u2 = rng_uniform(rng);
    double z = sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
    return mean + std * z;
}
