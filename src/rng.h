/* rng: small, seedable xorshift64* pseudo-random number generator.
 *
 * The state is an explicit struct (no global), so every run with the same seed
 * produces the same numbers on every machine, and tests can own their stream. */
#ifndef CGRAD_RNG_H
#define CGRAD_RNG_H

#include <stdint.h>

typedef struct {
    uint64_t state; /* never 0: 0 is a fixed point of xorshift */
} Rng;

/* Initialise rng from seed. Any seed is valid, including 0. */
void rng_seed(Rng *rng, uint64_t seed);

/* Next raw 64-bit output. */
uint64_t rng_next_u64(Rng *rng);

/* Uniform double in [0, 1). */
double rng_uniform(Rng *rng);

/* Normally distributed double with the given mean and standard deviation. */
double rng_normal(Rng *rng, double mean, double std);

#endif /* CGRAD_RNG_H */
