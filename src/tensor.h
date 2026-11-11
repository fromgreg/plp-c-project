/* tensor: 2D row-major matrix of doubles.
 *
 * Element (r, c) of a rows x cols tensor lives at data[r * cols + c].
 * Scalars are 1x1, vectors are 1xN. Every tensor returned by a function in
 * this module is owned by the caller and must be released with tensor_free(). */
#ifndef CGRAD_TENSOR_H
#define CGRAD_TENSOR_H

#include <assert.h>
#include <stddef.h>
#include <stdio.h>

#include "rng.h"

typedef struct {
    size_t rows;
    size_t cols;
    double *data; /* rows * cols elements, owned by the tensor */
} Tensor;

/* ---- lifetime ------------------------------------------------------------ */

/* New rows x cols tensor, all elements 0.0. Caller frees.
 * Aborts if a dimension is 0 or rows * cols overflows. */
Tensor *tensor_create(size_t rows, size_t cols);

/* Free t and its data. NULL is a no-op, like free(). */
void tensor_free(Tensor *t);

/* New tensor with a copy of rows * cols values (row-major). Caller frees the
 * tensor; values stays owned by the caller and is not kept. */
Tensor *tensor_from_array(size_t rows, size_t cols, const double *values);

/* Deep copy of src (own data buffer). Caller frees. */
Tensor *tensor_copy(const Tensor *src);

/* ---- in-place fills -------------------------------------------------------- */

void tensor_zeros(Tensor *t);
void tensor_fill(Tensor *t, double value);

/* Fill with samples from U[lo, hi) / N(mean, std^2), drawn from rng. */
void tensor_fill_uniform(Tensor *t, Rng *rng, double lo, double hi);
void tensor_fill_normal(Tensor *t, Rng *rng, double mean, double std);

/* ---- element access ------------------------------------------------------ */
/* static inline: defined in the header so calls can be inlined in hot loops.
 * The bounds asserts are compiled out in release builds (-DNDEBUG). */

static inline size_t tensor_size(const Tensor *t)
{
    return t->rows * t->cols;
}

static inline double tensor_get(const Tensor *t, size_t r, size_t c)
{
    assert(r < t->rows && c < t->cols);
    return t->data[r * t->cols + c];
}

static inline void tensor_set(Tensor *t, size_t r, size_t c, double value)
{
    assert(r < t->rows && c < t->cols);
    t->data[r * t->cols + c] = value;
}

/* ---- debugging and math ---------------------------------------------------- */

/* Print shape and values of t to out, one row per line. name may be NULL. */
void tensor_print(FILE *out, const Tensor *t, const char *name);

/* Matrix product a (n x k) * b (k x m) -> new n x m tensor. Caller frees.
 * Forward computation only (no gradients). Aborts if a->cols != b->rows. */
Tensor *tensor_matmul(const Tensor *a, const Tensor *b);

#endif /* CGRAD_TENSOR_H */
