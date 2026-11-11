#include "tensor.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

Tensor *tensor_create(size_t rows, size_t cols)
{
    if (rows == 0 || cols == 0)
        fatal("tensor_create: zero dimension (%zu x %zu)", rows, cols);
    /* rows * cols would wrap around silently for size_t; check before multiplying. */
    if (rows > SIZE_MAX / cols)
        fatal("tensor_create: %zu x %zu elements overflow size_t", rows, cols);

    Tensor *t = xmalloc(sizeof *t);
    t->rows = rows;
    t->cols = cols;
    /* calloc zero-fills; all-zero bits is 0.0 for IEEE 754 doubles. */
    t->data = xcalloc(rows * cols, sizeof *t->data);
    return t;
}

void tensor_free(Tensor *t)
{
    if (t == NULL)
        return;
    free(t->data);
    free(t);
}

Tensor *tensor_from_array(size_t rows, size_t cols, const double *values)
{
    Tensor *t = tensor_create(rows, cols);
    memcpy(t->data, values, tensor_size(t) * sizeof *t->data);
    return t;
}

Tensor *tensor_copy(const Tensor *src)
{
    return tensor_from_array(src->rows, src->cols, src->data);
}

void tensor_zeros(Tensor *t)
{
    tensor_fill(t, 0.0);
}

void tensor_fill(Tensor *t, double value)
{
    size_t n = tensor_size(t);
    for (size_t i = 0; i < n; i++)
        t->data[i] = value;
}

void tensor_fill_uniform(Tensor *t, Rng *rng, double lo, double hi)
{
    size_t n = tensor_size(t);
    for (size_t i = 0; i < n; i++)
        t->data[i] = lo + (hi - lo) * rng_uniform(rng);
}

void tensor_fill_normal(Tensor *t, Rng *rng, double mean, double std)
{
    size_t n = tensor_size(t);
    for (size_t i = 0; i < n; i++)
        t->data[i] = rng_normal(rng, mean, std);
}

void tensor_print(FILE *out, const Tensor *t, const char *name)
{
    fprintf(out, "%s [%zux%zu]\n", name ? name : "tensor", t->rows, t->cols);
    for (size_t r = 0; r < t->rows; r++) {
        for (size_t c = 0; c < t->cols; c++)
            fprintf(out, "%10.4g", tensor_get(t, r, c));
        fputc('\n', out);
    }
}

Tensor *tensor_matmul(const Tensor *a, const Tensor *b)
{
    if (a->cols != b->rows)
        fatal("tensor_matmul: shape mismatch (%zux%zu) * (%zux%zu)",
              a->rows, a->cols, b->rows, b->cols);

    size_t n = a->rows, k = a->cols, m = b->cols;
    Tensor *out = tensor_create(n, m);

    /* Naive i-j-k order: out[i][j] = sum_p a[i][p] * b[p][j].
     * Indexing data directly keeps the inner loop free of assert overhead;
     * the loop bounds already guarantee every index is in range. */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            double sum = 0.0;
            for (size_t p = 0; p < k; p++)
                sum += a->data[i * k + p] * b->data[p * m + j];
            out->data[i * m + j] = sum;
        }
    }
    return out;
}
