/* tensor: 2D row-major matrix of doubles, and a node of the computational graph.
 *
 * Element (r, c) of a rows x cols tensor lives at data[r * cols + c].
 * Scalars are 1x1, vectors are 1xN.
 *
 * Ownership depends on the kind of the tensor:
 *   TENSOR_LEAF          plain data (inputs, targets). Caller frees.
 *   TENSOR_PARAM         trainable, has a grad buffer. Caller frees.
 *   TENSOR_INTERMEDIATE  result of an op. Graph-owned: released by
 *                        graph_free() (autograd.h), never by hand. */
#ifndef CGRAD_TENSOR_H
#define CGRAD_TENSOR_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "rng.h"

/* Forward declaration: the struct contains pointers to its own type, so the
 * name Tensor must exist before the struct body. */
typedef struct Tensor Tensor;

/* Backward function of an op. self is the op's result: the function reads
 * self->grad and adds (+=) each parent's share to that parent's grad. */
typedef void (*BackwardFn)(Tensor *self);

typedef enum {
    TENSOR_LEAF,
    TENSOR_PARAM,
    TENSOR_INTERMEDIATE
} TensorKind;

struct Tensor {
    size_t rows;
    size_t cols;
    double *data; /* rows * cols elements, owned by the tensor */

    /* ---- autograd ---- */
    double *grad;        /* d(root)/d(data), same size as data, owned by the
                          * tensor. NULL means: no gradient is tracked. */
    TensorKind kind;     /* decides who frees the tensor (see above) */
    Tensor *parents[2];  /* inputs of the op that produced this node.
                          * Borrowed: the node does not own its parents. */
    int n_parents;       /* 0 for leaves and params */
    BackwardFn backward; /* NULL for leaves and params */
    void *ctx;           /* op-specific saved state: a single malloc block,
                          * owned by the tensor and freed with it. May be NULL. */
    uint32_t visit_mark; /* number of the last topological sort that reached
                          * this node (see autograd.c) */
};

/* ---- lifetime ------------------------------------------------------------ */

/* New rows x cols TENSOR_LEAF, all elements 0.0, no grad. Caller frees.
 * Aborts if a dimension is 0 or rows * cols overflows. */
Tensor *tensor_create(size_t rows, size_t cols);

/* Like tensor_create, but a TENSOR_PARAM with a zeroed grad buffer.
 * Caller frees. */
Tensor *tensor_create_param(size_t rows, size_t cols);

/* Like tensor_create, but a TENSOR_INTERMEDIATE with a zeroed grad buffer.
 * For ops only. Graph-owned: graph_free() releases it. */
Tensor *tensor_create_intermediate(size_t rows, size_t cols);

/* Free t together with its data, grad and ctx. NULL is a no-op, like free().
 * Do not call this on an intermediate: use graph_free() on the root. */
void tensor_free(Tensor *t);

/* New TENSOR_LEAF with a copy of rows * cols values (row-major). Caller frees
 * the tensor; values stays owned by the caller and is not kept. */
Tensor *tensor_from_array(size_t rows, size_t cols, const double *values);

/* Deep copy of the data of src into a new TENSOR_LEAF. Grad and graph links
 * are not copied. Caller frees. */
Tensor *tensor_copy(const Tensor *src);

/* ---- in-place fills -------------------------------------------------------- */

void tensor_zeros(Tensor *t);
void tensor_fill(Tensor *t, double value);

/* Fill with samples from U[lo, hi) / N(mean, std^2), drawn from rng. */
void tensor_fill_uniform(Tensor *t, Rng *rng, double lo, double hi);
void tensor_fill_normal(Tensor *t, Rng *rng, double mean, double std);

/* Set every element of t->grad to 0.0. No-op if t has no grad. Call this on
 * every parameter before each backward pass, because gradients accumulate. */
void tensor_zero_grad(Tensor *t);

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

/* Matrix product a (n x k) * b (k x m) -> new n x m TENSOR_LEAF. Caller frees.
 * Forward computation only (no gradients). Aborts if a->cols != b->rows. */
Tensor *tensor_matmul(const Tensor *a, const Tensor *b);

#endif /* CGRAD_TENSOR_H */
