#include "ops.h"

#include "util.h"

/* Backward functions
 *
 * Each one gets the result node of its op. self->grad already holds
 * d(root)/d(self) when it is called. By the chain rule, a parent's share is
 *     d(root)/d(parent) = d(root)/d(self) * d(self)/d(parent).
 *
 * The share is ADDED (+=) to the parent's grad, never assigned: a tensor that
 * feeds several ops receives one share from each of them, and the gradient is
 * their sum. This also covers a == b (x + x, x * x): both shares then land in
 * the same buffer.
 *
 * A parent without a grad buffer (a plain leaf) is skipped. */

static void add_backward(Tensor *self)
{
    size_t n = tensor_size(self);
    for (int p = 0; p < self->n_parents; p++) {
        Tensor *parent = self->parents[p];
        if (parent->grad == NULL)
            continue;
        /* d(a + b)/da = d(a + b)/db = 1: the gradient passes through unchanged. */
        for (size_t i = 0; i < n; i++)
            parent->grad[i] += self->grad[i];
    }
}

static void mul_backward(Tensor *self)
{
    Tensor *a = self->parents[0];
    Tensor *b = self->parents[1];
    size_t n = tensor_size(self);

    /* d(a * b)/da = b and d(a * b)/db = a, element by element. */
    if (a->grad != NULL)
        for (size_t i = 0; i < n; i++)
            a->grad[i] += b->data[i] * self->grad[i];
    if (b->grad != NULL)
        for (size_t i = 0; i < n; i++)
            b->grad[i] += a->data[i] * self->grad[i];
}

/* Helpers */

/* New result node of a binary op: links it to its inputs and stores the
 * function that will propagate its gradient to them. */
static Tensor *op_result(size_t rows, size_t cols, BackwardFn backward,
                         Tensor *a, Tensor *b)
{
    Tensor *out = tensor_create_intermediate(rows, cols);
    out->parents[0] = a;
    out->parents[1] = b;
    out->n_parents = 2;
    out->backward = backward;
    return out;
}

static void require_same_shape(const char *op, const Tensor *a, const Tensor *b)
{
    if (a->rows != b->rows || a->cols != b->cols)
        fatal("%s: shape mismatch (%zux%zu) vs (%zux%zu)",
              op, a->rows, a->cols, b->rows, b->cols);
}

/* Ops */

Tensor *ops_add(Tensor *a, Tensor *b)
{
    require_same_shape("ops_add", a, b);
    Tensor *out = op_result(a->rows, a->cols, add_backward, a, b);

    size_t n = tensor_size(out);
    for (size_t i = 0; i < n; i++)
        out->data[i] = a->data[i] + b->data[i];
    return out;
}

Tensor *ops_mul(Tensor *a, Tensor *b)
{
    require_same_shape("ops_mul", a, b);
    Tensor *out = op_result(a->rows, a->cols, mul_backward, a, b);

    size_t n = tensor_size(out);
    for (size_t i = 0; i < n; i++)
        out->data[i] = a->data[i] * b->data[i];
    return out;
}
