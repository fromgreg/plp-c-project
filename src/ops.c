#include "ops.h"

#include <math.h>
#include <string.h>

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

static void matmul_backward(Tensor *self)
{
    Tensor *a = self->parents[0]; /* n x k */
    Tensor *b = self->parents[1]; /* k x m */
    size_t n = a->rows, k = a->cols, m = b->cols;
    const double *g = self->grad; /* n x m */

    /* out[i][j] = sum_p a[i][p] * b[p][j].
     *
     * a[i][p] appears in every out[i][j] of row i, with factor b[p][j]:
     *     d(root)/d(a[i][p]) = sum_j g[i][j] * b[p][j]      (g times b transposed) */
    if (a->grad != NULL) {
        for (size_t i = 0; i < n; i++) {
            for (size_t p = 0; p < k; p++) {
                double sum = 0.0;
                for (size_t j = 0; j < m; j++)
                    sum += g[i * m + j] * b->data[p * m + j];
                a->grad[i * k + p] += sum;
            }
        }
    }

    /* b[p][j] appears in every out[i][j] of column j, with factor a[i][p]:
     *     d(root)/d(b[p][j]) = sum_i a[i][p] * g[i][j]      (a transposed times g)
     * The sum over i is the outer loop here, so the inner loop walks along one
     * row of b->grad and one row of g (contiguous memory). */
    if (b->grad != NULL) {
        for (size_t i = 0; i < n; i++) {
            for (size_t p = 0; p < k; p++) {
                double a_ip = a->data[i * k + p];
                for (size_t j = 0; j < m; j++)
                    b->grad[p * m + j] += a_ip * g[i * m + j];
            }
        }
    }
}

static void add_bias_backward(Tensor *self)
{
    Tensor *x = self->parents[0];    /* n x m */
    Tensor *bias = self->parents[1]; /* 1 x m */
    size_t n = self->rows, m = self->cols;

    if (x->grad != NULL)
        for (size_t i = 0; i < n * m; i++)
            x->grad[i] += self->grad[i];

    /* bias[c] was added to every row, so it collects the gradient of its
     * whole column. */
    if (bias->grad != NULL)
        for (size_t r = 0; r < n; r++)
            for (size_t c = 0; c < m; c++)
                bias->grad[c] += self->grad[r * m + c];
}

static void sum_backward(Tensor *self)
{
    Tensor *x = self->parents[0];
    if (x->grad == NULL)
        return;

    /* Every element contributes with factor 1 to the 1x1 result. */
    size_t n = tensor_size(x);
    for (size_t i = 0; i < n; i++)
        x->grad[i] += self->grad[0];
}

static void mean_backward(Tensor *self)
{
    Tensor *x = self->parents[0];
    if (x->grad == NULL)
        return;

    /* Like sum, but every element contributes with factor 1/n. */
    size_t n = tensor_size(x);
    double share = self->grad[0] / (double)n;
    for (size_t i = 0; i < n; i++)
        x->grad[i] += share;
}

/* The activations compute their derivative from the stored OUTPUT y
 * (self->data), which is cheaper than going back to the input x. */

static void relu_backward(Tensor *self)
{
    Tensor *x = self->parents[0];
    if (x->grad == NULL)
        return;

    /* Derivative 1 where x > 0 (exactly where the output is > 0), else 0. */
    size_t n = tensor_size(self);
    for (size_t i = 0; i < n; i++)
        if (self->data[i] > 0.0)
            x->grad[i] += self->grad[i];
}

static void tanh_backward(Tensor *self)
{
    Tensor *x = self->parents[0];
    if (x->grad == NULL)
        return;

    /* d tanh(x)/dx = 1 - tanh(x)^2 = 1 - y^2. */
    size_t n = tensor_size(self);
    for (size_t i = 0; i < n; i++) {
        double y = self->data[i];
        x->grad[i] += (1.0 - y * y) * self->grad[i];
    }
}

static void sigmoid_backward(Tensor *self)
{
    Tensor *x = self->parents[0];
    if (x->grad == NULL)
        return;

    /* d sigmoid(x)/dx = sigmoid(x) * (1 - sigmoid(x)) = y * (1 - y). */
    size_t n = tensor_size(self);
    for (size_t i = 0; i < n; i++) {
        double y = self->data[i];
        x->grad[i] += y * (1.0 - y) * self->grad[i];
    }
}

static void mse_backward(Tensor *self)
{
    Tensor *pred = self->parents[0];
    Tensor *target = self->parents[1];
    size_t n = tensor_size(pred);

    /* loss = (1/n) * sum (p - t)^2, so d(loss)/dp = 2 (p - t) / n and
     * d(loss)/dt is the same with the opposite sign. */
    double scale = 2.0 * self->grad[0] / (double)n;
    for (size_t i = 0; i < n; i++) {
        double share = scale * (pred->data[i] - target->data[i]);
        if (pred->grad != NULL)
            pred->grad[i] += share;
        if (target->grad != NULL)
            target->grad[i] -= share;
    }
}

static void softmax_cross_entropy_backward(Tensor *self)
{
    Tensor *logits = self->parents[0]; /* n x c */
    if (logits->grad == NULL)
        return;

    /* ctx is a void *: only this op knows that the forward pass stored a copy
     * of the labels there. void * converts to any object pointer type without
     * a cast in C. */
    const size_t *labels = self->ctx;
    size_t n = logits->rows, c = logits->cols;
    double scale = self->grad[0] / (double)n;

    /* d(loss)/d(z[r][j]) = (softmax[r][j] - y[r][j]) / n, where y is 1 in the
     * column of the correct class and 0 elsewhere. */
    for (size_t r = 0; r < n; r++) {
        double lse = tensor_row_logsumexp(logits, r);
        for (size_t j = 0; j < c; j++) {
            size_t i = r * c + j;
            double p = exp(logits->data[i] - lse);
            double y = (j == labels[r]) ? 1.0 : 0.0;
            logits->grad[i] += scale * (p - y);
        }
    }
}

/* Helpers */

/* New result node of an op: links it to its inputs and stores the function
 * that will propagate its gradient to them. b is NULL for an op with a single
 * input. */
static Tensor *op_result(size_t rows, size_t cols, BackwardFn backward,
                         Tensor *a, Tensor *b)
{
    Tensor *out = tensor_create_intermediate(rows, cols);
    out->parents[0] = a;
    out->parents[1] = b;
    out->n_parents = (b != NULL) ? 2 : 1;
    out->backward = backward;
    return out;
}

static void require_same_shape(const char *op, const Tensor *a, const Tensor *b)
{
    if (a->rows != b->rows || a->cols != b->cols)
        fatal("%s: shape mismatch (%zux%zu) vs (%zux%zu)",
              op, a->rows, a->cols, b->rows, b->cols);
}

/* static inline: a small function the compiler can paste into the loop that
 * calls it. Unlike a macro it is type-checked and evaluates x only once.
 *
 * The textbook form 1 / (1 + exp(-x)) overflows in exp for very negative x.
 * The two branches are the same function; each one only ever calls exp with
 * an argument <= 0. */
static inline double sigmoid(double x)
{
    if (x >= 0.0)
        return 1.0 / (1.0 + exp(-x));
    double e = exp(x);
    return e / (1.0 + e);
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

Tensor *ops_matmul(Tensor *a, Tensor *b)
{
    if (a->cols != b->rows)
        fatal("ops_matmul: shape mismatch (%zux%zu) * (%zux%zu)",
              a->rows, a->cols, b->rows, b->cols);

    size_t n = a->rows, k = a->cols, m = b->cols;
    Tensor *out = op_result(n, m, matmul_backward, a, b);

    /* Same naive i-j-k loop as tensor_matmul, which stays as the plain
     * reference without a graph. */
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

Tensor *ops_add_bias(Tensor *x, Tensor *bias)
{
    if (bias->rows != 1 || bias->cols != x->cols)
        fatal("ops_add_bias: bias must be 1x%zu for a %zux%zu input, got %zux%zu",
              x->cols, x->rows, x->cols, bias->rows, bias->cols);

    size_t n = x->rows, m = x->cols;
    Tensor *out = op_result(n, m, add_bias_backward, x, bias);

    for (size_t r = 0; r < n; r++)
        for (size_t c = 0; c < m; c++)
            out->data[r * m + c] = x->data[r * m + c] + bias->data[c];
    return out;
}

Tensor *ops_sum(Tensor *x)
{
    Tensor *out = op_result(1, 1, sum_backward, x, NULL);

    size_t n = tensor_size(x);
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
        sum += x->data[i];
    out->data[0] = sum;
    return out;
}

Tensor *ops_mean(Tensor *x)
{
    Tensor *out = op_result(1, 1, mean_backward, x, NULL);

    size_t n = tensor_size(x);
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
        sum += x->data[i];
    /* n is a size_t; the cast makes the conversion to double explicit
     * (-Wconversion warns about the implicit one). */
    out->data[0] = sum / (double)n;
    return out;
}

Tensor *ops_relu(Tensor *x)
{
    Tensor *out = op_result(x->rows, x->cols, relu_backward, x, NULL);

    size_t n = tensor_size(out);
    for (size_t i = 0; i < n; i++)
        out->data[i] = x->data[i] > 0.0 ? x->data[i] : 0.0;
    return out;
}

Tensor *ops_tanh(Tensor *x)
{
    Tensor *out = op_result(x->rows, x->cols, tanh_backward, x, NULL);

    size_t n = tensor_size(out);
    for (size_t i = 0; i < n; i++)
        out->data[i] = tanh(x->data[i]);
    return out;
}

Tensor *ops_sigmoid(Tensor *x)
{
    Tensor *out = op_result(x->rows, x->cols, sigmoid_backward, x, NULL);

    size_t n = tensor_size(out);
    for (size_t i = 0; i < n; i++)
        out->data[i] = sigmoid(x->data[i]);
    return out;
}

Tensor *ops_mse(Tensor *pred, Tensor *target)
{
    require_same_shape("ops_mse", pred, target);
    Tensor *out = op_result(1, 1, mse_backward, pred, target);

    size_t n = tensor_size(pred);
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = pred->data[i] - target->data[i];
        sum += diff * diff;
    }
    out->data[0] = sum / (double)n;
    return out;
}

Tensor *ops_softmax_cross_entropy(Tensor *logits, const size_t *labels)
{
    size_t n = logits->rows, c = logits->cols;

    /* Checked here, once, so that the backward pass can trust the labels. */
    for (size_t r = 0; r < n; r++)
        if (labels[r] >= c)
            fatal("ops_softmax_cross_entropy: label %zu of row %zu is out of "
                  "range for %zu classes", labels[r], r, c);

    Tensor *out = op_result(1, 1, softmax_cross_entropy_backward, logits, NULL);

    /* The backward pass runs after this function has returned, and the caller
     * may have refilled its label array by then (next batch). So the node
     * keeps its own copy in ctx; tensor_free releases it with the node. */
    size_t *saved = xmalloc(n * sizeof *saved);
    memcpy(saved, labels, n * sizeof *saved);
    out->ctx = saved;

    /* Loss of one row = -log(softmax[label]) = logsumexp(z) - z[label].
     * No probability is ever computed and then passed to log, so a
     * probability that underflows to 0 cannot produce log(0) = -inf. */
    double total = 0.0;
    for (size_t r = 0; r < n; r++)
        total += tensor_row_logsumexp(logits, r) - logits->data[r * c + labels[r]];
    out->data[0] = total / (double)n;
    return out;
}
