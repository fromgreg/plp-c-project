/* Gradient checks: the backward function of every op against finite
 * differences (gradcheck.h), plus tests of the checker itself.
 *
 * Run only these, with the measured errors:  ./build/debug/test_runner gradcheck */
#include <stdio.h>
#include <string.h>

#include "autograd.h"
#include "gradcheck.h"
#include "ops.h"
#include "rng.h"
#include "tensor.h"
#include "test.h"

#define SEED 2026

/* ---- helpers ---------------------------------------------------------------- */

/* Param / plain leaf with values from U[-1, 1). Caller frees. */
static Tensor *random_param(Rng *rng, size_t rows, size_t cols)
{
    Tensor *t = tensor_create_param(rows, cols);
    tensor_fill_uniform(t, rng, -1.0, 1.0);
    return t;
}

static Tensor *random_leaf(Rng *rng, size_t rows, size_t cols)
{
    Tensor *t = tensor_create(rows, cols);
    tensor_fill_uniform(t, rng, -1.0, 1.0);
    return t;
}

/* Reduce a matrix-shaped result to a scalar: sum(out * w), with a different
 * fixed weight w for every element.
 *
 * Without the weights, every element of out would receive the same gradient
 * (1.0), and a backward function that mixes up two indices could still pass.
 * With them, each element's gradient is different. */
static Tensor *weighted_sum(Tensor *out, Tensor *w)
{
    return ops_sum(ops_mul(out, w));
}

/* Run the check, print the measured error and record a failure if it is too
 * large. */
static void expect_gradients_match(const char *name, GraphFn f, Tensor **inputs,
                                   size_t n_inputs, void *ctx)
{
    double error = gradcheck_max_error(f, inputs, n_inputs, ctx);
    printf("    %-22s max relative error %.2e\n", name, error);
    CHECK(error < GRADCHECK_TOLERANCE);
}

/* ---- the functions under test (GraphFn) ---------------------------------------- */
/* For ops with a matrix result, ctx is the weight tensor of weighted_sum.
 * void * converts to Tensor * without a cast. */

static Tensor *f_add(Tensor **in, void *ctx)
{
    return weighted_sum(ops_add(in[0], in[1]), ctx);
}

static Tensor *f_mul(Tensor **in, void *ctx)
{
    return weighted_sum(ops_mul(in[0], in[1]), ctx);
}

static Tensor *f_matmul(Tensor **in, void *ctx)
{
    return weighted_sum(ops_matmul(in[0], in[1]), ctx);
}

static Tensor *f_matmul_square(Tensor **in, void *ctx)
{
    return weighted_sum(ops_matmul(in[0], in[0]), ctx); /* x * x: both parents */
}

static Tensor *f_add_bias(Tensor **in, void *ctx)
{
    return weighted_sum(ops_add_bias(in[0], in[1]), ctx);
}

static Tensor *f_relu(Tensor **in, void *ctx)
{
    return weighted_sum(ops_relu(in[0]), ctx);
}

static Tensor *f_tanh(Tensor **in, void *ctx)
{
    return weighted_sum(ops_tanh(in[0]), ctx);
}

static Tensor *f_sigmoid(Tensor **in, void *ctx)
{
    return weighted_sum(ops_sigmoid(in[0]), ctx);
}

/* The results of the next three are already 1x1, so ctx is not needed.
 * (void)ctx; tells the compiler that this is on purpose (-Wunused-parameter). */

static Tensor *f_sum(Tensor **in, void *ctx)
{
    (void)ctx;
    return ops_sum(in[0]);
}

static Tensor *f_mean(Tensor **in, void *ctx)
{
    (void)ctx;
    return ops_mean(in[0]);
}

static Tensor *f_mse(Tensor **in, void *ctx)
{
    (void)ctx;
    return ops_mse(in[0], in[1]);
}

/* Here ctx is the label array. */
static Tensor *f_softmax_cross_entropy(Tensor **in, void *ctx)
{
    return ops_softmax_cross_entropy(in[0], ctx);
}

/* ---- one check per op ----------------------------------------------------------- */
/* The shapes are never square, so swapped rows and columns cannot pass. */

/* Check an op with two inputs of the given shapes and a result of shape
 * out_rows x out_cols. */
static void check_binary(const char *name, GraphFn f,
                         size_t a_rows, size_t a_cols,
                         size_t b_rows, size_t b_cols,
                         size_t out_rows, size_t out_cols)
{
    Rng rng;
    rng_seed(&rng, SEED);
    Tensor *a = random_param(&rng, a_rows, a_cols);
    Tensor *b = random_param(&rng, b_rows, b_cols);
    Tensor *w = random_leaf(&rng, out_rows, out_cols);
    Tensor *inputs[] = {a, b};

    expect_gradients_match(name, f, inputs, ARRAY_LEN(inputs), w);

    tensor_free(a);
    tensor_free(b);
    tensor_free(w);
}

/* Check an op with one 3x4 input and a result of the same shape or 1x1. */
static void check_unary(const char *name, GraphFn f)
{
    Rng rng;
    rng_seed(&rng, SEED);
    Tensor *x = random_param(&rng, 3, 4);
    Tensor *w = random_leaf(&rng, 3, 4);
    Tensor *inputs[] = {x};

    expect_gradients_match(name, f, inputs, ARRAY_LEN(inputs), w);

    tensor_free(x);
    tensor_free(w);
}

static void check_add(void)
{
    check_binary("add", f_add, 3, 4, 3, 4, 3, 4);
}

static void check_mul(void)
{
    check_binary("mul", f_mul, 3, 4, 3, 4, 3, 4);
}

static void check_matmul(void)
{
    check_binary("matmul", f_matmul, 3, 4, 4, 2, 3, 2);
}

static void check_matmul_same_tensor(void)
{
    /* x * x needs a square x. Both parents are the same tensor, so its grad
     * receives two different shares. */
    Rng rng;
    rng_seed(&rng, SEED);
    Tensor *x = random_param(&rng, 3, 3);
    Tensor *w = random_leaf(&rng, 3, 3);
    Tensor *inputs[] = {x};

    expect_gradients_match("matmul(x, x)", f_matmul_square, inputs,
                           ARRAY_LEN(inputs), w);

    tensor_free(x);
    tensor_free(w);
}

static void check_add_bias(void)
{
    check_binary("add_bias", f_add_bias, 3, 4, 1, 4, 3, 4);
}

static void check_sum(void)
{
    check_unary("sum", f_sum);
}

static void check_mean(void)
{
    check_unary("mean", f_mean);
}

static void check_relu(void)
{
    /* ReLU has a kink at 0: if x - h and x + h lie on different sides, the
     * central difference measures neither slope. Keep every input at
     * |x| >= 0.1, far away from h = 1e-6. */
    Rng rng;
    rng_seed(&rng, SEED);
    Tensor *x = tensor_create_param(3, 4);
    tensor_fill_uniform(x, &rng, 0.1, 1.0);
    for (size_t i = 0; i < tensor_size(x); i++)
        if (rng_uniform(&rng) < 0.5)
            x->data[i] = -x->data[i];
    Tensor *w = random_leaf(&rng, 3, 4);
    Tensor *inputs[] = {x};

    expect_gradients_match("relu", f_relu, inputs, ARRAY_LEN(inputs), w);

    tensor_free(x);
    tensor_free(w);
}

static void check_tanh(void)
{
    check_unary("tanh", f_tanh);
}

static void check_sigmoid(void)
{
    check_unary("sigmoid", f_sigmoid);
}

static void check_mse(void)
{
    /* Both arguments are params here, so the gradient of the target is
     * checked as well. w is not used by f_mse. */
    check_binary("mse", f_mse, 3, 4, 3, 4, 1, 1);
}

static void check_softmax_cross_entropy(void)
{
    Rng rng;
    rng_seed(&rng, SEED);
    Tensor *logits = random_param(&rng, 4, 5);
    size_t labels[] = {0, 3, 1, 4}; /* one class index per row */
    Tensor *inputs[] = {logits};

    expect_gradients_match("softmax_cross_entropy", f_softmax_cross_entropy,
                           inputs, ARRAY_LEN(inputs), labels);

    tensor_free(logits);
}

/* ---- composed check: the loss of a 2-layer MLP ---------------------------------- */

/* Everything f_mlp needs besides the parameters. A GraphFn has a single
 * void *ctx, so several values travel in a struct. */
typedef struct {
    Tensor *x;            /* 4 x 3 input batch, a plain leaf */
    const size_t *labels; /* 4 class indices */
} MlpCase;

/* in = {W1, b1, W2, b2}:
 *     hidden = tanh(x * W1 + b1)
 *     logits = hidden * W2 + b2
 *     loss   = softmax_cross_entropy(logits, labels) */
static Tensor *f_mlp(Tensor **in, void *ctx)
{
    const MlpCase *c = ctx;
    Tensor *hidden = ops_tanh(ops_add_bias(ops_matmul(c->x, in[0]), in[1]));
    Tensor *logits = ops_add_bias(ops_matmul(hidden, in[2]), in[3]);
    return ops_softmax_cross_entropy(logits, c->labels);
}

static void check_mlp_loss(void)
{
    Rng rng;
    rng_seed(&rng, SEED);
    Tensor *x = random_leaf(&rng, 4, 3);
    Tensor *w1 = random_param(&rng, 3, 5);
    Tensor *b1 = random_param(&rng, 1, 5);
    Tensor *w2 = random_param(&rng, 5, 3);
    Tensor *b2 = random_param(&rng, 1, 3);
    size_t labels[] = {2, 0, 1, 2};
    MlpCase mlp = {.x = x, .labels = labels};
    Tensor *inputs[] = {w1, b1, w2, b2};

    expect_gradients_match("2-layer MLP loss", f_mlp, inputs,
                           ARRAY_LEN(inputs), &mlp);
    CHECK(x->grad == NULL); /* the input batch stays a plain leaf */

    tensor_free(x);
    tensor_free(w1);
    tensor_free(b1);
    tensor_free(w2);
    tensor_free(b2);
}

/* ---- tests of the checker itself ------------------------------------------------- */

/* An op with a deliberately wrong backward function, built outside ops.c in
 * the same way the real ops are built. Forward: out = 2 * x. */
static void broken_double_backward(Tensor *self)
{
    Tensor *x = self->parents[0];
    size_t n = tensor_size(self);
    for (size_t i = 0; i < n; i++)
        x->grad[i] += self->grad[i]; /* BUG on purpose: the factor 2 is missing */
}

static Tensor *broken_double(Tensor *x)
{
    Tensor *out = tensor_create_intermediate(x->rows, x->cols);
    out->parents[0] = x;
    out->n_parents = 1;
    out->backward = broken_double_backward;

    size_t n = tensor_size(out);
    for (size_t i = 0; i < n; i++)
        out->data[i] = 2.0 * x->data[i];
    return out;
}

static Tensor *f_broken_double(Tensor **in, void *ctx)
{
    return weighted_sum(broken_double(in[0]), ctx);
}

static void detects_wrong_backward(void)
{
    /* A checker that always says "fine" would be useless, so show that it
     * can fail. Analytic gradient: w. True gradient: 2w. Relative error:
     * |w - 2w| / (|w| + |2w|) = 1/3 for every element. */
    Rng rng;
    rng_seed(&rng, SEED);
    Tensor *x = random_param(&rng, 2, 3);
    Tensor *w = random_leaf(&rng, 2, 3);
    Tensor *inputs[] = {x};

    double error = gradcheck_max_error(f_broken_double, inputs,
                                       ARRAY_LEN(inputs), w);
    CHECK(error > GRADCHECK_TOLERANCE);
    CHECK_NEAR(error, 1.0 / 3.0, 1e-6);

    tensor_free(x);
    tensor_free(w);
}

static void restores_inputs(void)
{
    /* The checker perturbs the inputs; afterwards they must hold exactly the
     * old values again, bit for bit. */
    Rng rng;
    rng_seed(&rng, SEED);
    Tensor *a = random_param(&rng, 3, 4);
    Tensor *b = random_param(&rng, 3, 4);
    Tensor *w = random_leaf(&rng, 3, 4);
    Tensor *a_before = tensor_copy(a);
    Tensor *b_before = tensor_copy(b);
    Tensor *inputs[] = {a, b};

    gradcheck_max_error(f_mul, inputs, ARRAY_LEN(inputs), w);

    size_t bytes = tensor_size(a) * sizeof *a->data;
    CHECK(memcmp(a->data, a_before->data, bytes) == 0);
    CHECK(memcmp(b->data, b_before->data, bytes) == 0);

    tensor_free(a);
    tensor_free(b);
    tensor_free(w);
    tensor_free(a_before);
    tensor_free(b_before);
}

static const TestCase tests[] = {
    {"detects_wrong_backward", detects_wrong_backward},
    {"restores_inputs", restores_inputs},
    {"add", check_add},
    {"mul", check_mul},
    {"matmul", check_matmul},
    {"matmul_same_tensor", check_matmul_same_tensor},
    {"add_bias", check_add_bias},
    {"sum", check_sum},
    {"mean", check_mean},
    {"relu", check_relu},
    {"tanh", check_tanh},
    {"sigmoid", check_sigmoid},
    {"mse", check_mse},
    {"softmax_cross_entropy", check_softmax_cross_entropy},
    {"mlp_loss", check_mlp_loss},
};

const TestSuite gradcheck_suite = {"gradcheck", tests, ARRAY_LEN(tests)};
