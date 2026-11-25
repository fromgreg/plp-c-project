/* Tests for ops: forward values, the graph links stored in the result, and
 * numerical stability for extreme inputs.
 *
 * Expected values are worked out by hand in the comment of each test. The
 * gradients of every op are checked against finite differences in
 * test_gradcheck.c; a few are also checked here against hand-derived values. */
#include <math.h>

#include "autograd.h"
#include "ops.h"
#include "tensor.h"
#include "test.h"

#define TOL 1e-12

/* ---- add, mul ----------------------------------------------------------------- */

static void add_forward(void)
{
    double av[] = {1, 2, 3, 4, 5, 6};
    double bv[] = {10, 20, 30, 40, 50, 60};
    double expected[] = {11, 22, 33, 44, 55, 66};
    Tensor *a = tensor_from_array(2, 3, av);
    Tensor *b = tensor_from_array(2, 3, bv);
    Tensor *out = ops_add(a, b);

    CHECK(out->rows == 2 && out->cols == 3);
    for (size_t i = 0; i < ARRAY_LEN(expected); i++)
        CHECK(out->data[i] == expected[i]);

    graph_free(out); /* out is graph-owned */
    tensor_free(a);
    tensor_free(b);
}

static void mul_forward(void)
{
    double av[] = {1, 2, 3, 4, 5, 6};
    double bv[] = {2, 0.5, -1, 0, 3, 10};
    double expected[] = {2, 1, -3, 0, 15, 60};
    Tensor *a = tensor_from_array(2, 3, av);
    Tensor *b = tensor_from_array(2, 3, bv);
    Tensor *out = ops_mul(a, b);

    CHECK(out->rows == 2 && out->cols == 3);
    for (size_t i = 0; i < ARRAY_LEN(expected); i++)
        CHECK(out->data[i] == expected[i]);

    graph_free(out);
    tensor_free(a);
    tensor_free(b);
}

static void result_records_graph_links(void)
{
    Tensor *a = tensor_create_param(1, 2);
    Tensor *b = tensor_create(1, 2);
    Tensor *out = ops_mul(a, b);

    CHECK(out->kind == TENSOR_INTERMEDIATE);
    CHECK(out->n_parents == 2);
    CHECK(out->parents[0] == a && out->parents[1] == b);
    CHECK(out->backward != NULL);
    CHECK(out->grad != NULL);
    if (out->grad != NULL)
        CHECK(out->grad[0] == 0.0 && out->grad[1] == 0.0);

    graph_free(out);
    tensor_free(a);
    tensor_free(b);
}

static void unary_result_has_one_parent(void)
{
    Tensor *x = tensor_create_param(2, 3);
    Tensor *out = ops_relu(x);

    CHECK(out->kind == TENSOR_INTERMEDIATE);
    CHECK(out->n_parents == 1);
    CHECK(out->parents[0] == x && out->parents[1] == NULL);
    CHECK(out->backward != NULL);

    graph_free(out);
    tensor_free(x);
}

/* ---- matmul, add_bias --------------------------------------------------------- */

static void matmul_forward(void)
{
    /* [1 2 3]   [ 7  8]   [ 58  64]
     * [4 5 6] * [ 9 10] = [139 154]
     *           [11 12]                */
    double av[] = {1, 2, 3, 4, 5, 6};
    double bv[] = {7, 8, 9, 10, 11, 12};
    double expected[] = {58, 64, 139, 154};
    Tensor *a = tensor_from_array(2, 3, av);
    Tensor *b = tensor_from_array(3, 2, bv);
    Tensor *out = ops_matmul(a, b);

    CHECK(out->rows == 2 && out->cols == 2);
    for (size_t i = 0; i < ARRAY_LEN(expected); i++)
        CHECK(out->data[i] == expected[i]);

    graph_free(out);
    tensor_free(a);
    tensor_free(b);
}

static void matmul_matches_reference(void)
{
    /* Same result as the graph-free tensor_matmul on random matrices. */
    Rng rng;
    rng_seed(&rng, 7);
    Tensor *a = tensor_create(4, 6);
    Tensor *b = tensor_create(6, 3);
    tensor_fill_uniform(a, &rng, -1.0, 1.0);
    tensor_fill_uniform(b, &rng, -1.0, 1.0);

    Tensor *out = ops_matmul(a, b);
    Tensor *reference = tensor_matmul(a, b);
    CHECK(out->rows == reference->rows && out->cols == reference->cols);
    for (size_t i = 0; i < tensor_size(reference); i++)
        CHECK(out->data[i] == reference->data[i]);

    graph_free(out);
    tensor_free(reference); /* a plain leaf: caller frees */
    tensor_free(a);
    tensor_free(b);
}

static void add_bias_forward(void)
{
    /* The bias row is added to both rows of x. */
    double xv[] = {1, 2, 3, 4, 5, 6};
    double bv[] = {10, 20, 30};
    double expected[] = {11, 22, 33, 14, 25, 36};
    Tensor *x = tensor_from_array(2, 3, xv);
    Tensor *bias = tensor_from_array(1, 3, bv);
    Tensor *out = ops_add_bias(x, bias);

    CHECK(out->rows == 2 && out->cols == 3);
    for (size_t i = 0; i < ARRAY_LEN(expected); i++)
        CHECK(out->data[i] == expected[i]);

    graph_free(out);
    tensor_free(x);
    tensor_free(bias);
}

static void add_bias_gradient_is_column_sum(void)
{
    /* The root is seeded with ones. x gets them unchanged; each bias element
     * was used in 3 rows and collects 3 of them. */
    Tensor *x = tensor_create_param(3, 2);
    Tensor *bias = tensor_create_param(1, 2);
    Tensor *out = ops_add_bias(x, bias);
    graph_backward(out);

    for (size_t i = 0; i < tensor_size(x); i++)
        CHECK(x->grad[i] == 1.0);
    CHECK(bias->grad[0] == 3.0 && bias->grad[1] == 3.0);

    graph_free(out);
    tensor_free(x);
    tensor_free(bias);
}

/* ---- sum, mean ---------------------------------------------------------------- */

static void sum_and_mean_forward(void)
{
    double xv[] = {1, 2, 3, 4, 5, 6};
    Tensor *x = tensor_from_array(2, 3, xv);
    Tensor *sum = ops_sum(x);
    Tensor *mean = ops_mean(x);

    CHECK(sum->rows == 1 && sum->cols == 1);
    CHECK(sum->data[0] == 21.0);
    CHECK(mean->rows == 1 && mean->cols == 1);
    CHECK(mean->data[0] == 3.5);

    /* Two separate graphs that share only the leaf x. */
    graph_free(sum);
    graph_free(mean);
    tensor_free(x);
}

/* ---- activations -------------------------------------------------------------- */

static void relu_forward_and_gradient(void)
{
    /* Derivative: 1 for x > 0, otherwise 0 (also at exactly 0). */
    double xv[] = {-2, 0, 0.5, 3};
    double expected[] = {0, 0, 0.5, 3};
    double expected_grad[] = {0, 0, 1, 1};
    Tensor *x = tensor_create_param(1, 4);
    for (size_t i = 0; i < 4; i++)
        x->data[i] = xv[i];

    Tensor *out = ops_relu(x);
    graph_backward(out);
    for (size_t i = 0; i < 4; i++) {
        CHECK(out->data[i] == expected[i]);
        CHECK(x->grad[i] == expected_grad[i]);
    }

    graph_free(out);
    tensor_free(x);
}

static void tanh_forward(void)
{
    /* Reference values from an independent source (Python's math.tanh). */
    double xv[] = {0, 0.5, -1};
    double expected[] = {0, 0.46211715726000974, -0.7615941559557649};
    Tensor *x = tensor_from_array(1, 3, xv);
    Tensor *out = ops_tanh(x);

    for (size_t i = 0; i < ARRAY_LEN(expected); i++)
        CHECK_NEAR(out->data[i], expected[i], TOL);

    graph_free(out);
    tensor_free(x);
}

static void sigmoid_forward(void)
{
    /* sigmoid(0) = 1/2; sigmoid(log 3) = 1 / (1 + 1/3) = 3/4;
     * sigmoid(-log 3) = 1 - 3/4 = 1/4. The last one takes the x < 0 branch. */
    Tensor *x = tensor_create(1, 3);
    x->data[0] = 0.0;
    x->data[1] = log(3.0);
    x->data[2] = -log(3.0);
    Tensor *out = ops_sigmoid(x);

    CHECK_NEAR(out->data[0], 0.5, TOL);
    CHECK_NEAR(out->data[1], 0.75, TOL);
    CHECK_NEAR(out->data[2], 0.25, TOL);

    graph_free(out);
    tensor_free(x);
}

/* ---- losses ------------------------------------------------------------------- */

static void mse_forward_and_gradient(void)
{
    /* diff = pred - target = [0 2 0 -4]; loss = (0 + 4 + 0 + 16) / 4 = 5.
     * d(loss)/d(pred) = 2 * diff / 4 = [0 1 0 -2]. */
    double tv[] = {1, 0, 3, 8};
    double expected_grad[] = {0, 1, 0, -2};
    Tensor *pred = tensor_create_param(2, 2);
    for (size_t i = 0; i < 4; i++)
        pred->data[i] = (double)i + 1.0; /* 1 2 3 4 */
    Tensor *target = tensor_from_array(2, 2, tv);

    Tensor *loss = ops_mse(pred, target);
    CHECK(loss->rows == 1 && loss->cols == 1);
    CHECK_NEAR(loss->data[0], 5.0, TOL);

    graph_backward(loss);
    for (size_t i = 0; i < 4; i++)
        CHECK_NEAR(pred->grad[i], expected_grad[i], TOL);
    CHECK(target->grad == NULL); /* a plain leaf is skipped, not given a grad */

    graph_free(loss);
    tensor_free(pred);
    tensor_free(target);
}

static void softmax_cross_entropy_uniform(void)
{
    /* All logits equal: every class has probability 1/4, so the loss is
     * -log(1/4) = log 4 whatever the labels are. */
    Tensor *logits = tensor_create(2, 4); /* zeros */
    size_t labels[] = {1, 3};
    Tensor *loss = ops_softmax_cross_entropy(logits, labels);

    CHECK(loss->rows == 1 && loss->cols == 1);
    CHECK_NEAR(loss->data[0], log(4.0), TOL);

    graph_free(loss);
    tensor_free(logits);
}

static void softmax_cross_entropy_known_values(void)
{
    /* Both rows are z = [log 1, log 2, log 3], so exp(z) = [1 2 3] and
     * softmax = [1/6, 2/6, 3/6].
     *   row 0, label 2: -log(3/6) = log 2
     *   row 1, label 0: -log(1/6) = log 6
     * loss = (log 2 + log 6) / 2 = log(12) / 2.
     * Gradient = (softmax - onehot) / 2:
     *   row 0: [ 1/12, 2/12, -3/12]
     *   row 1: [-5/12, 2/12,  3/12] */
    double expected_grad[] = {1.0 / 12, 2.0 / 12, -3.0 / 12,
                              -5.0 / 12, 2.0 / 12, 3.0 / 12};
    Tensor *logits = tensor_create_param(2, 3);
    for (size_t r = 0; r < 2; r++)
        for (size_t c = 0; c < 3; c++)
            tensor_set(logits, r, c, log((double)c + 1.0));
    size_t labels[] = {2, 0};

    Tensor *loss = ops_softmax_cross_entropy(logits, labels);
    CHECK_NEAR(loss->data[0], log(12.0) / 2.0, TOL);

    graph_backward(loss);
    for (size_t i = 0; i < ARRAY_LEN(expected_grad); i++)
        CHECK_NEAR(logits->grad[i], expected_grad[i], TOL);

    graph_free(loss);
    tensor_free(logits);
}

static void softmax_cross_entropy_copies_labels(void)
{
    /* The op keeps its own copy of the labels in ctx: overwriting the
     * caller's array after the forward pass must not change the gradient. */
    Tensor *logits = tensor_create_param(1, 2); /* zeros: softmax = [1/2 1/2] */
    size_t labels[] = {0};
    Tensor *loss = ops_softmax_cross_entropy(logits, labels);
    CHECK(loss->ctx != NULL);
    CHECK(loss->ctx != (void *)labels);

    labels[0] = 1; /* e.g. the next batch */
    graph_backward(loss);
    CHECK_NEAR(logits->grad[0], -0.5, TOL); /* 1/2 - 1: label 0 was used */
    CHECK_NEAR(logits->grad[1], 0.5, TOL);

    graph_free(loss); /* also frees the copy; `make leaks` verifies it */
    tensor_free(logits);
}

/* ---- numerical stability ------------------------------------------------------ */
/* Inputs for which the textbook formulas produce inf or NaN: exp(1000) is inf,
 * and inf / inf is NaN. */

static void softmax_cross_entropy_extreme_logits(void)
{
    /* Both rows are [1000, -1000]: softmax = [1, 0] (exp(-2000) rounds to 0).
     *   row 0, label 0: loss 0     gradient ([1 0] - [1 0]) / 2 = [0    0  ]
     *   row 1, label 1: loss 2000  gradient ([1 0] - [0 1]) / 2 = [0.5 -0.5]
     * The second row asks for -log(0); the fused formula returns
     * logsumexp(z) - z[label] = 1000 - (-1000) instead of inf. */
    double zv[] = {1000, -1000, 1000, -1000};
    double expected_grad[] = {0, 0, 0.5, -0.5};
    Tensor *logits = tensor_create_param(2, 2);
    for (size_t i = 0; i < 4; i++)
        logits->data[i] = zv[i];
    size_t labels[] = {0, 1};

    Tensor *loss = ops_softmax_cross_entropy(logits, labels);
    CHECK(isfinite(loss->data[0]));
    CHECK_NEAR(loss->data[0], 1000.0, TOL);

    graph_backward(loss);
    for (size_t i = 0; i < 4; i++) {
        CHECK(isfinite(logits->grad[i]));
        CHECK_NEAR(logits->grad[i], expected_grad[i], TOL);
    }

    graph_free(loss);
    tensor_free(logits);
}

static void activations_extreme_inputs(void)
{
    /* Saturated activations: the outputs hit their limits and the gradients
     * become 0, but nothing turns into inf or NaN. */
    double xv[] = {1000, -1000, 0};
    double sigmoid_out[] = {1, 0, 0.5};
    double sigmoid_grad[] = {0, 0, 0.25}; /* y * (1 - y) */
    double tanh_out[] = {1, -1, 0};
    double tanh_grad[] = {0, 0, 1}; /* 1 - y^2 */
    Tensor *x = tensor_create_param(1, 3);
    for (size_t i = 0; i < 3; i++)
        x->data[i] = xv[i];

    Tensor *s = ops_sigmoid(x);
    graph_backward(s);
    for (size_t i = 0; i < 3; i++) {
        CHECK(isfinite(s->data[i]) && isfinite(x->grad[i]));
        CHECK_NEAR(s->data[i], sigmoid_out[i], TOL);
        CHECK_NEAR(x->grad[i], sigmoid_grad[i], TOL);
    }
    graph_free(s);

    tensor_zero_grad(x); /* gradients accumulate: clear before the next graph */
    Tensor *t = ops_tanh(x);
    graph_backward(t);
    for (size_t i = 0; i < 3; i++) {
        CHECK(isfinite(t->data[i]) && isfinite(x->grad[i]));
        CHECK_NEAR(t->data[i], tanh_out[i], TOL);
        CHECK_NEAR(x->grad[i], tanh_grad[i], TOL);
    }
    graph_free(t);

    tensor_free(x);
}

static const TestCase tests[] = {
    {"add_forward", add_forward},
    {"mul_forward", mul_forward},
    {"result_records_graph_links", result_records_graph_links},
    {"unary_result_has_one_parent", unary_result_has_one_parent},
    {"matmul_forward", matmul_forward},
    {"matmul_matches_reference", matmul_matches_reference},
    {"add_bias_forward", add_bias_forward},
    {"add_bias_gradient_is_column_sum", add_bias_gradient_is_column_sum},
    {"sum_and_mean_forward", sum_and_mean_forward},
    {"relu_forward_and_gradient", relu_forward_and_gradient},
    {"tanh_forward", tanh_forward},
    {"sigmoid_forward", sigmoid_forward},
    {"mse_forward_and_gradient", mse_forward_and_gradient},
    {"softmax_cross_entropy_uniform", softmax_cross_entropy_uniform},
    {"softmax_cross_entropy_known_values", softmax_cross_entropy_known_values},
    {"softmax_cross_entropy_copies_labels", softmax_cross_entropy_copies_labels},
    {"softmax_cross_entropy_extreme_logits", softmax_cross_entropy_extreme_logits},
    {"activations_extreme_inputs", activations_extreme_inputs},
};

const TestSuite ops_suite = {"ops", tests, ARRAY_LEN(tests)};
