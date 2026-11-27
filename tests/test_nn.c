/* Tests for nn: construction, initialization, parameter list, forward pass,
 * gradients of a whole network and a short training run. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "autograd.h"
#include "gradcheck.h"
#include "nn.h"
#include "ops.h"
#include "optim.h"
#include "rng.h"
#include "tensor.h"
#include "test.h"

#define TOL 1e-12
#define SEED 2026

/* ---- helpers ---------------------------------------------------------------- */

static MLP *make_mlp(const size_t *sizes, size_t n_sizes, Activation act, uint64_t seed)
{
    Rng rng;
    rng_seed(&rng, seed);
    return mlp_create(sizes, n_sizes, act, &rng);
}

/* Overwrite the data of t with rows * cols values. */
static void set_values(Tensor *t, const double *values)
{
    memcpy(t->data, values, tensor_size(t) * sizeof *t->data);
}

static double max_abs(const Tensor *t)
{
    double max = 0.0;
    for (size_t i = 0; i < tensor_size(t); i++)
        max = fmax(max, fabs(t->data[i]));
    return max;
}

/* ---- construction ------------------------------------------------------------ */

static void create_shapes_and_kinds(void)
{
    const size_t sizes[] = {3, 5, 2};
    MLP *mlp = make_mlp(sizes, ARRAY_LEN(sizes), ACT_TANH, SEED);

    CHECK(mlp->n_layers == 2);
    CHECK(mlp->act == ACT_TANH);
    for (size_t i = 0; i < mlp->n_layers; i++) {
        const Linear *layer = &mlp->layers[i];
        CHECK(layer->W->rows == sizes[i] && layer->W->cols == sizes[i + 1]);
        CHECK(layer->b->rows == 1 && layer->b->cols == sizes[i + 1]);
        CHECK(layer->W->kind == TENSOR_PARAM && layer->W->grad != NULL);
        CHECK(layer->b->kind == TENSOR_PARAM && layer->b->grad != NULL);
    }

    mlp_free(mlp);
    mlp_free(NULL); /* must be a no-op */
}

/* Every weight lies inside (-limit, limit), the range is actually used, and
 * the biases are 0. */
static void check_init(Activation act, const size_t *sizes, size_t n_sizes,
                       const double *limits)
{
    MLP *mlp = make_mlp(sizes, n_sizes, act, SEED);

    for (size_t i = 0; i < mlp->n_layers; i++) {
        double largest = max_abs(mlp->layers[i].W);
        CHECK(largest < limits[i]);
        CHECK(largest > 0.5 * limits[i]);
        CHECK(max_abs(mlp->layers[i].b) == 0.0);
    }
    mlp_free(mlp);
}

static void init_he_for_relu(void)
{
    /* limit = sqrt(6 / in) */
    const size_t sizes[] = {4, 6, 3};
    const double limits[] = {sqrt(6.0 / 4.0), sqrt(6.0 / 6.0)};
    check_init(ACT_RELU, sizes, ARRAY_LEN(sizes), limits);
}

static void init_xavier_for_tanh_and_sigmoid(void)
{
    /* limit = sqrt(6 / (in + out)) */
    const size_t sizes[] = {4, 6, 3};
    const double limits[] = {sqrt(6.0 / 10.0), sqrt(6.0 / 9.0)};
    check_init(ACT_TANH, sizes, ARRAY_LEN(sizes), limits);
    check_init(ACT_SIGMOID, sizes, ARRAY_LEN(sizes), limits);
}

static void init_depends_only_on_seed(void)
{
    const size_t sizes[] = {3, 4};
    MLP *a = make_mlp(sizes, ARRAY_LEN(sizes), ACT_RELU, 1);
    MLP *b = make_mlp(sizes, ARRAY_LEN(sizes), ACT_RELU, 1);
    MLP *c = make_mlp(sizes, ARRAY_LEN(sizes), ACT_RELU, 2);
    size_t bytes = tensor_size(a->layers[0].W) * sizeof(double);

    CHECK(memcmp(a->layers[0].W->data, b->layers[0].W->data, bytes) == 0);
    CHECK(memcmp(a->layers[0].W->data, c->layers[0].W->data, bytes) != 0);

    mlp_free(a);
    mlp_free(b);
    mlp_free(c);
}

static void parameters_in_layer_order(void)
{
    const size_t sizes[] = {3, 5, 2};
    MLP *mlp = make_mlp(sizes, ARRAY_LEN(sizes), ACT_RELU, SEED);

    size_t count = 0;
    Tensor **params = mlp_parameters(mlp, &count);
    CHECK(count == 4);
    CHECK(params[0] == mlp->layers[0].W);
    CHECK(params[1] == mlp->layers[0].b);
    CHECK(params[2] == mlp->layers[1].W);
    CHECK(params[3] == mlp->layers[1].b);

    free(params); /* only the array: the tensors belong to the MLP */
    mlp_free(mlp);
}

/* ---- forward pass ------------------------------------------------------------- */

static void forward_single_layer_is_linear(void)
{
    /* One layer = no activation at all: out = x * W + b.
     *   x = [1  0]   W = [1 2 3]   b = [0.5 -1 2]
     *       [2 -1]       [4 5 6]
     *   row 0: [1 2 3] + b              = [ 1.5  1  5]
     *   row 1: 2*[1 2 3] - [4 5 6] + b  = [-1.5 -2  2] */
    const size_t sizes[] = {2, 3};
    MLP *mlp = make_mlp(sizes, ARRAY_LEN(sizes), ACT_SIGMOID, SEED);
    set_values(mlp->layers[0].W, (const double[]){1, 2, 3, 4, 5, 6});
    set_values(mlp->layers[0].b, (const double[]){0.5, -1, 2});
    Tensor *x = tensor_from_array(2, 2, (const double[]){1, 0, 2, -1});

    Tensor *out = mlp_forward(mlp, x);
    const double expected[] = {1.5, 1, 5, -1.5, -2, 2};
    CHECK(out->rows == 2 && out->cols == 3);
    CHECK(out->kind == TENSOR_INTERMEDIATE);
    for (size_t i = 0; i < ARRAY_LEN(expected); i++)
        CHECK_NEAR(out->data[i], expected[i], TOL);

    graph_free(out);
    tensor_free(x);
    mlp_free(mlp);
}

static void forward_two_layers_relu(void)
{
    /* x = [ 1 2]   W0 = [1 -1]   b0 = [0 -1]   W1 = [ 2]   b1 = [0.5]
     *     [-3 1]        [1  1]                      [-3]
     *   hidden row 0: [ 3 1] + b0 = [ 3 0] -> relu [3 0] -> 6 + 0.5 =  6.5
     *   hidden row 1: [-2 4] + b0 = [-2 3] -> relu [0 3] -> -9 + 0.5 = -8.5
     * The second output is negative: the last layer has no ReLU. */
    const size_t sizes[] = {2, 2, 1};
    MLP *mlp = make_mlp(sizes, ARRAY_LEN(sizes), ACT_RELU, SEED);
    set_values(mlp->layers[0].W, (const double[]){1, -1, 1, 1});
    set_values(mlp->layers[0].b, (const double[]){0, -1});
    set_values(mlp->layers[1].W, (const double[]){2, -3});
    set_values(mlp->layers[1].b, (const double[]){0.5});
    Tensor *x = tensor_from_array(2, 2, (const double[]){1, 2, -3, 1});

    Tensor *out = mlp_forward(mlp, x);
    CHECK(out->rows == 2 && out->cols == 1);
    CHECK_NEAR(out->data[0], 6.5, TOL);
    CHECK_NEAR(out->data[1], -8.5, TOL);

    graph_free(out);
    tensor_free(x);
    mlp_free(mlp);
}

/* ---- gradients of a whole network ---------------------------------------------- */

typedef struct {
    MLP *mlp;
    Tensor *x;
    Tensor *target;
} LossCase;

/* GraphFn for the gradient checker. The inputs array is not needed: it holds
 * the parameters of the MLP, which mlp_forward reaches on its own. */
static Tensor *f_mlp_mse(Tensor **inputs, void *ctx)
{
    (void)inputs;
    const LossCase *c = ctx;
    return ops_mse(mlp_forward(c->mlp, c->x), c->target);
}

/* Gradient check of an MLP with 3 inputs and 2 outputs on a batch of 6. */
static void check_gradients(const char *name, Activation act, const size_t *sizes,
                            size_t n_sizes)
{
    Rng rng;
    rng_seed(&rng, SEED);
    MLP *mlp = mlp_create(sizes, n_sizes, act, &rng);
    Tensor *x = tensor_create(6, 3);
    Tensor *target = tensor_create(6, 2);
    tensor_fill_uniform(x, &rng, -1.0, 1.0);
    tensor_fill_uniform(target, &rng, -1.0, 1.0);

    size_t n_params;
    Tensor **params = mlp_parameters(mlp, &n_params);
    /* The biases start at 0; move them so every parameter is random. */
    for (size_t i = 1; i < n_params; i += 2)
        tensor_fill_uniform(params[i], &rng, -0.5, 0.5);

    LossCase c = {.mlp = mlp, .x = x, .target = target};
    double error = gradcheck_max_error(f_mlp_mse, params, n_params, &c);
    printf("    %-22s max relative error %.2e\n", name, error);
    CHECK(error < GRADCHECK_TOLERANCE);

    free(params);
    tensor_free(x);
    tensor_free(target);
    mlp_free(mlp);
}

static const size_t THREE_LAYERS[] = {3, 5, 4, 2};
static const size_t TWO_LAYERS[] = {3, 5, 2};

static void gradients_relu(void)
{
    check_gradients("mlp relu", ACT_RELU, THREE_LAYERS, ARRAY_LEN(THREE_LAYERS));
}

static void gradients_tanh(void)
{
    check_gradients("mlp tanh", ACT_TANH, THREE_LAYERS, ARRAY_LEN(THREE_LAYERS));
}

static void gradients_sigmoid(void)
{
    /* Two layers only. The derivative of the sigmoid is at most 0.25, so
     * behind two sigmoid layers the gradients of the first layer are tiny
     * (around 4e-5 here). The finite difference has a rounding error of about
     * 1e-10 whatever the size of the gradient, which is then 1e-6 of the
     * value: the limit of the checker, not an error in the backward pass. */
    check_gradients("mlp sigmoid", ACT_SIGMOID, TWO_LAYERS, ARRAY_LEN(TWO_LAYERS));
}

/* ---- training ------------------------------------------------------------------- */

static void training_reduces_loss(void)
{
    /* Fit y = sin(2 * x) on 16 points with a 1-8-1 tanh network. */
    Rng rng;
    rng_seed(&rng, SEED);
    Tensor *x = tensor_create(16, 1);
    Tensor *y = tensor_create(16, 1);
    for (size_t i = 0; i < 16; i++) {
        x->data[i] = -1.0 + 2.0 * (double)i / 15.0;
        y->data[i] = sin(2.0 * x->data[i]);
    }

    const size_t sizes[] = {1, 8, 1};
    MLP *mlp = mlp_create(sizes, ARRAY_LEN(sizes), ACT_TANH, &rng);
    size_t n_params;
    Tensor **params = mlp_parameters(mlp, &n_params);
    Optimizer *opt = optim_sgd_create(params, n_params, 0.1);
    free(params);

    double first_loss = 0.0, last_loss = 0.0;
    for (int step = 0; step < 1000; step++) {
        Tensor *loss = ops_mse(mlp_forward(mlp, x), y);
        if (step == 0)
            first_loss = loss->data[0];
        last_loss = loss->data[0];

        optim_zero_grad(opt);
        graph_backward(loss);
        optim_step(opt);
        graph_free(loss);
    }
    CHECK(isfinite(last_loss));
    CHECK(last_loss < 0.1 * first_loss);

    optim_free(opt);
    mlp_free(mlp);
    tensor_free(x);
    tensor_free(y);
}

static const TestCase tests[] = {
    {"create_shapes_and_kinds", create_shapes_and_kinds},
    {"init_he_for_relu", init_he_for_relu},
    {"init_xavier_for_tanh_and_sigmoid", init_xavier_for_tanh_and_sigmoid},
    {"init_depends_only_on_seed", init_depends_only_on_seed},
    {"parameters_in_layer_order", parameters_in_layer_order},
    {"forward_single_layer_is_linear", forward_single_layer_is_linear},
    {"forward_two_layers_relu", forward_two_layers_relu},
    {"gradients_relu", gradients_relu},
    {"gradients_tanh", gradients_tanh},
    {"gradients_sigmoid", gradients_sigmoid},
    {"training_reduces_loss", training_reduces_loss},
};

const TestSuite nn_suite = {"nn", tests, ARRAY_LEN(tests)};
