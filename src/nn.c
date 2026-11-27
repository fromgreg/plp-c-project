#include "nn.h"

#include <math.h>
#include <stdlib.h>

#include "ops.h"
#include "util.h"

/* ---- one layer --------------------------------------------------------------- */

/* Half-width of the uniform distribution the weights are drawn from.
 *
 * The goal of both schemes is that the signal keeps roughly the same size
 * from layer to layer, so it neither dies out nor blows up in a deep network.
 * U(-limit, limit) has variance limit^2 / 3:
 *     He     (ReLU):          variance 2 / in          -> limit = sqrt(6 / in)
 *     Xavier (tanh, sigmoid): variance 2 / (in + out)  -> limit = sqrt(6 / (in + out))
 * ReLU sets half of its inputs to 0, which He makes up for with the larger
 * variance. */
static double init_limit(size_t in, size_t out, Activation act)
{
    if (act == ACT_RELU)
        return sqrt(6.0 / (double)in);
    return sqrt(6.0 / (double)(in + out));
}

/* Allocate and initialise the parameters of one layer in place. */
static void linear_init(Linear *layer, size_t in, size_t out, Activation act, Rng *rng)
{
    double limit = init_limit(in, out, act);

    layer->W = tensor_create_param(in, out);
    tensor_fill_uniform(layer->W, rng, -limit, limit);
    /* The bias stays 0: tensor_create_param zero-fills. The random weights
     * already make the units of a layer different from each other. */
    layer->b = tensor_create_param(1, out);
}

/* x (n x in) -> x * W + b (n x out). */
static Tensor *linear_forward(const Linear *layer, Tensor *x)
{
    return ops_add_bias(ops_matmul(x, layer->W), layer->b);
}

/* Apply the activation function selected by act.
 *
 * Dispatch by switch, not by function pointer: the set of activations is
 * small and fixed, the choice sits in one place, and the compiler warns if a
 * new enum value is added without a case here. Tensor.backward is the
 * opposite situation: every op brings its own function, and autograd.c calls
 * it without knowing which ops exist. */
static Tensor *activate(Tensor *x, Activation act)
{
    switch (act) {
    case ACT_RELU:
        return ops_relu(x);
    case ACT_TANH:
        return ops_tanh(x);
    case ACT_SIGMOID:
        return ops_sigmoid(x);
    }
    /* Only reached if act holds a value that is not in the enum. */
    fatal("mlp_forward: unknown activation %d", (int)act);
}

/* ---- MLP ----------------------------------------------------------------------- */

MLP *mlp_create(const size_t *sizes, size_t n_sizes, Activation act, Rng *rng)
{
    if (n_sizes < 2)
        fatal("mlp_create: need at least an input and an output size, got %zu size(s)",
              n_sizes);

    MLP *mlp = xmalloc(sizeof *mlp);
    mlp->n_layers = n_sizes - 1;
    mlp->act = act;
    /* One block for all Linear structs: layers[i] is a struct, not a pointer. */
    mlp->layers = xmalloc(mlp->n_layers * sizeof *mlp->layers);

    /* Layer i maps sizes[i] inputs to sizes[i + 1] outputs. */
    for (size_t i = 0; i < mlp->n_layers; i++)
        linear_init(&mlp->layers[i], sizes[i], sizes[i + 1], act, rng);
    return mlp;
}

void mlp_free(MLP *mlp)
{
    if (mlp == NULL)
        return;
    for (size_t i = 0; i < mlp->n_layers; i++) {
        tensor_free(mlp->layers[i].W);
        tensor_free(mlp->layers[i].b);
    }
    free(mlp->layers);
    free(mlp);
}

Tensor *mlp_forward(const MLP *mlp, Tensor *x)
{
    Tensor *h = x;

    for (size_t i = 0; i < mlp->n_layers; i++) {
        h = linear_forward(&mlp->layers[i], h);
        if (i + 1 < mlp->n_layers) /* no activation after the last layer */
            h = activate(h, mlp->act);
    }
    /* The intermediate results are not lost when h is overwritten: each one
     * is a parent of the next, so graph_free finds them all from the root. */
    return h;
}

Tensor **mlp_parameters(const MLP *mlp, size_t *count)
{
    size_t n = 2 * mlp->n_layers;
    Tensor **params = xmalloc(n * sizeof *params);

    for (size_t i = 0; i < mlp->n_layers; i++) {
        params[2 * i] = mlp->layers[i].W;
        params[2 * i + 1] = mlp->layers[i].b;
    }
    *count = n;
    return params;
}
