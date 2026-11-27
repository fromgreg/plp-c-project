/* nn: fully connected layers and the multi-layer perceptron (MLP).
 *
 * An MLP is a chain of linear layers, y = x * W + b, with an activation
 * function between them. It owns its parameters (W and b of every layer):
 * mlp_create allocates them, mlp_free releases them.
 *
 * mlp_forward builds a graph with ops (ops.h), so its result is graph-owned
 * like every op result. */
#ifndef CGRAD_NN_H
#define CGRAD_NN_H

#include <stddef.h>

#include "rng.h"
#include "tensor.h"

/* Activation function between the layers. */
typedef enum {
    ACT_RELU,
    ACT_TANH,
    ACT_SIGMOID
} Activation;

/* One fully connected layer: y = x * W + b for a batch x (n x in). */
typedef struct {
    Tensor *W; /* in x out, TENSOR_PARAM */
    Tensor *b; /* 1 x out,  TENSOR_PARAM */
} Linear;

typedef struct {
    size_t n_layers;
    Linear *layers; /* array of n_layers structs, owned by the MLP */
    Activation act; /* applied after every layer except the last */
} MLP;

/* New MLP with n_sizes - 1 layers. sizes lists the width of the input and of
 * every layer: {2, 8, 1} is a network with 2 inputs, one hidden layer of 8
 * units and 1 output. sizes is only read during the call.
 *
 * Weights are drawn from rng (He uniform for ReLU, Xavier uniform for tanh
 * and sigmoid); biases start at 0. Caller frees with mlp_free.
 * Aborts if n_sizes < 2 or a size is 0. */
MLP *mlp_create(const size_t *sizes, size_t n_sizes, Activation act, Rng *rng);

/* Free the MLP together with all of its parameters. NULL is a no-op. */
void mlp_free(MLP *mlp);

/* Forward pass for a batch x (n x in, one sample per row) -> n x out.
 * The last layer has no activation: the result is raw scores, ready for
 * ops_mse or ops_softmax_cross_entropy.
 *
 * Graph-owned: release with graph_free() on the result or on a loss computed
 * from it. x must outlive that graph. */
Tensor *mlp_forward(const MLP *mlp, Tensor *x);

/* All parameters in the order W0, b0, W1, b1, ... as a new array; the number
 * of entries (2 * n_layers) is stored in *count. Caller frees the array with
 * free(); the tensors in it stay owned by the MLP. */
Tensor **mlp_parameters(const MLP *mlp, size_t *count);

#endif /* CGRAD_NN_H */
