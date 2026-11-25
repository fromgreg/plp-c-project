/* ops: differentiable operations.
 *
 * Every op computes its result (forward) and records in the result node how
 * to propagate gradients back to its inputs (parents + backward function).
 * Calling ops therefore builds the computational graph as a side effect.
 *
 * Every result is a TENSOR_INTERMEDIATE and graph-owned: release the whole
 * graph with graph_free(root) (autograd.h), never with tensor_free().
 *
 * The inputs are not const: the result keeps them as parents, and the backward
 * pass later writes into their grad buffers. The inputs must outlive the graph.
 *
 * Every op aborts with a message if the shapes of its inputs do not fit. */
#ifndef CGRAD_OPS_H
#define CGRAD_OPS_H

#include <stddef.h>

#include "tensor.h"

/* ---- elementwise, two inputs of the same shape ------------------------------ */

/* a + b. */
Tensor *ops_add(Tensor *a, Tensor *b);

/* a * b, element by element (not the matrix product). */
Tensor *ops_mul(Tensor *a, Tensor *b);

/* ---- linear layer building blocks -------------------------------------------- */

/* Matrix product a (n x k) * b (k x m) -> n x m. */
Tensor *ops_matmul(Tensor *a, Tensor *b);

/* x (n x m) + bias (1 x m): the bias row is added to every row of x. This is
 * the only form of broadcasting in the library. */
Tensor *ops_add_bias(Tensor *x, Tensor *bias);

/* ---- reductions to a 1x1 scalar ---------------------------------------------- */

/* Sum / arithmetic mean of all elements of x. */
Tensor *ops_sum(Tensor *x);
Tensor *ops_mean(Tensor *x);

/* ---- activations, elementwise, same shape as x ------------------------------- */

/* max(x, 0). The derivative at exactly 0 is taken as 0. */
Tensor *ops_relu(Tensor *x);

Tensor *ops_tanh(Tensor *x);

/* 1 / (1 + exp(-x)), computed without overflow for large |x|. */
Tensor *ops_sigmoid(Tensor *x);

/* ---- losses, result is 1x1 ---------------------------------------------------- */

/* Mean squared error: the mean of (pred - target)^2 over all elements.
 * pred and target must have the same shape. */
Tensor *ops_mse(Tensor *pred, Tensor *target);

/* Softmax followed by cross-entropy, averaged over the rows of logits (n x c,
 * one row per sample, one column per class). labels holds n class indices,
 * each below c. The two steps are fused into one op because the combination
 * is numerically stable and has a simple gradient, (softmax - onehot) / n.
 *
 * labels is copied; the caller keeps its array and may change it afterwards. */
Tensor *ops_softmax_cross_entropy(Tensor *logits, const size_t *labels);

#endif /* CGRAD_OPS_H */
