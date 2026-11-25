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
 * pass later writes into their grad buffers. The inputs must outlive the graph. */
#ifndef CGRAD_OPS_H
#define CGRAD_OPS_H

#include "tensor.h"

/* Elementwise a + b. Aborts unless a and b have the same shape. Graph-owned. */
Tensor *ops_add(Tensor *a, Tensor *b);

/* Elementwise a * b (not the matrix product). Aborts unless a and b have the
 * same shape. Graph-owned. */
Tensor *ops_mul(Tensor *a, Tensor *b);

#endif /* CGRAD_OPS_H */
