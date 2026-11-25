/* autograd: reverse-mode automatic differentiation on the graph built by ops.
 *
 * The graph is the set of tensors reachable from a root by following the
 * parents pointers. One training step looks like this:
 *
 *     Tensor *loss = ...ops...;     build the graph (forward pass)
 *     tensor_zero_grad(param);      for every parameter
 *     graph_backward(loss);         fill the grad buffers
 *     ...use the gradients...
 *     graph_free(loss);             release every intermediate
 *
 * The traversal is recursive, so the longest path in the graph is limited by
 * the call stack. A network has a few nodes per layer; chains of several
 * thousand nodes are still fine. */
#ifndef CGRAD_AUTOGRAD_H
#define CGRAD_AUTOGRAD_H

#include <stddef.h>

#include "tensor.h"

/* Topological order of the graph below root: every reachable tensor exactly
 * once, each one after all of its parents, root last. The number of entries
 * is stored in *count. Caller frees the returned array with free(); the
 * tensors in it are borrowed. */
Tensor **graph_topo_order(Tensor *root, size_t *count);

/* Backward pass: compute d(root)/d(t) for every tensor t below root that has
 * a grad buffer, and ADD it to t->grad.
 *
 * root->grad is set to 1 for every element. For a 1x1 root (a loss) this is
 * d(root)/d(root) = 1; for a larger root the result is the gradient of the
 * sum of its elements.
 *
 * Gradients accumulate: zero the parameters' grads first, and call this only
 * once per graph. Aborts if root has no grad buffer (a plain leaf). */
void graph_backward(Tensor *root);

/* Free every TENSOR_INTERMEDIATE below root, including root itself if it is
 * one. Leaves and params are left alone: they belong to the caller. After
 * this call, every pointer to an intermediate of this graph is dangling.
 * Two roots that share intermediates must not both be freed this way. */
void graph_free(Tensor *root);

#endif /* CGRAD_AUTOGRAD_H */
