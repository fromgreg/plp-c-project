/* gradcheck: compare the gradients of the backward pass with finite differences.
 *
 * The backward pass gives an "analytic" gradient. The same derivative can be
 * approximated without any calculus: nudge one input element by a small h in
 * both directions and see how much the result changes,
 *
 *     d f / d x  ~=  (f(x + h) - f(x - h)) / (2 h)      (central difference).
 *
 * This is far too slow for training (two forward passes per element), but it
 * is an independent check: if the two numbers agree for every element, the
 * backward function is correct. */
#ifndef CGRAD_GRADCHECK_H
#define CGRAD_GRADCHECK_H

#include <stddef.h>

#include "tensor.h"

/* The function under test: builds a graph from inputs and returns its root.
 * The checker calls it many times and frees each graph itself. ctx is passed
 * through unchanged, for anything else the function needs (fixed data, labels). */
typedef Tensor *(*GraphFn)(Tensor **inputs, void *ctx);

/* Step h of the central difference. Smaller is not better: the error of the
 * formula shrinks with h^2, but the rounding error of f grows with 1/h. */
#define GRADCHECK_STEP 1e-6

/* A check passes if the largest relative error stays below this. */
#define GRADCHECK_TOLERANCE 1e-6

/* Largest relative error |a - n| / max(1e-8, |a| + |n|) between the analytic
 * gradient a and the numeric gradient n, over every element of every input.
 * NaN if any of them is NaN.
 *
 * The inputs must have grad buffers (params). Their data is perturbed during
 * the check and restored afterwards; their grads hold the analytic gradient
 * when the function returns. If the root is not 1x1, the function checked is
 * the sum of its elements (the same convention as graph_backward). */
double gradcheck_max_error(GraphFn f, Tensor **inputs, size_t n_inputs, void *ctx);

#endif /* CGRAD_GRADCHECK_H */
