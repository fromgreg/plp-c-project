/* optim: stochastic gradient descent (SGD).
 *
 * The optimizer knows which tensors are the parameters and changes them using
 * the gradients that graph_backward left in their grad buffers. One training
 * step:
 *
 *     Tensor *loss = ...forward pass...;
 *     optim_zero_grad(opt);     gradients accumulate, so clear the old ones
 *     graph_backward(loss);     fill the grad buffers
 *     optim_step(opt);          move every parameter against its gradient
 *     graph_free(loss);         release the intermediates of this step
 *
 * The optimizer does not own the parameters. They belong to the model and
 * must outlive the optimizer. */
#ifndef CGRAD_OPTIM_H
#define CGRAD_OPTIM_H

#include <stddef.h>

#include "tensor.h"

typedef struct {
    Tensor **params; /* own copy of the pointer array; the tensors are borrowed */
    size_t n_params;
    double lr;       /* learning rate: the step size */
} Optimizer;

/* New SGD optimizer for n_params parameters. The pointer array is copied, so
 * the caller may free or reuse its own array right after the call. Caller
 * frees the optimizer with optim_free.
 * Aborts if n_params is 0, lr is not positive, or a parameter has no grad
 * buffer. */
Optimizer *optim_sgd_create(Tensor **params, size_t n_params, double lr);

/* Free the optimizer. The parameters are not touched. NULL is a no-op. */
void optim_free(Optimizer *opt);

/* Set the gradient of every parameter to 0. */
void optim_zero_grad(Optimizer *opt);

/* One SGD update of every parameter: data -= lr * grad. The gradients are
 * left as they are. */
void optim_step(Optimizer *opt);

#endif /* CGRAD_OPTIM_H */
