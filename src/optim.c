#include "optim.h"

#include <stdlib.h>
#include <string.h>

#include "util.h"

Optimizer *optim_sgd_create(Tensor **params, size_t n_params, double lr)
{
    if (n_params == 0)
        fatal("optim_sgd_create: no parameters");
    /* Written as !(lr > 0) so that a NaN is rejected too. */
    if (!(lr > 0.0))
        fatal("optim_sgd_create: learning rate must be positive, got %g", lr);
    /* Checked once here, so optim_step can use grad without testing it. */
    for (size_t i = 0; i < n_params; i++)
        if (params[i]->grad == NULL)
            fatal("optim_sgd_create: parameter %zu has no grad buffer", i);

    Optimizer *opt = xmalloc(sizeof *opt);
    opt->n_params = n_params;
    opt->lr = lr;
    /* Copy the pointers, not the tensors: n_params * sizeof(Tensor *) bytes. */
    opt->params = xmalloc(n_params * sizeof *opt->params);
    memcpy(opt->params, params, n_params * sizeof *opt->params);
    return opt;
}

void optim_free(Optimizer *opt)
{
    if (opt == NULL)
        return;
    free(opt->params);
    free(opt);
}

void optim_zero_grad(Optimizer *opt)
{
    for (size_t i = 0; i < opt->n_params; i++)
        tensor_zero_grad(opt->params[i]);
}

void optim_step(Optimizer *opt)
{
    for (size_t i = 0; i < opt->n_params; i++) {
        Tensor *p = opt->params[i];
        size_t n = tensor_size(p);
        /* The gradient points in the direction in which the loss grows
         * fastest, so step the other way. */
        for (size_t j = 0; j < n; j++)
            p->data[j] -= opt->lr * p->grad[j];
    }
}
