#include "gradcheck.h"

#include <math.h>

#include "autograd.h"
#include "util.h"

/* Build the graph once, return the value of its root (the sum of its
 * elements) and free the graph again. */
static double evaluate(GraphFn f, Tensor **inputs, void *ctx)
{
    Tensor *root = f(inputs, ctx);

    size_t n = tensor_size(root);
    double value = 0.0;
    for (size_t i = 0; i < n; i++)
        value += root->data[i];

    graph_free(root);
    return value;
}

double gradcheck_max_error(GraphFn f, Tensor **inputs, size_t n_inputs, void *ctx)
{
    /* Analytic gradients: one forward and one backward pass. graph_free only
     * releases the intermediates, so the result stays in inputs[k]->grad. */
    for (size_t k = 0; k < n_inputs; k++) {
        if (inputs[k]->grad == NULL)
            fatal("gradcheck: input %zu has no grad buffer (not a param)", k);
        tensor_zero_grad(inputs[k]);
    }
    Tensor *root = f(inputs, ctx);
    graph_backward(root);
    graph_free(root);

    /* Numeric gradients: two forward passes per element, no backward pass, so
     * the analytic gradients above are not touched. */
    double max_error = 0.0;
    for (size_t k = 0; k < n_inputs; k++) {
        Tensor *x = inputs[k];
        size_t n = tensor_size(x);

        for (size_t i = 0; i < n; i++) {
            double saved = x->data[i];

            x->data[i] = saved + GRADCHECK_STEP;
            double f_plus = evaluate(f, inputs, ctx);
            x->data[i] = saved - GRADCHECK_STEP;
            double f_minus = evaluate(f, inputs, ctx);
            x->data[i] = saved; /* restore exactly, not by adding h back */

            double numeric = (f_plus - f_minus) / (2.0 * GRADCHECK_STEP);
            double analytic = x->grad[i];

            /* Relative error: independent of the size of the gradient. The
             * floor of 1e-8 avoids 0/0 when both gradients are zero. */
            double error = fabs(analytic - numeric)
                           / fmax(1e-8, fabs(analytic) + fabs(numeric));

            /* A NaN would lose every comparison below and go unnoticed. */
            if (isnan(error))
                return error;
            if (error > max_error)
                max_error = error;
        }
    }
    return max_error;
}
