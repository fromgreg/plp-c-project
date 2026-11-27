/* Tests for optim: the SGD update, zero_grad, ownership of the parameter
 * list, and a small optimisation run with the full training-step lifecycle. */
#include <stdlib.h>

#include "autograd.h"
#include "ops.h"
#include "optim.h"
#include "tensor.h"
#include "test.h"
#include "util.h"

#define TOL 1e-12

static void step_moves_against_gradient(void)
{
    /* data -= lr * grad:  1 - 0.1 * 0.5 = 0.95,  2 - 0.1 * (-1) = 2.1 */
    Tensor *p = tensor_create_param(1, 2);
    p->data[0] = 1.0;
    p->data[1] = 2.0;
    p->grad[0] = 0.5;
    p->grad[1] = -1.0;
    Optimizer *opt = optim_sgd_create(&p, 1, 0.1); /* &p: an array of one */

    optim_step(opt);
    CHECK_NEAR(p->data[0], 0.95, TOL);
    CHECK_NEAR(p->data[1], 2.1, TOL);
    /* The step does not clear the gradients: that is optim_zero_grad's job. */
    CHECK(p->grad[0] == 0.5 && p->grad[1] == -1.0);

    optim_free(opt);
    optim_free(NULL); /* must be a no-op */
    tensor_free(p);
}

static void zero_grad_clears_every_parameter(void)
{
    Tensor *a = tensor_create_param(2, 2);
    Tensor *b = tensor_create_param(1, 3);
    for (size_t i = 0; i < tensor_size(a); i++)
        a->grad[i] = 1.0;
    for (size_t i = 0; i < tensor_size(b); i++)
        b->grad[i] = -2.0;
    Tensor *params[] = {a, b};
    Optimizer *opt = optim_sgd_create(params, ARRAY_LEN(params), 0.1);

    optim_zero_grad(opt);
    for (size_t i = 0; i < tensor_size(a); i++)
        CHECK(a->grad[i] == 0.0);
    for (size_t i = 0; i < tensor_size(b); i++)
        CHECK(b->grad[i] == 0.0);

    optim_free(opt);
    tensor_free(a);
    tensor_free(b);
}

static void keeps_own_copy_of_the_list(void)
{
    Tensor *a = tensor_create_param(1, 1);
    Tensor *b = tensor_create_param(1, 1);
    a->grad[0] = 1.0;
    b->grad[0] = 2.0;

    /* The caller's array goes away right after the optimizer is created. If
     * the optimizer had kept the pointer, ASan would report the next step. */
    Tensor **list = xmalloc(2 * sizeof *list);
    list[0] = a;
    list[1] = b;
    Optimizer *opt = optim_sgd_create(list, 2, 0.5);
    list[0] = NULL;
    list[1] = NULL;
    free(list);

    optim_step(opt);
    CHECK_NEAR(a->data[0], -0.5, TOL);
    CHECK_NEAR(b->data[0], -1.0, TOL);

    optim_free(opt);
    tensor_free(a);
    tensor_free(b);
}

static void minimises_mse(void)
{
    /* loss = mean((x - t)^2) over 3 elements, so grad = 2 (x - t) / 3. With
     * lr = 0.3 one step is x -= 0.2 (x - t): the distance to t shrinks by
     * the factor 0.8 per step, and 0.8^200 is about 4e-20. */
    Tensor *x = tensor_create_param(1, 3);
    Tensor *target = tensor_from_array(1, 3, (const double[]){1.0, -2.0, 3.0});
    Optimizer *opt = optim_sgd_create(&x, 1, 0.3);

    for (int step = 0; step < 200; step++) {
        Tensor *loss = ops_mse(x, target);
        optim_zero_grad(opt);
        graph_backward(loss);
        optim_step(opt);
        graph_free(loss);
    }
    for (size_t i = 0; i < 3; i++)
        CHECK_NEAR(x->data[i], target->data[i], 1e-9);

    optim_free(opt);
    tensor_free(x);
    tensor_free(target);
}

static const TestCase tests[] = {
    {"step_moves_against_gradient", step_moves_against_gradient},
    {"zero_grad_clears_every_parameter", zero_grad_clears_every_parameter},
    {"keeps_own_copy_of_the_list", keeps_own_copy_of_the_list},
    {"minimises_mse", minimises_mse},
};

const TestSuite optim_suite = {"optim", tests, ARRAY_LEN(tests)};
