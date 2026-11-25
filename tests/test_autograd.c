/* Tests for autograd: topological order, backward pass, graph teardown.
 * Expected gradients are derived by hand in the comment of each test. */
#include <stdlib.h>

#include "autograd.h"
#include "ops.h"
#include "tensor.h"
#include "test.h"
#include "util.h"

#define TOL 1e-12

/* 1x1 parameter holding value. Caller frees. */
static Tensor *scalar_param(double value)
{
    Tensor *t = tensor_create_param(1, 1);
    t->data[0] = value;
    return t;
}

/* Position of t in order, or count if it is not in there. */
static size_t index_of(Tensor **order, size_t count, const Tensor *t)
{
    for (size_t i = 0; i < count; i++)
        if (order[i] == t)
            return i;
    return count;
}

/* ---- single ops on matrices ------------------------------------------------ */

static void add_gradients(void)
{
    /* out = a + b; the root is seeded with ones, so both grads are all ones. */
    Tensor *a = tensor_create_param(2, 2);
    Tensor *b = tensor_create_param(2, 2);
    Tensor *out = ops_add(a, b);
    graph_backward(out);

    for (size_t i = 0; i < 4; i++) {
        CHECK(a->grad[i] == 1.0);
        CHECK(b->grad[i] == 1.0);
    }
    graph_free(out);
    tensor_free(a);
    tensor_free(b);
}

static void mul_gradients(void)
{
    /* out = a * b elementwise: d(out)/da = b and d(out)/db = a. */
    Tensor *a = tensor_create_param(2, 2);
    Tensor *b = tensor_create_param(2, 2);
    for (size_t i = 0; i < 4; i++) {
        a->data[i] = (double)i + 1.0;   /* 1 2 3 4 */
        b->data[i] = 10.0 * (double)i;  /* 0 10 20 30 */
    }
    Tensor *out = ops_mul(a, b);
    graph_backward(out);

    for (size_t i = 0; i < 4; i++) {
        CHECK(a->grad[i] == b->data[i]);
        CHECK(b->grad[i] == a->data[i]);
    }
    graph_free(out);
    tensor_free(a);
    tensor_free(b);
}

/* ---- composed expressions -------------------------------------------------- */

static void scalar_expression(void)
{
    /* f = (a*b + a) * b = a*b^2 + a*b
     * df/da = b^2 + b   = 12   at a = 2, b = 3
     * df/db = 2*a*b + a = 14 */
    Tensor *a = scalar_param(2.0);
    Tensor *b = scalar_param(3.0);
    Tensor *f = ops_mul(ops_add(ops_mul(a, b), a), b);
    graph_backward(f);

    CHECK_NEAR(f->data[0], 24.0, TOL);
    CHECK_NEAR(a->grad[0], 12.0, TOL);
    CHECK_NEAR(b->grad[0], 14.0, TOL);

    graph_free(f);
    tensor_free(a);
    tensor_free(b);
}

static void diamond(void)
{
    /* s feeds two ops (p and q) whose results meet again in f:
     *
     *     s = a*b      p = s + a      q = s * b      f = p * q
     *
     * f = (a*b + a) * (a*b^2)
     * df/da = (b + 1)*a*b^2 + (a*b + a)*b^2     = 72 + 72 = 144   at a = 2, b = 3
     * df/db = a*a*b^2 + (a*b + a)*2*a*b         = 36 + 96 = 132
     *
     * This only works if s adds up the gradients from p and q, and passes the
     * sum on to a and b exactly once. */
    Tensor *a = scalar_param(2.0);
    Tensor *b = scalar_param(3.0);
    Tensor *s = ops_mul(a, b);
    Tensor *p = ops_add(s, a);
    Tensor *q = ops_mul(s, b);
    Tensor *f = ops_mul(p, q);
    graph_backward(f);

    CHECK_NEAR(f->data[0], 144.0, TOL);
    CHECK_NEAR(s->grad[0], 42.0, TOL); /* df/ds = q + p*b = 18 + 24 */
    CHECK_NEAR(a->grad[0], 144.0, TOL);
    CHECK_NEAR(b->grad[0], 132.0, TOL);

    graph_free(f);
    tensor_free(a);
    tensor_free(b);
}

static void same_tensor_twice(void)
{
    /* y = x * x: dy/dx = 2x = 6 at x = 3. Both parents are x. */
    Tensor *x = scalar_param(3.0);
    Tensor *y = ops_mul(x, x);
    graph_backward(y);
    CHECK_NEAR(y->data[0], 9.0, TOL);
    CHECK_NEAR(x->grad[0], 6.0, TOL);
    graph_free(y);

    /* z = x + x: dz/dx = 2. */
    tensor_zero_grad(x);
    Tensor *z = ops_add(x, x);
    graph_backward(z);
    CHECK_NEAR(x->grad[0], 2.0, TOL);
    graph_free(z);

    tensor_free(x);
}

static void leaf_without_grad(void)
{
    /* c is plain data (like an input batch): no grad buffer, so backward must
     * skip it and still produce the gradient of w. */
    double cv[] = {4, 5};
    Tensor *c = tensor_from_array(1, 2, cv);
    Tensor *w = tensor_create_param(1, 2);
    Tensor *out = ops_mul(c, w);
    graph_backward(out);

    CHECK(c->grad == NULL);
    CHECK(w->grad[0] == 4.0 && w->grad[1] == 5.0);

    graph_free(out);
    tensor_free(c);
    tensor_free(w);
}

/* ---- topological order ----------------------------------------------------- */

static void topo_order(void)
{
    /* Same diamond as above: 6 nodes, s is reachable along two paths. */
    Tensor *a = scalar_param(2.0);
    Tensor *b = scalar_param(3.0);
    Tensor *s = ops_mul(a, b);
    Tensor *p = ops_add(s, a);
    Tensor *q = ops_mul(s, b);
    Tensor *f = ops_mul(p, q);

    size_t count = 0;
    Tensor **order = graph_topo_order(f, &count);

    CHECK(count == 6); /* every node once: no duplicates, nothing missing */
    CHECK(order[count - 1] == f);
    Tensor *nodes[] = {a, b, s, p, q, f};
    for (size_t n = 0; n < ARRAY_LEN(nodes); n++) {
        size_t pos = index_of(order, count, nodes[n]);
        CHECK(pos < count);
        /* every parent comes before its child */
        for (int k = 0; k < nodes[n]->n_parents; k++)
            CHECK(index_of(order, count, nodes[n]->parents[k]) < pos);
    }
    free(order);

    /* A second sort gets a new epoch and must see all nodes again. */
    order = graph_topo_order(f, &count);
    CHECK(count == 6);
    free(order);

    graph_free(f);
    tensor_free(a);
    tensor_free(b);
}

static void deep_chain(void)
{
    /* x_0 = a, x_i = x_(i-1) + a: after N steps x = (N + 1) * a, so
     * dx/da = N + 1. The longest path has N + 1 nodes, which is also the
     * recursion depth of the topological sort. */
    enum { N = 10000 };
    Tensor *a = scalar_param(1.0);
    Tensor *x = a;
    for (int i = 0; i < N; i++)
        x = ops_add(x, a);
    graph_backward(x);

    CHECK_NEAR(x->data[0], (double)N + 1.0, TOL);
    CHECK_NEAR(a->grad[0], (double)N + 1.0, TOL);

    graph_free(x);
    tensor_free(a);
}

/* ---- accumulation and zero_grad -------------------------------------------- */

static void zero_grad_between_graphs(void)
{
    /* y = x * x, dy/dx = 6 at x = 3. */
    Tensor *x = scalar_param(3.0);

    Tensor *y = ops_mul(x, x);
    graph_backward(y);
    graph_free(y);
    CHECK_NEAR(x->grad[0], 6.0, TOL);

    /* Without zero_grad the second pass adds on top of the first. */
    y = ops_mul(x, x);
    graph_backward(y);
    graph_free(y);
    CHECK_NEAR(x->grad[0], 12.0, TOL);

    tensor_zero_grad(x);
    y = ops_mul(x, x);
    graph_backward(y);
    graph_free(y);
    CHECK_NEAR(x->grad[0], 6.0, TOL);

    tensor_free(x);
}

/* ---- teardown ---------------------------------------------------------------
 * The real checks here are made by the tools: ASan reports a wrong or double
 * free, and `leaks` reports anything graph_free missed. */

static void graph_free_keeps_leaves_and_params(void)
{
    double cv[] = {5.0};
    Tensor *a = scalar_param(2.0);
    Tensor *c = tensor_from_array(1, 1, cv);
    Tensor *f = ops_add(ops_mul(a, c), a);
    graph_backward(f);
    graph_free(f);

    /* a and c must still be alive and unchanged. */
    CHECK(a->data[0] == 2.0);
    CHECK_NEAR(a->grad[0], 6.0, TOL); /* d(a*c + a)/da = c + 1 */
    CHECK(c->data[0] == 5.0);

    tensor_free(a);
    tensor_free(c);
}

static void graph_free_leaf_root(void)
{
    /* A root that is not an intermediate: nothing to free. */
    Tensor *a = scalar_param(1.0);
    graph_free(a);
    CHECK(a->data[0] == 1.0);
    tensor_free(a);
}

static void ctx_freed_with_node(void)
{
    /* add and mul have no saved state, so attach some by hand: the node owns
     * its ctx, and graph_free must release it together with the node. */
    Tensor *a = scalar_param(1.0);
    Tensor *out = ops_add(a, a);
    out->ctx = xmalloc(64);
    graph_free(out);
    CHECK(a->data[0] == 1.0);
    tensor_free(a);
}

static void thousand_graphs(void)
{
    /* One "training step" 1000 times: build, backward, check, free. Memory
     * must not grow: every step releases all of its intermediates. */
    Tensor *a = scalar_param(2.0);
    Tensor *b = scalar_param(3.0);
    int all_correct = 1;

    for (int step = 0; step < 1000; step++) {
        tensor_zero_grad(a);
        tensor_zero_grad(b);
        Tensor *f = ops_mul(ops_add(ops_mul(a, b), a), b);
        graph_backward(f);
        all_correct = all_correct && a->grad[0] == 12.0 && b->grad[0] == 14.0;
        graph_free(f);
    }
    CHECK(all_correct);

    tensor_free(a);
    tensor_free(b);
}

static const TestCase tests[] = {
    {"add_gradients", add_gradients},
    {"mul_gradients", mul_gradients},
    {"scalar_expression", scalar_expression},
    {"diamond", diamond},
    {"same_tensor_twice", same_tensor_twice},
    {"leaf_without_grad", leaf_without_grad},
    {"topo_order", topo_order},
    {"deep_chain", deep_chain},
    {"zero_grad_between_graphs", zero_grad_between_graphs},
    {"graph_free_keeps_leaves_and_params", graph_free_keeps_leaves_and_params},
    {"graph_free_leaf_root", graph_free_leaf_root},
    {"ctx_freed_with_node", ctx_freed_with_node},
    {"thousand_graphs", thousand_graphs},
};

const TestSuite autograd_suite = {"autograd", tests, ARRAY_LEN(tests)};
