/* Tests for ops: forward values and the graph links stored in the result.
 * Gradients are tested in test_autograd.c. */
#include "autograd.h"
#include "ops.h"
#include "tensor.h"
#include "test.h"

static void add_forward(void)
{
    double av[] = {1, 2, 3, 4, 5, 6};
    double bv[] = {10, 20, 30, 40, 50, 60};
    double expected[] = {11, 22, 33, 44, 55, 66};
    Tensor *a = tensor_from_array(2, 3, av);
    Tensor *b = tensor_from_array(2, 3, bv);
    Tensor *out = ops_add(a, b);

    CHECK(out->rows == 2 && out->cols == 3);
    for (size_t i = 0; i < ARRAY_LEN(expected); i++)
        CHECK(out->data[i] == expected[i]);

    graph_free(out); /* out is graph-owned */
    tensor_free(a);
    tensor_free(b);
}

static void mul_forward(void)
{
    double av[] = {1, 2, 3, 4, 5, 6};
    double bv[] = {2, 0.5, -1, 0, 3, 10};
    double expected[] = {2, 1, -3, 0, 15, 60};
    Tensor *a = tensor_from_array(2, 3, av);
    Tensor *b = tensor_from_array(2, 3, bv);
    Tensor *out = ops_mul(a, b);

    CHECK(out->rows == 2 && out->cols == 3);
    for (size_t i = 0; i < ARRAY_LEN(expected); i++)
        CHECK(out->data[i] == expected[i]);

    graph_free(out);
    tensor_free(a);
    tensor_free(b);
}

static void result_records_graph_links(void)
{
    Tensor *a = tensor_create_param(1, 2);
    Tensor *b = tensor_create(1, 2);
    Tensor *out = ops_mul(a, b);

    CHECK(out->kind == TENSOR_INTERMEDIATE);
    CHECK(out->n_parents == 2);
    CHECK(out->parents[0] == a && out->parents[1] == b);
    CHECK(out->backward != NULL);
    CHECK(out->grad != NULL);
    if (out->grad != NULL)
        CHECK(out->grad[0] == 0.0 && out->grad[1] == 0.0);

    graph_free(out);
    tensor_free(a);
    tensor_free(b);
}

static const TestCase tests[] = {
    {"add_forward", add_forward},
    {"mul_forward", mul_forward},
    {"result_records_graph_links", result_records_graph_links},
};

const TestSuite ops_suite = {"ops", tests, ARRAY_LEN(tests)};
