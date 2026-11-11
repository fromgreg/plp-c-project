/* Tests for tensor: lifetime, fills, row-major access, printing, matmul. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "tensor.h"
#include "test.h"

/* 1 if every element of t equals value exactly. */
static int all_equal(const Tensor *t, double value)
{
    for (size_t i = 0; i < tensor_size(t); i++)
        if (t->data[i] != value)
            return 0;
    return 1;
}

static void create_shape_and_zeroed(void)
{
    Tensor *t = tensor_create(3, 4);
    CHECK(t->rows == 3 && t->cols == 4);
    CHECK(tensor_size(t) == 12);
    CHECK(all_equal(t, 0.0));
    tensor_free(t);
}

static void free_null_is_noop(void)
{
    tensor_free(NULL);
    CHECK(1); /* reaching this line is the test */
}

static void from_array_copies(void)
{
    double values[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_array(2, 3, values);
    values[0] = 99; /* must not affect t: it owns a copy */
    CHECK(tensor_get(t, 0, 0) == 1);
    CHECK(tensor_get(t, 1, 2) == 6);
    tensor_free(t);
}

static void copy_is_deep(void)
{
    double values[] = {1, 2, 3, 4};
    Tensor *a = tensor_from_array(2, 2, values);
    Tensor *b = tensor_copy(a);
    CHECK(b->rows == 2 && b->cols == 2);
    CHECK(b->data != a->data);
    tensor_set(a, 0, 0, -1);
    CHECK(tensor_get(b, 0, 0) == 1);
    tensor_free(a);
    tensor_free(b);
}

static void fill_and_zeros(void)
{
    Tensor *t = tensor_create(2, 5);
    tensor_fill(t, 3.5);
    CHECK(all_equal(t, 3.5));
    tensor_zeros(t);
    CHECK(all_equal(t, 0.0));
    tensor_free(t);
}

static void row_major_layout(void)
{
    Tensor *t = tensor_create(2, 3);
    tensor_set(t, 1, 2, 7.0);
    CHECK(t->data[1 * 3 + 2] == 7.0); /* r * cols + c */
    tensor_set(t, 0, 1, 4.0);
    CHECK(t->data[1] == 4.0);
    CHECK(tensor_get(t, 1, 2) == 7.0);
    tensor_free(t);
}

static void print_format(void)
{
    double values[] = {1, -2.5, 0.125, 1000};
    Tensor *t = tensor_from_array(2, 2, values);
    FILE *f = tmpfile();
    CHECK(f != NULL);
    if (f == NULL) {
        tensor_free(t);
        return;
    }

    tensor_print(f, t, "x");
    rewind(f);
    char buf[256] = {0};
    size_t len = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);

    const char *expected = "x [2x2]\n"
                           "         1      -2.5\n"
                           "     0.125      1000\n";
    CHECK(len == strlen(expected));
    CHECK(strcmp(buf, expected) == 0);
    tensor_free(t);
}

static void matmul_known_values(void)
{
    /* [1 2 3]   [ 7  8]   [ 58  64]
     * [4 5 6] * [ 9 10] = [139 154]
     *           [11 12]                */
    double av[] = {1, 2, 3, 4, 5, 6};
    double bv[] = {7, 8, 9, 10, 11, 12};
    Tensor *a = tensor_from_array(2, 3, av);
    Tensor *b = tensor_from_array(3, 2, bv);
    Tensor *c = tensor_matmul(a, b);

    CHECK(c->rows == 2 && c->cols == 2);
    CHECK(tensor_get(c, 0, 0) == 58);
    CHECK(tensor_get(c, 0, 1) == 64);
    CHECK(tensor_get(c, 1, 0) == 139);
    CHECK(tensor_get(c, 1, 1) == 154);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

static void matmul_identity(void)
{
    Rng rng;
    rng_seed(&rng, 3);
    Tensor *a = tensor_create(3, 4);
    tensor_fill_uniform(a, &rng, -1.0, 1.0);
    Tensor *id = tensor_create(4, 4);
    for (size_t i = 0; i < 4; i++)
        tensor_set(id, i, i, 1.0);

    Tensor *c = tensor_matmul(a, id);
    CHECK(c->rows == 3 && c->cols == 4);
    /* Exact: each sum is one product with 1.0 plus zeros. */
    CHECK(memcmp(c->data, a->data, tensor_size(a) * sizeof *a->data) == 0);
    tensor_free(a);
    tensor_free(id);
    tensor_free(c);
}

static void matmul_row_times_column(void)
{
    double rv[] = {1, 2, 3};
    double cv[] = {4, 5, 6};
    Tensor *row = tensor_from_array(1, 3, rv);
    Tensor *col = tensor_from_array(3, 1, cv);
    Tensor *dot = tensor_matmul(row, col);
    CHECK(dot->rows == 1 && dot->cols == 1);
    CHECK(tensor_get(dot, 0, 0) == 32); /* 4 + 10 + 18 */
    tensor_free(row);
    tensor_free(col);
    tensor_free(dot);
}

static void fill_uniform_range(void)
{
    Rng rng;
    rng_seed(&rng, 11);
    Tensor *t = tensor_create(50, 50);
    tensor_fill_uniform(t, &rng, -2.0, 3.0);
    int in_range = 1;
    for (size_t i = 0; i < tensor_size(t); i++)
        in_range = in_range && t->data[i] >= -2.0 && t->data[i] < 3.0;
    CHECK(in_range);
    tensor_free(t);
}

static void fill_normal_statistics(void)
{
    Rng rng;
    rng_seed(&rng, 5);
    Tensor *t = tensor_create(200, 200);
    tensor_fill_normal(t, &rng, 0.0, 0.5);

    size_t n = tensor_size(t);
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += t->data[i];
        sum_sq += t->data[i] * t->data[i];
    }
    double mean = sum / (double)n;
    double std = sqrt(sum_sq / (double)n - mean * mean);
    CHECK_NEAR(mean, 0.0, 0.01);
    CHECK_NEAR(std, 0.5, 0.01);
    tensor_free(t);
}

static const TestCase tests[] = {
    {"create_shape_and_zeroed", create_shape_and_zeroed},
    {"free_null_is_noop", free_null_is_noop},
    {"from_array_copies", from_array_copies},
    {"copy_is_deep", copy_is_deep},
    {"fill_and_zeros", fill_and_zeros},
    {"row_major_layout", row_major_layout},
    {"print_format", print_format},
    {"matmul_known_values", matmul_known_values},
    {"matmul_identity", matmul_identity},
    {"matmul_row_times_column", matmul_row_times_column},
    {"fill_uniform_range", fill_uniform_range},
    {"fill_normal_statistics", fill_normal_statistics},
};

const TestSuite tensor_suite = {"tensor", tests, ARRAY_LEN(tests)};
