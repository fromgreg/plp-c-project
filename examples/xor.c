/* XOR example: train a 2-8-1 network on the four points of the XOR function.
 *
 *   xor            train for 5000 epochs
 *   xor <epochs>   train for the given number of epochs
 *
 * XOR is the smallest problem that a single linear layer cannot solve, so it
 * shows that the hidden layer, the backward pass and the optimizer work
 * together. Exit status 0 means: final loss below 0.01 and all four points on
 * the right side of 0.5. */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "autograd.h"
#include "nn.h"
#include "ops.h"
#include "optim.h"
#include "rng.h"
#include "tensor.h"
#include "util.h"

#define SEED 42
#define HIDDEN 8
/* Plain SGD has an upper limit for the step size: beyond it every step
 * overshoots further and the loss grows without bound (here from about 0.5). */
#define LEARNING_RATE 0.1
#define DEFAULT_EPOCHS 5000
#define LOG_EVERY 500
#define TARGET_LOSS 0.01

#define N_SAMPLES 4

/* One sample per row. With only four points, every epoch uses all of them
 * (a single full batch). */
static const double XOR_INPUTS[N_SAMPLES * 2] = {
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0,
};
static const double XOR_TARGETS[N_SAMPLES] = {0.0, 1.0, 1.0, 0.0};

/* Parse a positive number of epochs. Returns 1 on success, 0 if text is not
 * a valid number. */
static int parse_epochs(const char *text, long *epochs)
{
    char *end;

    errno = 0;
    long value = strtol(text, &end, 10);
    /* end == text: no digits at all. *end != '\0': junk after the number.
     * errno is set if the number does not fit into a long. */
    if (errno != 0 || end == text || *end != '\0' || value <= 0)
        return 0;
    *epochs = value;
    return 1;
}

int main(int argc, char **argv)
{
    long epochs = DEFAULT_EPOCHS;

    if (argc > 2 || (argc == 2 && !parse_epochs(argv[1], &epochs))) {
        fprintf(stderr, "usage: %s [epochs]\n", argv[0]);
        return 2;
    }

    /* ---- setup: everything created here is freed at the end of main ---- */
    Rng rng;
    rng_seed(&rng, SEED);

    Tensor *x = tensor_from_array(N_SAMPLES, 2, XOR_INPUTS);
    Tensor *y = tensor_from_array(N_SAMPLES, 1, XOR_TARGETS);

    const size_t sizes[] = {2, HIDDEN, 1};
    MLP *mlp = mlp_create(sizes, ARRAY_LEN(sizes), ACT_TANH, &rng);

    /* The optimizer keeps its own copy of the parameter list. */
    size_t n_params;
    Tensor **params = mlp_parameters(mlp, &n_params);
    Optimizer *opt = optim_sgd_create(params, n_params, LEARNING_RATE);
    free(params);

    /* ---- training ---- */
    for (long epoch = 1; epoch <= epochs; epoch++) {
        Tensor *pred = mlp_forward(mlp, x); /* builds the graph */
        Tensor *loss = ops_mse(pred, y);

        /* inf or NaN never recovers: stop with a message instead of
         * training on. */
        if (!isfinite(loss->data[0]))
            fatal("loss is %g in epoch %ld (learning rate too high?)",
                  loss->data[0], epoch);

        optim_zero_grad(opt);
        graph_backward(loss);
        optim_step(opt);

        if (epoch == 1 || epoch % LOG_EVERY == 0)
            printf("epoch %5ld  loss %.3e\n", epoch, loss->data[0]);

        /* Releases pred, loss and every intermediate in between. Neither
         * pointer may be used after this line. */
        graph_free(loss);
    }

    /* ---- result: one more forward pass with the trained weights ---- */
    Tensor *pred = mlp_forward(mlp, x);
    Tensor *loss = ops_mse(pred, y);
    double final_loss = loss->data[0];
    int n_correct = 0;

    puts("\n  x1  x2   prediction  class  target");
    for (size_t i = 0; i < N_SAMPLES; i++) {
        double p = tensor_get(pred, i, 0);
        int predicted = p > 0.5;
        int target = XOR_TARGETS[i] > 0.5;
        n_correct += (predicted == target);
        printf("  %2.0f  %2.0f   %10.4f  %5d  %6d\n",
               tensor_get(x, i, 0), tensor_get(x, i, 1), p, predicted, target);
    }
    printf("\nfinal loss %.3e, %d of %d correct\n", final_loss, n_correct, N_SAMPLES);
    graph_free(loss);

    /* ---- cleanup: the reverse of the setup ---- */
    optim_free(opt);
    mlp_free(mlp);
    tensor_free(y);
    tensor_free(x);

    return (final_loss < TARGET_LOSS && n_correct == N_SAMPLES) ? EXIT_SUCCESS : EXIT_FAILURE;
}
