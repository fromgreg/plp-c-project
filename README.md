# cgrad

A tiny neural network library with reverse-mode automatic differentiation, written from scratch in C11.

> This repository is the development repo for my course project in *Programming Languages and Paradigms* (University of Zurich, Fall 2026). The submitted project proposal is in [`proposal/`](proposal/) ([PDF](proposal/project-proposal_gregory-frommelt_c.pdf), [LaTeX source](proposal/project.tex)).

## Goal

A from-scratch, proof-of-concept deep-learning engine in plain C. It uses only the C standard library and `libm`, with no ML or linear-algebra libraries:

- 2D tensor type and a dynamically built computational graph
- Reverse-mode autodiff (topological sort and backward pass)
- Core ops: add, mul, matmul, sum, ReLU/tanh/sigmoid, softmax + cross-entropy, MSE
- A multi-layer perceptron trained with SGD
- Training on **XOR**, then on **MNIST** (IDX format loaded by hand)
- Gradient checks against finite differences
- No leaks, no undefined behaviour: clean under AddressSanitizer/UBSan and macOS `leaks`

## Toolchain

- C11 (`-std=c11`), Apple clang on macOS (Apple Silicon)
- Plain `Makefile`
- Warnings: `-Wall -Wextra -Wpedantic` (`-Werror` in debug builds)
- Checks: `-fsanitize=address,undefined` and `leaks --atExit`

Setup, build and run instructions will be added here as the implementation progresses.

## License

Released under the [MIT License](LICENSE).
