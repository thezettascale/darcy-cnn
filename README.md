# wavKAN-conv2D
A Julia implementation of Wavelet Kolmogorov-Arnold Networks (wavKAN).  Mutli-layer Perceptron (MLP) and wavKAN implementations of the Convolutional Neural Network (CNN) and Fourier Neural Operator (FNO) are applied to learn the solution operator of the 2D Darcy Flow Problem.

## Problem and Data - 2D Darcy Flow

The 2D Darcy flow problem is a fundamental problem in fluid dynamics and porous media flow, described by the partial differential equation (PDE):

$$
-\nabla \cdot (a(x) \nabla u(x)) = f(x), \quad x \in (0,1)^2
$$

where:
- $a(x)$ is the diffusion coefficient generated from a random Fourier field.
- $f(x)$ is a constant forcing function throughout the domain.
- $u(x)$ is the solution field to be learned.

### Boundary Condition

The problem is subject to the Dirichlet boundary condition:

$$
u(x) = 0, \quad x \in \partial(0,1)^2
$$

### Objective

The goal is to learn the operator mapping the diffusion coefficient \(a\) to the solution \(u\), denoted as:

$$
a \mapsto u
$$
