# wavKAN-conv2D
A Julia implementation of Wavelet Kolmogorov-Arnold Networks (wavKAN) for Convolutional Neural Networks.  Mutli-layer Perceptron (MLP) and wavKAN implementations of the Convolutional Neural Network (CNN) and Fourier Neural Operator (FNO) are applied to learn the solution operator of the 2D Darcy Flow Problem.

The MLP models were developed in a [previous side project](https://github.com/PritRaj1/Neural-Operator-Learning). The commit history attributed to their development can be found there. The Dense KAN layer was also developed in a [previous project](https://github.com/PritRaj1/Julia-Wav-KAN).

## To Run

1. Get dependencies:

```bash
julia requirements.jl
```

2. Tune hyperparameters:

```bash
julia src/models/MLP_CNN/hyperparameter_tuning.jl
```

```bash
julia src/models/MLP_FNO/hyperparameter_tuning.jl
```
    
```bash
julia src/models/wavKAN_CNN/hyperparameter_tuning.jl
```

3. (Alternatively to 2) Manually configure hyperparameters in the respective `config.ini` files.
    - [Vanilla RNO](https://github.com/PritRaj1/wavKAN-conv2D/blob/main/src/models/MLP_CNN/CNN_config.ini)
    - [wavKAN RNO](https://github.com/PritRaj1/wavKAN-conv2D/blob/main/src/models/MLP_FNO/FNO_config.ini)
    - [Vanilla Transformer](https://github.com/PritRaj1/wavKAN-conv2D/blob/main/src/models/wavKAN_CNN/KAN_CNN_config.ini)

4. Train the models, (model_name variable is set on line 26), and log the results:

```bash
julia train.jl
```

5. Compare the training loops:
  
```bash
python results.py
```

6. Visualize the results:

```bash
julia predict_flow.jl
```

## Results

The wavelet KAN CNN model with a hidden dimension of only 5 outperformed an MLP CNN with a hidden dimension of 90 in terms of generalisation capacity, (despite achieving a higher train loss). The 2D Darcy Flow dataset is a scientific problem that often incurs overfitting, therefore wavelet KAN seems superior to the MLP in this context. It is more proficient at learning the true solution operator of the problem, and generalises well to unseen data.

However, it did not surpass the MLP FNO, which is expected given the FNO's suitability for predicting 2D Darcy Flow. A wavelet KAN FNO was not implemented in this comparison - it would be interesting to see how it compares against the MLP FNO.

### Predictions
<p align="center">
  <img src="figures/MLP CNN_prediction.gif" alt="CNN Predicted Darcy Flow" width="32%" style="padding-right: 20px;">
  <img src="figures/MLP FNO_prediction.gif" alt="FNO Predicted Darcy Flow" width="32%" style="padding-right: 20px;">
  <img src="figures/KAN CNN_prediction.gif" alt="KAN CNN Predicted Darcy Flow" width="32%">
</p>

### Error Fields

<p align="center">
  <img src="figures/MLP CNN_error.gif" alt="CNN Error Field" width="32%" style="padding-right: 20px;">
  <img src="figures/MLP FNO_error.gif" alt="FNO Error Field" width="32%" style="padding-right: 20px;">
  <img src="figures/KAN CNN_error.gif" alt="KAN CNN Error Field" width="32%">
</p>

### Predictive Power and Consistency


| Model   | Train Loss        | Test Loss       | BIC                 | Time (mins)    | Param Count   |
|---------|-------------------|-----------------|---------------------|----------------|---------------|
| MLP CNN | 5115.36 ± 1635.76 | 1507.95 ± 36.70 | 16951640.94 ± 73.39 | 5.24 ± 0.12    | 5,982,121     |
| MLP FNO | 34.12 ± 32.05     | 5.03 ± 3.14     | 21304858.36 ± 6.29  | 2.62 ± 0.43    | 4,667,665     |
| KAN CNN | 6065.64 ± 502.48  | 612.37 ± 50.11  | 152794.31 ± 100.22  | 241.29 ± 45.38 | 35,919        |    |

### TODO - Plot FLOPs comparison

Training time was recorded for each of the models, but this is not considered a reliable estimate of the computational cost of the models, given that they were not run on the same hardware, and multiple tasks were running on the same machine. The number of FLOPs for each model will be calculated and compared in the future, once GFlops is updated to work with the latest Julia version.

## Wavelets

<p align="center">
  <img src="src/waveletKAN/wavelets/animations/DerivativeOfGaussian.gif" width="30%" />
    <img src="src/waveletKAN/wavelets/animations/MexicanHat.gif" width="30%" />
    <img src="src/waveletKAN/wavelets/animations/Meyer.gif" width="30%" />
    <img src="src/waveletKAN/wavelets/animations/Morlet.gif" width="30%" />
    <img src="src/waveletKAN/wavelets/animations/Shannon.gif" width="30%" />
</p>

## Message from author:

Wow! I was not expecting the wavelet-KAN CNN to outperform the MLP CNN to this extent, especially considering how it struggled for this [sequence modelling task](https://github.com/exa-laboratories/Julia-Wav-KAN). Generally, tuning parameters for the wavelet KAN was the most difficult aspect, and I brute forced it quite unintelligently with HyperTuning.jl. I personally still think the spline-based KANs hold the edge over wavelets because of their capacities for symbolic regression - I imagine that it's much more difficult to fit arbritrary functions to wavelets than the functions that splines manifest. Besides, the wavelets are built from simpler functions that a spline-based KAN model can represent, (although it's more parameter efficient to just use choose wavelets as your base univariate function than train an entire KAN model to approximate a wavelet for use as a smaller subset of a larger network).

But damn, KANs are hella exciting!

## Problem and Data - 2D Darcy Flow

The dataset has been sourced from the University of Cambridge Engineering Department's Part IIB course on [Data-Driven and Learning-Based Methods in Mechanics and Materials.](https://teaching.eng.cam.ac.uk/content/engineering-tripos-part-iib-4c11-data-driven-and-learning-based-methods-mechanics-and)

The 2D Darcy flow equation on the unit box is a problem in fluid dynamics and porous media flow, described by the partial differential equation (PDE):

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

## References

- [Bozorgasl, Z., & Chen, H. (2024). Wav-KAN: Wavelet Kolmogorov-Arnold Networks.](https://arxiv.org/abs/2405.12832)
- [Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2024). KAN: Kolmogorov-Arnold Networks.](https://arxiv.org/abs/2404.19756)
- [Mejade Dios, J.-A., Mezura-Montes, E., & Quiroz-Castellanos, M. (2021). Automated parameter tuning as a bilevel optimization problem solved by a surrogate-assisted population-based approach.](https://doi.org/10.1007/s10489-020-02151-y)
- [Liu, B., Cicirello, A. (2024). Cambridge University Engineering Department Part IIB Course on Data-Driven and Learning-Based Methods in Mechanics and Materials.](https://teaching.eng.cam.ac.uk/content/engineering-tripos-part-iib-4c11-data-driven-and-learning-based-methods-mechanics-and)
- [Detkov, N. (2020). Implementation of the generalized 2D convolution with dilation from scratch in Python and NumPy.](https://github.com/detkov/Convolution-From-Scratch/)
