# Wav-KAN Conv2D

MLP and Wavelet KAN implementations of the CNN and Fourier Neural Operator (FNO) in Julia, applied to the 2D Darcy Flow problem (diffusion coefficient to solution field mapping).

Built with [Lux.jl](https://github.com/LuxDL/Lux.jl) and [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl).

## Usage

```bash
# Train
julia --project=. main.jl CNN          # or FNO, KAN_CNN

# Tune hyperparameters
julia --project=. tune.jl CNN

# Predict + visualise
julia --project=. predict.jl KAN_CNN

# Compare models
julia --project=. compare.jl
```

Configs live in `config/` as `.ini` files.

## Problem

Dataset from the Cambridge Engineering Part IIB course on [Data-Driven Methods in Mechanics and Materials](https://teaching.eng.cam.ac.uk/content/engineering-tripos-part-iib-4c11-data-driven-and-learning-based-methods-mechanics-and).

The 2D Darcy flow equation on the unit box describes flow through porous media:

$$-\nabla \cdot (a(x) \nabla u(x)) = f(x), \quad x \in (0,1)^2$$

where $a(x)$ is a random diffusion coefficient, $f(x)$ is a constant forcing function, and $u(x)$ is the solution field. The objective is to learn the operator $a \mapsto u$. The FNO handles this naturally, but the wavelet KAN CNN generalises better than the MLP CNN despite far fewer parameters.

<table align="center">
<tr>
<td align="center"><img src="figures/MLP CNN_prediction.gif" width="100%"></td>
<td align="center"><img src="figures/MLP FNO_prediction.gif" width="100%"></td>
<td align="center"><img src="figures/KAN CNN_prediction.gif" width="100%"></td>
</tr>
<tr>
<td align="center"><em>MLP CNN. 5,982,121 params</em></td>
<td align="center"><em>MLP FNO. 4,667,665 params</em></td>
<td align="center"><em>wavKAN CNN. 35,919 params</em></td>
</tr>
</table>

## References

- [Bozorgasl & Chen (2024). Wav-KAN: Wavelet Kolmogorov-Arnold Networks.](https://arxiv.org/abs/2405.12832)
- [Liu et al. (2024). KAN: Kolmogorov-Arnold Networks.](https://arxiv.org/abs/2404.19756)
- [Liu & Cicirello (2024). Cambridge 4C11 Course.](https://teaching.eng.cam.ac.uk/content/engineering-tripos-part-iib-4c11-data-driven-and-learning-based-methods-mechanics-and)
- [Detkov (2020). 2D Convolution from Scratch.](https://github.com/detkov/Convolution-From-Scratch/)
