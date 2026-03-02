module WavKANConv

using Lux
using Lux: Training
using NNlib
using Optimisers
using Random: AbstractRNG
using Lux.Training: AutoEnzyme
using Reactant: @compile
using FFTW
using AbstractFFTs: rfft, irfft

include("utils.jl")
include("config.jl")
include("wavelets/wavelets.jl")
include("layers/layers.jl")
include("models/models.jl")
include("pipeline/pipeline.jl")

export load_config, create_model, get_darcy_loader, train_epoch
export loss_fcn, BIC, log_csv
export MinMaxNormaliser, UnitGaussianNormaliser, encode, decode
export CNNConfig, FNOConfig, KANCNNConfig
export AutoEnzyme

end
