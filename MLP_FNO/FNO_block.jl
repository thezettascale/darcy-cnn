module FNO_block

export FNO_hidden_block

include("./FNO_layers.jl")

using Flux, NNlib
using Flux: Conv, Chain
using .FNO_layers: SpectralConv2d, MLP

# Activation mapping
act_map = Dict(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.hardtanh,
    "sigmoid" => NNlib.hardsigmoid,
    "swish" => NNlib.hardswish,
    "gelu" => NNlib.gelu
)

struct FNO_hidden_block
    spect_conv
    mlp
    conv
    phi
end

function FNO_hidden_block(in_channels::Int, out_channels::Int)
    activation = get(ENV, "activation", "relu")
    width = parse(Int64, get(ENV, "width", "64"))
    spect_conv = SpectralConv2d(in_channels, out_channels)
    mlp = MLP(width, width, width)
    conv = Conv((1, 1), width => width)
    return FNO_hidden_block(spect_conv, mlp, conv, act_map[activation])
end

function (m::FNO_hidden_block)(x)
    x2 = m.conv(x)
    x = m.spect_conv(x)
    x = m.mlp(x)
    return m.phi(x .+ x2)
end

Flux.@layer FNO_hidden_block

end