module ConvNN

export CNN

using Flux, NNlib

struct CNN
    encoder 
    decoder
end

# Activation mapping
act_map = Dict(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.hardtanh,
    "sigmoid" => NNlib.hardsigmoid,
    "swish" => NNlib.hardswish,
    "gelu" => NNlib.gelu
)

function CNN(in_channels::Int, out_channels::Int)
    activation = get(ENV, "activation", "relu")
    hidden_dim = parse(Int32, get(ENV, "hidden_dim", "64"))
    phi = act_map[activation]

    encoder = Chain(
        Conv((3, 3), in_channels => 2 * hidden_dim, phi; pad=1),
        Conv((3, 3), 2 * hidden_dim => 4 * hidden_dim, phi; pad=1),
        Conv((3, 3), 32 => 8 * hidden_dim, phi; pad=1),
    )

    decoder = Chain(
        ConvTranspose((3, 3), 8 * hidden_dim => 4 * hidden_dim, phi; pad=1),
        ConvTranspose((3, 3), 4 * hidden_dim => 2 * hidden_dim, phi; pad=1),
        ConvTranspose((3, 3), 2 * hidden_dim => hidden_dim, phi; pad=1),
        ConvTranspose((3, 3), hidden_dim => out_channels, phi; pad=1)
    )

    return CNN(encoder, decoder)
end

function (m::CNN)(x)
    return m.decoder(m.encoder(x))
end
    
Flux.@layer CNN 

end