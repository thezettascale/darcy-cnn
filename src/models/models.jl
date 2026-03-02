function _make_named_layers(layers::Vector)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return NamedTuple{names}(Tuple(layers))
end

include("cnn.jl")
include("fno.jl")
include("kan_cnn.jl")

create_model(cfg::CNNConfig) = CNN(cfg)
create_model(cfg::FNOConfig) = FNO(cfg)
create_model(cfg::KANCNNConfig) = KANCNN(cfg)
