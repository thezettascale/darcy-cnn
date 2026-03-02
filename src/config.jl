using ConfParser

abstract type ModelConfig end


struct CNNConfig <: ModelConfig
    hidden_dim::Int
    activation::String
    learning_rate::Float32
    step_rate::Int
    gamma::Float32
    min_lr::Float32
    batch_size::Int
    num_epochs::Int
    p::Float32
end

function CNNConfig(path::String)
    conf = ConfParse(path)
    parse_conf!(conf)
    return CNNConfig(
        parse(Int, retrieve(conf, "Architecture", "hidden_dim")),
        retrieve(conf, "Architecture", "activation"),
        parse(Float32, retrieve(conf, "Optimizer", "learning_rate")),
        parse(Int, retrieve(conf, "Optimizer", "step_rate")),
        parse(Float32, retrieve(conf, "Optimizer", "gamma")),
        parse(Float32, retrieve(conf, "Optimizer", "min_lr")),
        parse(Int, retrieve(conf, "Dataloader", "batch_size")),
        parse(Int, retrieve(conf, "Pipeline", "num_epochs")),
        parse(Float32, retrieve(conf, "Loss", "p")),
    )
end


struct FNOConfig <: ModelConfig
    width::Int
    modes1::Int
    modes2::Int
    num_blocks::Int
    activation::String
    learning_rate::Float32
    step_rate::Int
    gamma::Float32
    min_lr::Float32
    batch_size::Int
    num_epochs::Int
    p::Float32
end

function FNOConfig(path::String)
    conf = ConfParse(path)
    parse_conf!(conf)
    return FNOConfig(
        parse(Int, retrieve(conf, "Architecture", "channel_width")),
        parse(Int, retrieve(conf, "Architecture", "modes1")),
        parse(Int, retrieve(conf, "Architecture", "modes2")),
        parse(Int, retrieve(conf, "Architecture", "num_hidden_blocks")),
        retrieve(conf, "Architecture", "activation"),
        parse(Float32, retrieve(conf, "Optimizer", "learning_rate")),
        parse(Int, retrieve(conf, "Optimizer", "step_rate")),
        parse(Float32, retrieve(conf, "Optimizer", "gamma")),
        parse(Float32, retrieve(conf, "Optimizer", "min_lr")),
        parse(Int, retrieve(conf, "Dataloader", "batch_size")),
        parse(Int, retrieve(conf, "Pipeline", "num_epochs")),
        parse(Float32, retrieve(conf, "Loss", "p")),
    )
end


struct KANCNNConfig <: ModelConfig
    hidden_dim::Int
    norm::Bool
    encoder_wavelet_names::Vector{String}
    encoder_activations::Vector{String}
    decoder_wavelet_names::Vector{String}
    decoder_activations::Vector{String}
    learning_rate::Float32
    step_rate::Int
    gamma::Float32
    min_lr::Float32
    batch_size::Int
    num_epochs::Int
    p::Float32
end

function KANCNNConfig(path::String)
    conf = ConfParse(path)
    parse_conf!(conf)
    enc_wav_keys = ["wav_one", "wav_two", "wav_three"]
    enc_act_keys = ["act_one", "act_two", "act_three"]
    dec_wav_keys = ["wav_one", "wav_two", "wav_three", "wav_four"]
    dec_act_keys = ["act_one", "act_two", "act_three", "act_four"]
    return KANCNNConfig(
        parse(Int, retrieve(conf, "Architecture", "hidden_dim")),
        parse(Bool, retrieve(conf, "Architecture", "norm")),
        [retrieve(conf, "EncoderWavelets", k) for k in enc_wav_keys],
        [retrieve(conf, "EncoderActivations", k) for k in enc_act_keys],
        [retrieve(conf, "DecoderWavelets", k) for k in dec_wav_keys],
        [retrieve(conf, "DecoderActivations", k) for k in dec_act_keys],
        parse(Float32, retrieve(conf, "Optimizer", "learning_rate")),
        parse(Int, retrieve(conf, "Optimizer", "step_rate")),
        parse(Float32, retrieve(conf, "Optimizer", "gamma")),
        parse(Float32, retrieve(conf, "Optimizer", "min_lr")),
        parse(Int, retrieve(conf, "Dataloader", "batch_size")),
        parse(Int, retrieve(conf, "Pipeline", "num_epochs")),
        parse(Float32, retrieve(conf, "Loss", "p")),
    )
end


function load_config(model_name::String)
    configs = Dict(
        "CNN" => () -> CNNConfig("config/cnn.ini"),
        "FNO" => () -> FNOConfig("config/fno.ini"),
        "KAN_CNN" => () -> KANCNNConfig("config/kan_cnn.ini"),
    )
    return configs[model_name]()
end
