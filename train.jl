include("hp_parsing.jl")
include("pipeline/data_processing/data_loader.jl")
include("utils.jl")
include("pipeline/train.jl")
include("MLP_CNN/CNN.jl")
include("MLP_FNO/FNO.jl")
include("wavKAN_CNN/KAN_CNN.jl")

using Random
using Flux, CUDA, KernelAbstractions
using Optimisers
using ProgressBars
using BSON: @save
using .training: train_step
using .loaders: get_darcy_loader
using .UTILS: loss_fcn, BIC, log_csv
using .ConvNN: CNN
using .FourierNO: FNO
using .KAN_Convolution: KAN_CNN
using .hyperparams: set_hyperparams

NUM_REPETITIONS = 5

model_name = "CNN"
hparams = set_hyperparams(model_name)
batch_size = parse(Int, get(ENV, "batch_size", "32"))
learning_rate = parse(Float32, get(ENV, "LR", "1e-3"))
num_epochs = parse(Int, get(ENV, "num_epochs", "50"))
optimizer_name = get(ENV, "optimizer", "Adam")

train_loader, test_loader = get_darcy_loader(batch_size)

function create_CNN()
    return CNN(1,1) |> gpu
end

function create_FNO()
    return FNO(3,1) |> gpu
end

function create_KAN_CNN()
    encoder_wavelet_names, encoder_activations, decoder_wavelet_names, decoder_activations = hparams
    return KAN_CNN(1, 1, encoder_wavelet_names, encoder_activations, decoder_wavelet_names, decoder_activations) |> gpu
end

instantiate_model = Dict(
    "CNN" => create_CNN,
    "FNO" => create_FNO,
    "KAN_CNN" => create_KAN_CNN
)[model_name]

log_file_base = Dict(
    "CNN" => "MLP_CNN/logs/",
    "FNO" => "MLP_FNO/logs/",
    "KAN_CNN" => "wavKAN_CNN/logs/"
)[model_name]

optimizer = Dict(
    "Adam" => Optimisers.Adam(learning_rate),
    "SGD" => Optimisers.Descent(learning_rate)
)[optimizer_name]

for num in 1:NUM_REPETITIONS
    file_name = log_file_base * "repetition_" * string(num) * ".csv"
    seed = Random.seed!(num)
    model = instantiate_model()
    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model)

    train_loss = 0.0
    test_loss = 0.0

    # Create csv with header
    open(file_name, "w") do file
        write(file, "Epoch,Time (s),Train Loss,Test Loss,BIC\n")
    end

    start_time = time()
    for epoch in ProgressBar(1:num_epochs)
        model, opt_state, train_loss, test_loss = train_step(model, opt_state, train_loader, test_loader, loss_fcn, epoch)
        BIC_val = BIC(model, first(train_loader)[2], test_loss)
        time_epoch = time() - start_time
        log_csv(epoch, train_loss, test_loss, BIC_val, time_epoch, file_name)  
    end

    save_file_name = log_file_base * "trained_models/model_" * string(num) * ".bson"
    
    model = model |> cpu
    @save save_file_name model

    model = nothing
end