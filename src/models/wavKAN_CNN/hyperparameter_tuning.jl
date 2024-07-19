include("../../pipeline/data_processing/data_loader.jl")
include("KAN_CNN.jl")
include("../../utils.jl")
include("../../pipeline/train.jl")

using HyperTuning
using ConfParser
using Random
using Flux, CUDA, KernelAbstractions
using Optimisers
using .training: train_step
using .KAN_Convolution: KAN_CNN
using .loaders: get_darcy_loader
using .UTILS: loss_fcn

function objective(trial)
    seed = get_seed(trial)
    Random.seed!(seed)

    @suggest step_rate in trial
    @suggest gamma in trial
    @suggest learning_rate in trial
    @suggest min_lr in trial
    @suggest hidden_dim in trial
    @suggest b_size in trial
    @suggest encoder_wav_one in trial
    @suggest encoder_wav_two in trial
    @suggest encoder_wav_three in trial
    @suggest encoder_activation_one in trial
    @suggest encoder_activation_two in trial
    @suggest encoder_activation_three in trial
    @suggest decoder_wav_one in trial
    @suggest decoder_wav_two in trial
    @suggest decoder_wav_three in trial
    @suggest decoder_wav_four in trial
    @suggest decoder_activation_one in trial
    @suggest decoder_activation_two in trial
    @suggest decoder_activation_three in trial
    @suggest decoder_activation_four in trial
    @suggest norm in trial

    ENV["p"] = "2.0"
    ENV["step"] = step_rate
    ENV["decay"] = gamma
    ENV["LR"] = learning_rate
    ENV["min_LR"] = min_lr
    ENV["hidden_dim"] = hidden_dim
    ENV["batch_size"] = b_size
    ENV["norm"] = norm

    num_epochs = 30

    train_loader, test_loader = get_darcy_loader(b_size)

    encoder_wavelet_names = [encoder_wav_one, encoder_wav_two, encoder_wav_three]
    encoder_activations = [encoder_activation_one, encoder_activation_two, encoder_activation_three]
    decoder_wavelet_names = [decoder_wav_one, decoder_wav_two, decoder_wav_three, decoder_wav_four]
    decoder_activations = [decoder_activation_one, decoder_activation_two, decoder_activation_three, decoder_activation_four]

    model = KAN_CNN(1, 1, encoder_wavelet_names, encoder_activations, decoder_wavelet_names, decoder_activations) |> gpu

    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model)

    train_loss = 0.0
    test_loss = 0.0

    for epoch in 1:num_epochs
        model, opt_state, train_loss, test_loss = train_step(model, opt_state, train_loader, test_loader, loss_fcn, epoch)
        report_value!(trial, test_loss)
        should_prune(trial) && (return)
    end

    model = nothing
    train_loader = nothing
    test_loader = nothing
    GC.gc(true) 
    CUDA.reclaim()

    test_loss < 100 && report_success!(trial)
    return test_loss
end

activation_list = ["relu", "selu", "leakyrelu", "swish", "gelu"]
wavelet_list = ["MexicanHat", "DerivativeOfGaussian", "Morlet", "Shannon", "Meyer"]

space = Scenario(
    step_rate = 10:40,
    gamma = (0.5..0.9),
    learning_rate = (1e-4..1e-2),
    min_lr = (1e-6..1e-2),
    hidden_dim = 2:10,
    b_size = 1:15,
    encoder_wav_one = wavelet_list,
    encoder_wav_two = wavelet_list,
    encoder_wav_three = wavelet_list,
    encoder_activation_one = activation_list,
    encoder_activation_two = activation_list,
    encoder_activation_three = activation_list,
    decoder_wav_one = wavelet_list,
    decoder_wav_two = wavelet_list,
    decoder_wav_three = wavelet_list,
    decoder_wav_four = wavelet_list,
    decoder_activation_one = activation_list,
    decoder_activation_two = activation_list,
    decoder_activation_three = activation_list,
    decoder_activation_four = activation_list,
    norm = [true, false],
    verbose = true,
    max_trials = 100,
    pruner = MedianPruner(),
)

HyperTuning.optimize(objective, space)

display(top_parameters(space))

# Save the best configuration
@unpack step_rate, gamma, learning_rate, min_lr, hidden_dim, b_size, encoder_wav_one, encoder_wav_two, encoder_wav_three, encoder_activation_one, encoder_activation_two, encoder_activation_three, decoder_wav_one, decoder_wav_two, decoder_wav_three, decoder_wav_four, decoder_activation_one, decoder_activation_two, decoder_activation_three, decoder_activation_four, norm = space

conf = ConfParse("src/models/wavKAN_CNN/KAN_CNN_config.ini")
parse_conf!(conf)

commit!(conf, "Loss", "p", "2.0")
commit!(conf, "Optimizer", "step_rate", string(step_rate))
commit!(conf, "Optimizer", "gamma", string(gamma))
commit!(conf, "Optimizer", "learning_rate", string(learning_rate))
commit!(conf, "Optimizer", "min_lr", string(min_lr))
commit!(conf, "Architecture", "hidden_dim", string(hidden_dim))
commit!(conf, "Dataloader", "batch_size", string(b_size))
commit!(conf, "Optimizer", "type", "Adam")
commit!(conf, "EncoderWavelets", "wav_one", encoder_wav_one)
commit!(conf, "EncoderWavelets", "wav_two", encoder_wav_two)
commit!(conf, "EncoderWavelets", "wav_three", encoder_wav_three)
commit!(conf, "EncoderActivations", "act_one", encoder_activation_one)
commit!(conf, "EncoderActivations", "act_two", encoder_activation_two)
commit!(conf, "EncoderActivations", "act_three", encoder_activation_three)
commit!(conf, "DecoderWavelets", "wav_one", decoder_wav_one)
commit!(conf, "DecoderWavelets", "wav_two", decoder_wav_two)
commit!(conf, "DecoderWavelets", "wav_three", decoder_wav_three)
commit!(conf, "DecoderWavelets", "wav_four", decoder_wav_four)
commit!(conf, "DecoderActivations", "act_one", decoder_activation_one)
commit!(conf, "DecoderActivations", "act_two", decoder_activation_two)
commit!(conf, "DecoderActivations", "act_three", decoder_activation_three)
commit!(conf, "DecoderActivations", "act_four", decoder_activation_four)
commit!(conf, "Architecture", "norm", string(norm))

save!(conf, "src/models/wavKAN_CNN/KAN_CNN_config.ini")