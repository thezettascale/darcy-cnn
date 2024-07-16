include("../pipeline/data_processing/data_loader.jl")
include("CNN.jl")
include("../utils.jl")
include("../pipeline/train.jl")
include("../hp_parsing.jl")

using HyperTuning
using ConfParser
using Random
using Flux, CUDA, KernelAbstractions
using Optimisers
using .training: train_step
using .ConvNN: CNN
using .loaders: get_darcy_loader
using .UTILS: loss_fcn
using .hyperparams: set_hyperparams

function objective(trial)
    seed = get_seed(trial)
    Random.seed!(seed)

    @suggest step_rate in trial
    @suggest gamma in trial
    @suggest learning_rate in trial
    @suggest min_lr in trial
    @suggest activation in trial
    @suggest hidden_dim in trial
    @suggest b_size in trial

    ENV["p"] = "2.0"
    ENV["step"] = step_rate
    ENV["decay"] = gamma
    ENV["LR"] = learning_rate
    ENV["min_LR"] = min_lr
    ENV["activation"] = activation
    ENV["hidden_dim"] = hidden_dim
    ENV["batch_size"] = b_size

    num_epochs = 100

    train_loader, test_loader = get_darcy_loader(b_size)

    model = CNN(1, 1) |> gpu

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

space = Scenario(
    step_rate = 10:40,
    gamma = (0.5..0.9),
    learning_rate = (1e-4..1e-2),
    min_lr = (1e-6..1e-2),
    activation = ["relu", "selu", "leakyrelu", "swish", "gelu"],
    hidden_dim = 2:100,
    b_size = 1:30,
    verbose = true,
    max_trials = 100,
    pruner = MedianPruner(),
)

HyperTuning.optimize(objective, space)

display(top_parameters(space))

# Save the best configuration
@unpack step_rate, gamma, learning_rate, min_lr, activation, hidden_dim, b_size = space

conf = ConfParse("MLP_CNN/CNN_config.ini")
parse_conf!(conf)

commit!(conf, "Loss", "p", "2.0")
commit!(conf, "Optimizer", "step_rate", string(step_rate))
commit!(conf, "Optimizer", "gamma", string(gamma))
commit!(conf, "Optimizer", "learning_rate", string(learning_rate))
commit!(conf, "Optimizer", "min_lr", string(min_lr))
commit!(conf, "Architecture", "activation", activation)
commit!(conf, "Architecture", "hidden_dim", string(hidden_dim))
commit!(conf, "Dataloader", "batch_size", string(b_size))
commit!(conf, "Optimizer", "type", "Adam")

save!(conf, "MLP_CNN/CNN_config.ini")