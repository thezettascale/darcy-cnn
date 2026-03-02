include("src/WavKANConv.jl")

using .WavKANConv
using Lux
using Reactant
using JLD2
using Plots; pythonplot()
using MLDataDevices: reactant_device, cpu_device

model_name = length(ARGS) >= 1 ? ARGS[1] : "KAN_CNN"
cfg = load_config(model_name)
dev = reactant_device()

train_loader, test_loader = get_darcy_loader(1; dev = dev)

model = create_model(cfg)
rng = Lux.default_rng()
_, st = Lux.setup(rng, model)

model_file = joinpath("logs", model_name, "trained_models", "model_1.jld2")
ps = JLD2.load(model_file, "ps") |> dev
st = st |> dev

a_first, u_first = first(test_loader)
model_compiled = @compile model(a_first, ps, Lux.testmode(st))
u_pred, _ = model_compiled(a_first, ps, Lux.testmode(st))
u_pred = copy(u_pred) |> cpu_device()

X, Y = range(0, stop = 1, length = 32), range(0, stop = 1, length = 32)

mkpath("figures")

# Prediction animation
anim = @animate for (a, u) in test_loader
    pred, _ = model_compiled(a, ps, Lux.testmode(st))
    pred = copy(pred) |> cpu_device()
    contourf(
        X, Y, pred[:, :, 1, 1];
        title = "$model_name Prediction",
        cbar = false, color = :viridis, aspect_ratio = :equal,
    )
end
gif(anim, "figures/$(model_name)_prediction.gif"; fps = 5)

# Error field animation
anim = @animate for (a, u) in test_loader
    pred, _ = model_compiled(a, ps, Lux.testmode(st))
    pred = copy(pred) |> cpu_device()
    u_cpu = copy(u) |> cpu_device()
    contourf(
        X, Y, pred[:, :, 1, 1] .- u_cpu[:, :, 1, 1];
        title = "$model_name Error Field",
        cbar = true, color = :viridis, aspect_ratio = :equal,
        clim = (0, 1), lims = (0, 1),
    )
end
gif(anim, "figures/$(model_name)_error.gif"; fps = 5)
