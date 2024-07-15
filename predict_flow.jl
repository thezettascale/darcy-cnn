include("pipeline/data_processing/data_loader.jl")
include("MLP_CNN/CNN.jl")
include("MLP_FNO/FNO.jl")

using Plots; pythonplot()
using Flux
using BSON: @load
using CUDA, KernelAbstractions
using .loaders: get_darcy_loader

train_loader, test_loader = get_darcy_loader(1)

MODEL_NAME = "FNO"

model_file = Dict(
    "CNN" => "MLP_CNN/logs/trained_models/model_1.bson",
    "FNO" => "MLP_FNO/logs/trained_models/model_1.bson",
    "KAN_CNN" => "wavKAN_CNN/logs/trained_models/model_1.bson"
)[MODEL_NAME]

# Load the model
@load model_file model

# Move the model to the GPU
model = gpu(model)

# Plot the prediction
X, Y = [x for x in range(0, stop=1, length=32)], [y for y in range(0, stop=1, length=32)]

anim = @animate for (a, u) in test_loader
    u_pred = model(a) |> cpu
    u_pred = u_pred[:, :, 1, 1]
    contourf(X, Y, u_pred, title="$MODEL_NAME Prediction", cbar=false, color=:viridis)
end

# Save the animation to file
gif(anim, "figures/$MODEL_NAME" * "_prediction.gif", fps=5)