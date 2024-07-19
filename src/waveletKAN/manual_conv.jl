module manual_conv

include("../utils.jl")

export KAN_conv2D

using LinearAlgebra
using Statistics
using Flux, CUDA, KernelAbstractions, Tullio
using .UTILS: unfold

function KAN_conv2D(matrix, kernel, kernel_size, stride=1, dilation=1, padding=0)
    n_channels, batch_size = size(matrix, 3), size(matrix, 4)
    patches = unfold(matrix, kernel_size; stride=stride, padding=padding, dilation=dilation)
    h, w = size(patches, 3), size(patches, 4)
    patches = reshape(patches, prod(kernel_size) * n_channels, h * w, batch_size)
    out = kernel(patches)
    return reshape(out, h, w, size(out, 1), batch_size)
    end
end