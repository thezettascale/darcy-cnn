module manual_convTranspose

include("../utils.jl")

export KAN_convTranspose2D

using LinearAlgebra
using Statistics
using Flux, CUDA, KernelAbstractions, Tullio
using .UTILS: unfold

function calc_out_dims_transpose(matrix, kernel_size, stride, dilation, padding)
    n, m, n_channels, batch_size = size(matrix)
    h_out = (n - 1) * stride - 2 * padding + dilation * (kernel_size[1] - 1) + 1
    w_out = (m - 1) * stride - 2 * padding + dilation * (kernel_size[2] - 1) + 1
    return h_out, w_out, n_channels, batch_size
end

function add_padding_transpose(matrix, padding=0)
    n, m, n_channels, batch_size = size(matrix)
    matrix_padded = CUDA.zeros(n + 2 * padding, m + 2 * padding, n_channels, batch_size)
    matrix_padded[padding + 1:end - padding, padding + 1:end - padding, :, :] = matrix
    return matrix_padded
end

function KAN_convTranspose2D(matrix, kernel, kernel_size, stride=1, dilation=1, padding=0)
    h_out, w_out, n_channels, batch_size = calc_out_dims_transpose(matrix, kernel_size, stride, dilation, padding)
    matrix_out = zeros(h_out, w_out, 0, 0) |> gpu

    for channel in 1:n_channels
        conv_groups = unfold(matrix[:, :, channel:channel, :], kernel_size; stride, padding, dilation)
        h_remain, w_remain = h_out - size(conv_groups, 3), w_out - size(conv_groups, 4)
        
        # Add zero-padding to the unfolded matrix - upsampling is implicit in the kernel
        zero_padding = zeros(kernel_size[1], kernel_size[2], h_remain, w_remain, n_channels, batch_size) |> gpu
        conv_groups = cat(conv_groups, zero_padding, dims=(3, 4))
        conv_groups = reshape(conv_groups, kernel_size[1] * kernel_size[2], h_out * w_out, batch_size)

        inner_output = zeros(h_out, w_out, 0) |> gpu
        for batch in 1:batch_size
            kernel_output = reshape(kernel(conv_groups[:, :, batch]), h_out, w_out, 1)
            inner_output = cat(inner_output, kernel_output, dims=3)
        end
        inner_output = reshape(inner_output, h_out, w_out, 1, batch_size)
        matrix_out = cat(matrix_out, inner_output, dims=(3, 4))
    end
    
    return matrix_out
end

end
