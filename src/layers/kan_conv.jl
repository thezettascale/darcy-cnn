struct KANConv2D{K} <: Lux.AbstractLuxContainerLayer{(:dense_kernel,)}
    dense_kernel::K
    kernel_size::Tuple{Int, Int}
    stride::Int
    dilation::Int
    padding::Int
end

function KANConv2D(
        in_channels::Int, out_channels::Int, kernel_size::Tuple{Int, Int},
        wavelet_name::String, base_activation::String;
        stride::Int = 1, dilation::Int = 1, padding::Int = 1, norm::Bool = false
    )
    dense_kernel = KANdense(
        prod(kernel_size) * in_channels, out_channels,
        wavelet_name, base_activation;
        norm = norm, is_2d = true,
    )
    return KANConv2D(dense_kernel, kernel_size, stride, dilation, padding)
end

function (c::KANConv2D)(x, ps, st)
    n_channels = size(x, 3)
    batch_size = size(x, 4)

    patches = unfold(
        x, c.kernel_size;
        stride = c.stride, padding = c.padding, dilation = c.dilation,
    )
    h, w = size(patches, 3), size(patches, 4)
    patches = reshape(patches, prod(c.kernel_size) * n_channels, h * w, batch_size)

    out, st_k = c.dense_kernel(patches, ps.dense_kernel, st.dense_kernel)
    out_channels = size(out, 1)
    out = reshape(out, out_channels, h, w, batch_size)
    out = permutedims(out, (2, 3, 1, 4))

    return out, (dense_kernel = st_k,)
end


struct KANConvTranspose2D{K} <: Lux.AbstractLuxContainerLayer{(:dense_kernel,)}
    dense_kernel::K
    kernel_size::Tuple{Int, Int}
    stride::Int
    dilation::Int
    padding::Int
end

function KANConvTranspose2D(
        in_channels::Int, out_channels::Int, kernel_size::Tuple{Int, Int},
        wavelet_name::String, base_activation::String;
        stride::Int = 1, dilation::Int = 1, padding::Int = 1, norm::Bool = false
    )
    dense_kernel = KANdense(
        prod(kernel_size) * in_channels, out_channels,
        wavelet_name, base_activation;
        norm = norm, is_2d = true,
    )
    return KANConvTranspose2D(dense_kernel, kernel_size, stride, dilation, padding)
end

function (c::KANConvTranspose2D)(x, ps, st)
    n_channels = size(x, 3)
    batch_size = size(x, 4)

    x_up = NNlib.upsample_nearest(x, (c.stride, c.stride))
    patches = unfold(
        x_up, c.kernel_size;
        stride = c.stride, padding = c.padding, dilation = c.dilation,
    )
    h, w = size(patches, 3), size(patches, 4)
    patches = reshape(patches, prod(c.kernel_size) * n_channels, h * w, batch_size)

    out, st_k = c.dense_kernel(patches, ps.dense_kernel, st.dense_kernel)
    out_channels = size(out, 1)
    out = reshape(out, out_channels, h, w, batch_size)
    out = permutedims(out, (2, 3, 1, 4))

    return out, (dense_kernel = st_k,)
end
