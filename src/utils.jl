using Statistics

const ACTIVATION_MAP = Dict{String, Function}(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.hardtanh,
    "sigmoid" => NNlib.hardsigmoid,
    "swish" => NNlib.hardswish,
    "gelu" => NNlib.gelu,
    "selu" => NNlib.selu,
)

get_activation(name::AbstractString) = ACTIVATION_MAP[name]

batch_mul(x, y) = x .* y
three_mul(x, y, z) = x .* y .* z

function node_mul(y::AbstractArray{T, 3}, w::AbstractMatrix{T}) where {T}
    output = reshape(w, size(w, 1), size(w, 2), 1) .* y
    return reshape(sum(output; dims = 1), size(w, 2), size(y, 3))
end
function node_mul(y::AbstractArray{T, 4}, w::AbstractMatrix{T}) where {T}
    output = reshape(w, size(w, 1), size(w, 2), 1, 1) .* y
    return reshape(sum(output; dims = 1), size(w, 2), size(y, 3), size(y, 4))
end

const NORM_EPS = Float32(1.0e-5)

struct UnitGaussianNormaliser{T <: AbstractFloat}
    mu::T
    sigma::T
end

UnitGaussianNormaliser(x::AbstractArray) =
    UnitGaussianNormaliser(Float32(mean(x)), Float32(std(x)))

encode(n::UnitGaussianNormaliser, x) = (x .- n.mu) ./ (n.sigma .+ NORM_EPS)
decode(n::UnitGaussianNormaliser, x) = x .* (n.sigma .+ NORM_EPS) .+ n.mu

struct MinMaxNormaliser{T <: AbstractFloat}
    lo::T
    hi::T
end

MinMaxNormaliser(x::AbstractArray) = MinMaxNormaliser(Float32(minimum(x)), Float32(maximum(x)))

encode(n::MinMaxNormaliser, x) = (x .- n.lo) ./ (n.hi - n.lo)
decode(n::MinMaxNormaliser, x) = x .* (n.hi - n.lo) .+ n.lo

function loss_fcn(y_pred, y; p::Real = 2)
    return sum(abs.(y_pred .- y) .^ p)
end

function BIC(model, n_samples::Int, loss_val::Real)
    k = Lux.parameterlength(model)
    return 2 * loss_val + k * log(n_samples)
end

function log_csv(epoch, train_loss, test_loss, bic, elapsed, file_name)
    return open(file_name, "a") do f
        write(f, "$epoch,$elapsed,$train_loss,$test_loss,$bic\n")
    end
end

# FNO grid: creates coord grids
const NX = 32
const NY = 32
const GRID_X = Float32.(reshape(range(0, 1, NX), 1, NX, 1, 1))
const GRID_Y = Float32.(reshape(range(0, 1, NY), 1, 1, NY, 1))

function get_grid(x)
    batch_size = size(x, 4)
    gridx = repeat(GRID_X, 1, 1, NY, batch_size)
    gridy = repeat(GRID_Y, 1, NX, 1, batch_size)
    grid = cat(gridx, gridy; dims = 1)
    x_reshaped = permutedims(x, (3, 1, 2, 4))
    return vcat(x_reshaped, grid)
end

# Conv unfolding for KAN conv layers
struct Slicer
    dh::Int
    dw::Int
    sh::Int
    sw::Int
    out_h::Int
    out_w::Int
end

function (s::Slicer)(input, i, j)
    h_start = (i - 1) * s.dh + 1
    w_start = (j - 1) * s.dw + 1
    h_indices = h_start:(s.sh):(h_start + s.sh * (s.out_h - 1))
    w_indices = w_start:(s.sw):(w_start + s.sw * (s.out_w - 1))
    slice = input[h_indices, w_indices, :, :]
    return reshape(slice, 1, size(slice)...)
end

function unfold(input, kernel_size; stride = 1, padding = 0, dilation = 1)
    H, W, C, N = size(input)
    kh, kw = kernel_size
    sh, sw = stride isa NTuple ? stride : (stride, stride)
    ph, pw = padding isa NTuple ? padding : (padding, padding)
    dh, dw = dilation isa NTuple ? dilation : (dilation, dilation)

    out_h = div(H + 2ph - (dh * (kh - 1) + 1), sh) + 1
    out_w = div(W + 2pw - (dw * (kw - 1) + 1), sw) + 1

    slicer = Slicer(dh, dw, sh, sw, out_h, out_w)
    padded_input = NNlib.pad_zeros(input, (ph, pw, 0, 0))

    output = similar(input, 0, kw, out_h, out_w, C, N)

    for i in 1:kh
        inner_output = similar(input, 0, out_h, out_w, C, N)
        for j in 1:kw
            inner_output = cat(inner_output, slicer(padded_input, i, j); dims = 1)
        end
        inner_output = reshape(inner_output, 1, size(inner_output)...)
        output = cat(output, inner_output; dims = 1)
    end
    return output
end
