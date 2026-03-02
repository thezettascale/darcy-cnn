include("../src/WavKANConv.jl")
using .WavKANConv
using Lux, Random, Test

const RNG = Xoshiro(0)

@testset "Wavelets" begin
    for name in keys(WavKANConv.WAVELET_MAP)
        w = WavKANConv.create_wavelet(name, 3, 5)
        ps, st = Lux.setup(RNG, w)
        x = randn(Float32, 3, 5, 4)
        y, _ = w(x, ps, st)
        @test size(y) == (5, 4)
    end
end

@testset "Wavelets 4D" begin
    for name in keys(WavKANConv.WAVELET_MAP)
        w = WavKANConv.create_wavelet(name, 3, 5)
        ps, st = Lux.setup(RNG, w)
        x = randn(Float32, 3, 5, 7, 4)
        y, _ = w(x, ps, st)
        @test size(y) == (5, 7, 4)
    end
end

@testset "KANdense" begin
    layer = WavKANConv.KANdense(4, 3, "Morlet")
    ps, st = Lux.setup(RNG, layer)
    y, _ = layer(randn(Float32, 4, 8), ps, st)
    @test size(y) == (3, 8)
end

@testset "KANConv2D" begin
    layer = WavKANConv.KANConv2D(
        1, 4, (3, 3), "Morlet", "relu";
        padding = 1, norm = false,
    )
    ps, st = Lux.setup(RNG, layer)
    x = randn(Float32, 8, 8, 1, 2)
    y, _ = layer(x, ps, st)
    @test size(y) == (8, 8, 4, 2)
end

@testset "KANConvTranspose2D" begin
    layer = WavKANConv.KANConvTranspose2D(
        4, 1, (3, 3), "MexicanHat", "relu";
        padding = 1, norm = false,
    )
    ps, st = Lux.setup(RNG, layer)
    x = randn(Float32, 8, 8, 4, 2)
    y, _ = layer(x, ps, st)
    @test size(y)[3:4] == (1, 2)
end

@testset "CNN forward" begin
    cfg = CNNConfig(2, "relu", 1.0f-3, 10, 0.8f0, 1.0f-5, 2, 1, 2.0f0)
    model = create_model(cfg)
    ps, st = Lux.setup(RNG, model)
    x = randn(Float32, 32, 32, 1, 2)
    out, _ = model(x, ps, st)
    @test size(out) == (32, 32, 1, 2)
end

@testset "FNO forward" begin
    cfg = FNOConfig(16, 4, 4, 2, "gelu", 1.0f-3, 10, 0.8f0, 1.0f-4, 2, 1, 2.0f0)
    model = create_model(cfg)
    ps, st = Lux.setup(RNG, model)
    x = randn(Float32, 32, 32, 1, 2)
    out, _ = model(x, ps, st)
    @test size(out) == (32, 32, 1, 2)
end

@testset "KANCNN forward" begin
    cfg = KANCNNConfig(
        2, false,
        ["Morlet", "MexicanHat", "Shannon"],
        ["relu", "relu", "relu"],
        ["Morlet", "MexicanHat", "Shannon", "Meyer"],
        ["relu", "relu", "relu", "relu"],
        1.0f-3, 10, 0.8f0, 1.0f-5, 2, 1, 2.0f0,
    )
    model = create_model(cfg)
    ps, st = Lux.setup(RNG, model)
    x = randn(Float32, 32, 32, 1, 2)
    out, _ = model(x, ps, st)
    @test size(out) == (32, 32, 1, 2)
end
