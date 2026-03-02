struct CNN{E, D} <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder::E
    decoder::D
end

function CNN(cfg::CNNConfig)
    phi = get_activation(cfg.activation)
    h = cfg.hidden_dim
    encoder = Lux.Chain(
        Lux.Conv((3, 3), 1 => 2h, phi; pad = 1),
        Lux.Conv((3, 3), 2h => 4h, phi; pad = 1),
        Lux.Conv((3, 3), 4h => 8h, phi; pad = 1),
    )
    decoder = Lux.Chain(
        Lux.ConvTranspose((3, 3), 8h => 4h, phi; pad = 1),
        Lux.ConvTranspose((3, 3), 4h => 2h, phi; pad = 1),
        Lux.ConvTranspose((3, 3), 2h => h, phi; pad = 1),
        Lux.ConvTranspose((3, 3), h => 1, phi; pad = 1),
    )
    return CNN(encoder, decoder)
end

function (m::CNN)(x, ps, st)
    x, st_e = m.encoder(x, ps.encoder, st.encoder)
    x, st_d = m.decoder(x, ps.decoder, st.decoder)
    return x, (encoder = st_e, decoder = st_d)
end
