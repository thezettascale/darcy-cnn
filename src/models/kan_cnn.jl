struct KANCNN{E, D} <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder::E
    decoder::D
end

function KANCNN(cfg::KANCNNConfig)
    h = cfg.hidden_dim
    ew, ea = cfg.encoder_wavelet_names, cfg.encoder_activations
    dw, da = cfg.decoder_wavelet_names, cfg.decoder_activations
    encoder_layers = [
        KANConv2D(1, 2h, (3, 3), ew[1], ea[1]; padding = 1, norm = cfg.norm),
        KANConv2D(2h, 4h, (3, 3), ew[2], ea[2]; padding = 1, norm = cfg.norm),
        KANConv2D(4h, 8h, (3, 3), ew[3], ea[3]; padding = 1, norm = cfg.norm),
    ]
    decoder_layers = [
        KANConvTranspose2D(8h, 4h, (3, 3), dw[1], da[1]; padding = 1, norm = cfg.norm),
        KANConvTranspose2D(4h, 2h, (3, 3), dw[2], da[2]; padding = 1, norm = cfg.norm),
        KANConvTranspose2D(2h, h, (3, 3), dw[3], da[3]; padding = 1, norm = cfg.norm),
        KANConvTranspose2D(h, 1, (3, 3), dw[4], da[4]; padding = 1, norm = cfg.norm),
    ]
    return KANCNN(_make_named_layers(encoder_layers), _make_named_layers(decoder_layers))
end

function (m::KANCNN)(x, ps, st)
    st_enc = st.encoder
    for k in keys(m.encoder)
        x, st_enc_k = m.encoder[k](x, ps.encoder[k], st_enc[k])
        st_enc = merge(st_enc, NamedTuple{(k,)}((st_enc_k,)))
    end

    st_dec = st.decoder
    for k in keys(m.decoder)
        x, st_dec_k = m.decoder[k](x, ps.decoder[k], st_dec[k])
        st_dec = merge(st_dec, NamedTuple{(k,)}((st_dec_k,)))
    end

    return x, (encoder = st_enc, decoder = st_dec)
end
