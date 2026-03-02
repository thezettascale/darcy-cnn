struct FNO{I, H, O} <: Lux.AbstractLuxContainerLayer{(:input_layer, :hidden_layers, :output_layer)}
    input_layer::I
    hidden_layers::H
    output_layer::O
end

function FNO(cfg::FNOConfig)
    input_layer = Lux.Dense(3 => cfg.width, get_activation(cfg.activation))
    blocks = [
        FNOBlock(cfg.width, cfg.modes1, cfg.modes2, cfg.activation)
            for _ in 1:(cfg.num_blocks)
    ]
    hidden_layers = _make_named_layers(blocks)
    output_layer = FNO_MLP(cfg.width, 1, cfg.width * 4, cfg.activation)
    return FNO(input_layer, hidden_layers, output_layer)
end

function (m::FNO)(x, ps, st)
    x = get_grid(x)
    x, st_i = m.input_layer(x, ps.input_layer, st.input_layer)
    x = permutedims(x, (2, 3, 1, 4))

    st_h = st.hidden_layers
    for k in keys(m.hidden_layers)
        x, st_h_k = m.hidden_layers[k](x, ps.hidden_layers[k], st_h[k])
        st_h = merge(st_h, NamedTuple{(k,)}((st_h_k,)))
    end

    x, st_o = m.output_layer(x, ps.output_layer, st.output_layer)
    return x, (input_layer = st_i, hidden_layers = st_h, output_layer = st_o)
end
