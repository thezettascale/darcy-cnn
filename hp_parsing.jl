module hyperparams

export set_hyperparams

using ConfParser

function set_CNN_params()
    conf = ConfParse("MLP_CNN/CNN_conf.ini")
    parse_conf!(conf)

    ENV["p"] = retrieve(conf, "Loss", "p")
    ENV["step"] = retrieve(conf, "Optimizer", "step_rate")
    ENV["decay"] = retrieve(conf, "Optimizer", "gamma")
    ENV["LR"] = retrieve(conf, "Optimizer", "learning_rate")
    ENV["min_LR"] = retrieve(conf, "Optimizer", "min_lr")
    ENV["activation"] = retrieve(conf, "Architecture", "activation")
    ENV["hidden_dim"] = retrieve(conf, "Architecture", "hidden_dim")
    ENV["batch_size"] = retrieve(conf, "Dataloader", "batch_size")
    ENV["num_epochs"] = retrieve(conf, "Pipeline", "num_epochs")
    ENV["optimizer"] = retrieve(conf, "Optimizer", "type")

    return nothing
end

function set_FNO_params()
    conf = ConfParse("MLP_FNO/FNO_conf.ini")
    parse_conf!(conf)

    ENV["p"] = retrieve(conf, "Loss", "p")
    ENV["step"] = retrieve(conf, "Optimizer", "step_rate")
    ENV["decay"] = retrieve(conf, "Optimizer", "gamma")
    ENV["LR"] = retrieve(conf, "Optimizer", "learning_rate")
    ENV["min_LR"] = retrieve(conf, "Optimizer", "min_lr")
    ENV["activation"] = retrieve(conf, "Architecture", "activation")
    ENV["width"] = retrieve(conf, "Architecture", "channel_width")
    ENV["modes1"] = retrieve(conf, "Architecture", "modes1")
    ENV["modes2"] = retrieve(conf, "Architecture", "modes2")
    ENV["num_blocks"] = retrieve(conf, "Architecture", "num_hidden_blocks")
    ENV["batch_size"] = retrieve(conf, "Dataloader", "batch_size")
    ENV["num_epochs"] = retrieve(conf, "Pipeline", "num_epochs")
    ENV["optimizer"] = retrieve(conf, "Optimizer", "type")

    return nothing
end

function set_hyperparams(model_name)

    if model_name == "CNN"
        out = set_CNN_params()
    elseif model_name == "FNO"
        out = set_FNO_params()
    else
        println("Invalid model name")
    end

    return out
end

end