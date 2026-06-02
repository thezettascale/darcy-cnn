function step_decay(epoch, lr, step, decay, min_lr)
    return max(lr * decay^(epoch / step), min_lr)
end

function compile_eval(model, ps, st, test_loader, loss_fn)
    (x_first, y_first) = first(test_loader)
    st_test = Lux.testmode(st)
    eval_loss(ps, st, x, y) = loss_fn(model(x, ps, st)[1], y)
    return @compile eval_loss(ps, st_test, x_first, y_first)
end

function train_epoch(
        train_state, train_loader, test_loader,
        loss_fn, model, eval_fwd, epoch, cfg::ModelConfig,
    )
    train_loss = 0.0
    test_loss = 0.0

    function objective(model, ps, st, (x, y))
        y_pred, st_new = model(x, ps, st)
        return loss_fn(y_pred, y), st_new, (;)
    end

    for (x, y) in train_loader
        _, loss_val, _, train_state = Training.single_train_step!(
            AutoEnzyme(), objective, (x, y), train_state,
        )
        train_loss += Float32(loss_val)
    end

    st_test = Lux.testmode(train_state.states)
    for (x, y) in test_loader
        test_loss += eval_fwd(train_state.parameters, st_test, x, y) |> Float32
    end

    new_lr = step_decay(epoch, cfg.learning_rate, cfg.step_rate, cfg.gamma, cfg.min_lr)
    Optimisers.adjust!(train_state.optimizer_state, new_lr)

    n_train = length(train_loader.data)
    n_test = length(test_loader.data)
    return train_state, train_loss / n_train, test_loss / n_test
end
