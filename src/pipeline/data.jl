using MAT
using MLUtils: DataLoader
using MLDataDevices: MLDataDevices

function load_darcy_data()
    train_matfile = matread("data/2D_DarcyFlow_Data/Darcy_2D_data_train.mat")
    test_matfile = matread("data/2D_DarcyFlow_Data/Darcy_2D_data_test.mat")

    a_train = Float32.(train_matfile["a_field"])
    u_train = Float32.(train_matfile["u_field"])
    a_test = Float32.(test_matfile["a_field"])
    u_test = Float32.(test_matfile["u_field"])

    return a_train, u_train, a_test, u_test
end

function get_darcy_loader(batch_size::Int; dev = MLDataDevices.cpu_device())
    a_train, u_train, a_test, u_test = load_darcy_data()

    a_normaliser = UnitGaussianNormaliser(a_train)
    u_normaliser = UnitGaussianNormaliser(u_train)

    a_train = encode(a_normaliser, a_train)
    u_train = encode(u_normaliser, u_train)
    a_test = encode(a_normaliser, a_test)
    u_test = encode(u_normaliser, u_test)

    # (samples, H, W) -> (H, W, 1, samples)
    to_hwcn(x) = permutedims(
        reshape(x, size(x, 1), size(x, 2), size(x, 3), 1),
        (2, 3, 4, 1),
    )
    a_train = to_hwcn(a_train)
    u_train = to_hwcn(u_train)
    a_test = to_hwcn(a_test)
    u_test = to_hwcn(u_test)

    train_data = dev((a_train, u_train))
    test_data = dev((a_test, u_test))

    train_loader = DataLoader(train_data; batchsize = batch_size, shuffle = true)
    test_loader = DataLoader(test_data; batchsize = batch_size, shuffle = false)
    return train_loader, test_loader
end
