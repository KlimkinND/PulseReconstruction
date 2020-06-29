using Flux, CuArrays, JLD2, Random, Statistics, LinearAlgebra, FFTW

using Flux: @epochs, train!, throttle

using ProgressMeter

const batch_size = 2048

chirp = true

prefix = "2906_1531"#"2004_1549"#"
file = jldopen("resps-cep-$prefix.jld2")
N_pars = size(file["data_freq"])[3]
perm = randperm(N_pars)
N_cycles = file["N"]
pars = file["pars"]
data = file["data_freq"]

Nω = size(data)[1]
Nφ = size(data)[2]

cep = permutedims(file["cep"])
if chirp
    try
        global chp = permutedims(file["chirp"])
    catch
        chirp = false
    end
end

N_test = 4

N_train = N_pars - N_test*batch_size

data_tst = data[:,:,N_train+1:end]
pars_tst = pars[:,N_train+1:end]
cep_tst = cep[:,N_train+1:end]
chirp && (chp_tst = chp[:,N_train+1:end])

data_trn = data[:,:,1:N_train]
pars_trn = pars[:,1:N_train]
cep_trn = cep[:,1:N_train]
chirp && (chp_trn = chp[:,1:N_train])

#data_x_cv = Array{Float32}(undef, size(data)[1], size(data)[2], 3, size(data)[3])

feed_fft = true

if feed_fft
    data_l = rfft(data, 2)

    imgs = reshape(abs.(data_l), :, N_pars)
    imgs_phase_cos = reshape(real.(data_l), :, N_pars)
    imgs_phase_sin = reshape(imag.(data_l), :, N_pars)
    imgs_φ = reshape(data, :, N_pars)

    data_x_tst = vcat(imgs, imgs_phase_cos, imgs_phase_sin, imgs_φ)[:,N_train+1:end]

    data_l_trn = rfft(data_trn, 2)

    imgs = reshape(abs.(data_l_trn), :, N_train)
    imgs_phase_cos = reshape(real.(data_l_trn), :, N_train)
    imgs_phase_sin = reshape(imag.(data_l_trn), :, N_train)
    imgs_φ = reshape(data_trn, :, N_train)

    imgs_aug = copy(imgs)#Array{Float32}(undef, size(imgs)[1], N_pars*Nφ)
    imgs_real_aug = copy(imgs_phase_cos)#Array{Float32}(undef, size(imgs)[1], N_pars*Nφ)
    imgs_imag_aug = copy(imgs_phase_sin)#Array{Float32}(undef, size(imgs)[1], N_pars*Nφ)
    imgs_φ_aug = copy(imgs_φ)#Array{Float32}(undef, size(imgs_φ)[1], N_pars*Nφ)

    pars_aug = copy(pars_trn)
    cep_aug = copy(cep_trn)
    chirp && (chp_aug = copy(chp_trn))

    @showprogress for (j, φ) = collect(enumerate(collect(range(0.0, stop=2π, length=Nφ+1))[1:end-1]))[1:8:end][2:end]
        data_shft = circshift(data_trn, (0, j-1, 0))
        data_l_shft = rfft(data_shft, 2)

        global imgs_aug = hcat(imgs_aug, reshape(abs.(data_l_shft), :, N_train))
        global imgs_real_aug = hcat(imgs_real_aug, reshape(real.(data_l_shft), :, N_train))
        global imgs_imag_aug = hcat(imgs_imag_aug, reshape(imag.(data_l_shft), :, N_train))
        global imgs_φ_aug = hcat(imgs_φ_aug, reshape(data_shft, :, N_train))

        global pars_aug = hcat(pars_aug, pars_trn)
        global cep_aug = hcat(cep_aug, cep_trn .- φ)
        chirp && (global chp_aug = hcat(chp_aug, chp_trn))
    end

    data_x = vcat(imgs_aug, imgs_real_aug, imgs_imag_aug, imgs_φ_aug)
else
    data_x = reshape(data, :, N_pars)
end

κ = range(-π, stop=π, length=101)
ord = size(pars)[1]

Threads.@threads for j=1:N_pars
    pars[1, j] += minimum([sum(pars[i,j]*cos((i-1)*κ) for i=2:ord) for κ=κ])
end

#=data_y = chirp ? vcat(pars_aug, chp_aug) : pars_aug
data_yφ = vcat(cos.(cep_aug), sin.(cep_aug))
data_yφ_tst = vcat(cos.(cep_tst), sin.(cep_tst))=#

data_y = pars_aug
data_yφ = chirp ? vcat(cos.(cep_aug), sin.(cep_aug), chp_aug) : vcat(cos.(cep_aug), sin.(cep_aug))
data_yφ_tst = chirp ? vcat(cos.(cep_tst), sin.(cep_tst), chp_tst) : vcat(cos.(cep_tst), sin.(cep_tst))

ws = vcat(ones(ord), 0.01.*ones(2)) |> gpu

display(size(data_x))
display(size(data_y))
display(size(data_yφ))

#imgs .= imgs.*(1.0 .+ 0.01.*randn(size(imgs))) #log.(imgs) .* (1.0 .+ 2e-1.*randn(size(imgs)))

#display(size(imgs))

D1 = size(data_x)[1]
D2 = 400
D3 = 800
D4 = ord+2
D5 = 200
D6 = size(data_y)[1]

M = Chain(BatchNorm(D1), Dense(D1, D2, swish), Dense(D2, D3, swish)) |> gpu
Φ = Dense(D3, chirp ? 3 : 2) |> gpu
Π = Chain(Dense(D3, D5, swish), Dense(D5, D6)) |> gpu

inds = collect(Iterators.partition(randperm(size(data_x)[2]), batch_size))
N_batches = length(inds)

train_data = Array{Tuple{CuArray{Float32, 2}, CuArray{Float32, 2}, CuArray{Float32, 2}}, 1}(undef, N_batches)
test_data = Array{Tuple{CuArray{Float32, 2}, CuArray{Float32, 2}, CuArray{Float32, 2}}, 1}(undef, 1)

@showprogress for (j, ind) = enumerate(inds)
    train_data[j] = (data_x[:,ind], data_y[:,ind], data_yφ[:,ind]) |> gpu
end

P = gpu(data_x)
Q = gpu(data_y)
R = gpu(data_yφ)

train_loader = Flux.Data.DataLoader((P, Q, R), batchsize=batch_size, shuffle=true)

train_last = size(data_x)[2]-N_test*batch_size
test_data[1] = (gpu(data_x_tst), gpu(pars_tst), gpu(data_yφ_tst))

M(test_data[1][1])
Φ(M(test_data[1][1]))
Π(M(test_data[1][1]))

display(size.(test_data[1]))

opt = ADAM(1f-3)

L(x, pars, φ) = Flux.mse(Φ(M(x)), φ) + Flux.mse(Π(M(x)), pars)
evalcb() = (lng = length(test_data); test_loss = sum(L(test_data[i]...) for i=1:lng)/lng; @show(test_loss))

function train_NN!(N_epochs; last = false, pars = true, cep = true)
    for epoch=1:N_epochs
        print("Epoch $epoch: ")
        if !last
            cep && train!(L, params(M, Φ), train_loader, opt)
            pars && train!(L, params(M, Π), train_loader, opt)
        else
            cep && train!(L, params(Φ), train_loader, opt)
            pars && train!(L, params(Π), train_loader, opt)
        end
        println(L(test_data[1]...))
    end
end

opt.eta=1f-3
train_NN!(10, pars=false, )

opt.eta = 1f-4
train_NN!(10)

#TODO: make this a function

#=imgs_aug = copy(imgs)#Array{Float32}(undef, size(imgs)[1], N_pars*Nφ)
imgs_real_aug = copy(imgs_phase_cos)#Array{Float32}(undef, size(imgs)[1], N_pars*Nφ)
imgs_imag_aug = copy(imgs_phase_sin)#Array{Float32}(undef, size(imgs)[1], N_pars*Nφ)
imgs_φ_aug = copy(imgs_φ)#Array{Float32}(undef, size(imgs_φ)[1], N_pars*Nφ)

pars_aug = copy(pars_trn)
cep_aug = copy(cep_trn)
chp_aug = copy(chp_trn)

@showprogress for (j, φ) = collect(enumerate(collect(range(0.0, stop=2π, length=Nφ+1))[1:end-1]))[5:8:end][2:end]
    data_shft = circshift(data_trn, (0, j-1, 0))
    data_l_shft = rfft(data_shft, 2)

    global imgs_aug = hcat(imgs_aug, reshape(abs.(data_l_shft), :, N_train))
    global imgs_real_aug = hcat(imgs_real_aug, reshape(real.(data_l_shft), :, N_train))
    global imgs_imag_aug = hcat(imgs_imag_aug, reshape(imag.(data_l_shft), :, N_train))
    global imgs_φ_aug = hcat(imgs_φ_aug, reshape(data_shft, :, N_train))

    global pars_aug = hcat(pars_aug, pars_trn)
    global cep_aug = hcat(cep_aug, cep_trn .- φ)
    chirp && (global chp_aug = hcat(chp_aug, chp_trn))
end

data_x = vcat(imgs_aug, imgs_real_aug, imgs_imag_aug, imgs_φ_aug)

train_loader = nothing
P = nothing
P = gpu(data_x)
data_y = chirp ? vcat(pars_aug, chp_aug) : pars_aug
data_yφ = vcat(cos.(cep_aug), sin.(cep_aug))

train_loader = Flux.Data.DataLoader((P, Q, R), batchsize=batch_size, shuffle=true)

opt.eta = 1f-4
train_NN!(40, last=true)=#

using BenchmarkTools

pars_eval = cpu(test_data[1][2])
pars_infd = cpu(Π(M(test_data[1][1])))

Δ = pars_eval .- pars_infd

#display(collect(zip(pars_eval, pars_infd)))
#println("")
println(round.(vec(std(Δ, dims=2)), digits=3))
println("")

cep_eval = cpu(test_data[1][3])[1:2, :]
cep_infd = cpu(Φ(M(test_data[1][1])))[1:2, :]

nrms = mapslices(norm, cep_infd, dims=1)
cep_infd .= cep_infd./nrms

cθ = diag(cep_eval'cep_infd)
δφ = sqrt(2*mean(1.0 .- cθ))
println(round(δφ, digits=3))

if chirp
    chirp_eval = cpu(test_data[1][3])[3, :]
    chirp_infd = cpu(Φ(M(test_data[1][1])))[3, :]

    println(round(std(vec(chirp_eval .- chirp_infd)), digits=3))
end
