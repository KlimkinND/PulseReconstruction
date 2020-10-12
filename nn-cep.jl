using Flux, CUDA, JLD2, Random, Statistics, LinearAlgebra, FFTW, ProgressMeter

using Flux: @epochs, train!, throttle

using ProgressMeter

const batch_size = 512
const aug_step = 22

cubic = true
chirp = true
augment = true
ssh = false
corr_pars = !ssh

learn_pars = true

prefix = "0110_2037"#"0809_0958"##"3007_2230"#"2906_1917"#"2004_1549"#""0809_0958"#
file = jldopen("resps-cep-$prefix.jld2")

cep = permutedims(file["cep"])
if chirp
    try
        global chp = permutedims(file["chirp"])
    catch
        global chirp = false
    end
end
if cubic
    try
        global cbp = permutedims(file["cbp"])#).^(1//3)
    catch
        global cubic = false
    end
end

exp_smp = (1:size(cep)[2])

N_cycles = file["N"]
data = file["data_freq"][:,:,exp_smp]
N_pars = size(data)[3]
pars = ssh ? permutedims(hcat(file["delta"], file["h1"], file["h2"]))[:, exp_smp] : file["pars"][:,exp_smp]

ord = size(pars)[1]

if corr_pars
    κ = range(-π, stop=π, length=101)

    Threads.@threads for j=1:N_pars
        pars[1, j] += minimum([sum(pars[i,j]*cos((i-1)*κ) for i=2:ord) for κ=κ])
    end
end

Nω = size(data)[1]
Nφ = size(data)[2]

N_test = 32

N_train = N_pars - N_test*batch_size

data_tst = data[:,:,N_train+1:end]
pars_tst = pars[:,N_train+1:end]
cep_tst = cep[:,N_train+1:end]
chirp && (chp_tst = chp[:,N_train+1:end])
cubic && (cbp_tst = cbp[:,N_train+1:end])

data_trn = data[:,:,1:N_train]
pars_trn = pars[:,1:N_train]
cep_trn = cep[:,1:N_train]
chirp && (chp_trn = chp[:,1:N_train])
cubic && (cbp_trn = cbp[:,1:N_train])

feed_fft = true

if feed_fft
    data_l = rfft(data_tst, 2)

    imgs = reshape(abs.(data_l), :, N_pars-N_train)
    imgs_phase_cos = reshape(real.(data_l), :, N_pars-N_train)
    imgs_phase_sin = reshape(imag.(data_l), :, N_pars-N_train)
    imgs_φ = reshape(data_tst, :, N_pars-N_train)

    data_x_tst = vcat(imgs, imgs_phase_cos, imgs_φ)

    data_l_trn = rfft(data_trn, 2)

    imgs = reshape(abs.(data_l_trn), :, N_train)
    imgs_phase_cos = reshape(real.(data_l_trn), :, N_train)
    imgs_phase_sin = reshape(imag.(data_l_trn), :, N_train)
    imgs_φ = reshape(data_trn, :, N_train)

    imgs_aug = copy(imgs)
    imgs_real_aug = copy(imgs_phase_cos)
    imgs_imag_aug = copy(imgs_phase_sin)
    imgs_φ_aug = copy(imgs_φ)

    pars_aug = copy(pars_trn)
    cep_aug = copy(cep_trn)
    chirp && (chp_aug = copy(chp_trn))
    cubic && (cbp_aug = copy(cbp_trn))

    if augment
        @showprogress "Preparing data..." for (j, φ) = collect(enumerate(collect(range(0.0, stop=2π, length=Nφ+1))[1:end-1]))[1:aug_step:end][2:end]
            data_shft = circshift(data_trn, (0, j-1, 0))
            data_l_shft = rfft(data_shft, 2)

            global imgs_aug = hcat(imgs_aug, reshape(abs.(data_l_shft), :, N_train))
            global imgs_real_aug = hcat(imgs_real_aug, reshape(real.(data_l_shft), :, N_train))
            global imgs_imag_aug = hcat(imgs_imag_aug, reshape(imag.(data_l_shft), :, N_train))
            global imgs_φ_aug = hcat(imgs_φ_aug, reshape(data_shft, :, N_train))

            global pars_aug = hcat(pars_aug, pars_trn)
            global cep_aug = hcat(cep_aug, cep_trn .- φ)
            chirp && (global chp_aug = hcat(chp_aug, chp_trn))
            cubic && (global cbp_aug = hcat(cbp_aug, cbp_trn))
        end
    end

    data_x = vcat(imgs_aug, imgs_real_aug, imgs_φ_aug)
else
    data_x = reshape(data, :, N_pars)
end

data_y = pars_aug
data_yφ = vcat(cos.(cep_aug), sin.(cep_aug), eval.([:chp_aug, :cbp_aug][BitArray([chirp, cubic])])...) #: vcat(cos.(cep_aug), sin.(cep_aug))
data_yφ_tst = vcat(cos.(cep_tst), sin.(cep_tst), eval.([:chp_tst, :cbp_tst][BitArray([chirp, cubic])])...) #: vcat(cos.(cep_tst), sin.(cep_tst))

ws = vcat(ones(ord), 0.01.*ones(2)) |> gpu

D1 = size(data_x)[1]
D2 = 800
D3 = 400
D4 = ord+2
D5 = 400
D6 = size(data_y)[1]

M_pars = Chain(BatchNorm(D1), Dense(D1, D2, swish), Dense(D2, D3, swish), Dense(D3, ord)) |> gpu
M_cep = Chain(BatchNorm(D1), Dense(D1, D2, swish), Dense(D2, D3, swish), Dense(D3, D3, swish)) |> gpu

phi = Chain(Dense(D3, 2, tanh)) |> gpu
sphase = Dense(D3, chirp+cubic) |> gpu

inds = collect(Iterators.partition(randperm(size(data_x)[2]), batch_size))
N_batches = length(inds)

P = data_x |> gpu
Q = data_y |> gpu
R = data_yφ |> gpu

train_loader_p = Flux.Data.DataLoader((P, Q), batchsize=batch_size, shuffle=true)
train_loader_cep = Flux.Data.DataLoader((P, R), batchsize=batch_size, shuffle=true)

train_last = size(data_x)[2]-N_test*batch_size
test_data_p = (gpu(data_x_tst), gpu(pars_tst))
test_data_cep = (gpu(data_x_tst), gpu(data_yφ_tst))

opt = ADAM()

evalcb() = (lng = length(test_data); test_loss = sum(L(test_data[i]...) for i=1:lng)/lng; @show(test_loss))
diagonal(A::AbstractMatrix, k::Integer=0) = view(A, diagind(A, k))

function train_NN!(N, model, loader, test_data, L)
    loss_best = L(test_data...)
    pars_best = similar.(params(model))
    [pars_best[i] .= copy(collect(params(model))[i]) for i=eachindex(pars_best)]

    @showprogress "Training..." for i=1:N
        Flux.train!(L, params(model), loader, opt)
        loss_curr = L(test_data...)

        if loss_best > loss_curr
            loss_best = loss_curr
            [pars_best[i] .= copy(collect(params(model))[i]) for i=eachindex(pars_best)]
        end
    end

    return pars_best
end

L_pars(x, y) = Flux.mse(M_pars(x), y)

if learn_pars
    opt.eta=1f-3
    train_NN!(40, M_pars, train_loader_p, test_data_p, L_pars)
    opt.eta = 1f-4
    train_NN!(20, M_pars, train_loader_p, test_data_p, L_pars)
    opt.eta = 5f-5
    band_model = train_NN!(10, M_pars, train_loader_p, test_data_p, L_pars)
end

CUDA.allowscalar(false)

L_cep(x, y) = Flux.mse(phi(M_cep(x)), y[1:2, :]) + ((cubic || chirp) ? Flux.mse(sphase(M_cep(x)), y[3:end, :]) : 0f0)

opt.eta = 1f-3
train_NN!(40, (M_cep, phi, sphase), train_loader_cep, test_data_cep, L_cep)
opt.eta = 1f-4
train_NN!(20, (M_cep, phi, sphase), train_loader_cep, test_data_cep, L_cep)
opt.eta = 5f-5
pulse_model = train_NN!(10, (M_cep, phi, sphase), train_loader_cep, test_data_cep, L_cep)

using BenchmarkTools

Flux.loadparams!(M_pars, band_model)
Flux.loadparams!((M_cep, phi, sphase), pulse_model)

pars_eval = cpu(test_data_p[2])
pars_infd = cpu(M_pars(test_data_p[1]))

Δ = pars_eval .- pars_infd
Δ_rel = (pars_infd .- pars_eval)./pars_eval

#display(collect(zip(pars_eval[:, 1:7], pars_infd[:, 1:7])))

println("Absolute parameter errors:")
println(round.(vec(std(Δ, dims=2)), digits=3))
println("")
println("Relative parameter errors:")
println(round.(vec(sqrt.(mean(abs2.(Δ_rel), dims=2))), digits=3)) #if parameter changes sign, relative error will probably be greater than 1
println("")

cep_eval = test_data_cep[2][1:2, :] |> cpu
cep_infd = phi(M_cep(test_data_cep[1])) |> cpu

nrms = mapslices(norm, cep_infd, dims=1)
cep_infd .= cep_infd./nrms

#display(collect(zip(cep_eval[:, 1:7], cep_infd[:, 1:7])))

cθ = diag(cep_eval'cep_infd)
δφ = sqrt(2*mean(1.0 .- cθ))
println("CEP error:")
println(round(δφ, digits=3))

if chirp
    chirp_eval = test_data_cep[2][3, :] |> cpu
    chirp_infd = sphase(M_cep(test_data_cep[1]))[1, :] |> cpu
    println("Chirp error:")
    println(round(std(vec(chirp_eval .- chirp_infd)), digits=3))
end
if cubic
    cubic_eval = test_data_cep[2][4, :] |> cpu
    cubic_infd = sphase(M_cep(test_data_cep[1]))[2, :] |> cpu
    println("Cubic phase error:")
    println(round(std(vec(cubic_eval .- cubic_infd)), digits=3))
end

pulse_true = cpu(test_data_cep[2])
pulse_infd = vcat(cpu(phi(M_cep(test_data_cep[1]))), cpu(sphase(M_cep(test_data_cep[1]))))

jldopen("results-$prefix.jld2", true, true, true, IOStream) do io
    write(io, "data_tst", data_tst)
    write(io, "pars_true", pars_eval)
    write(io, "pars_infd", pars_infd)
    write(io, "pulse_true", vcat(pulse_true, zeros(4-size(pulse_true)[1], size(pulse_true)[2])))
    write(io, "pulse_infd", vcat(pulse_infd, zeros(4-size(pulse_true)[1], size(pulse_true)[2])))
    write(io, "M1", cpu(collect(band_model)))
    write(io, "M2", cpu(collect(pulse_model)))
end
