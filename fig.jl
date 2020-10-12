using Plots, JLD2, SpecialFunctions, DSP, FFTW, Dates, Colors, ImageFiltering, LaTeXStrings

mrk = Dates.format(now(), "ddmm_HHMM")

N_cycles = 40
ω = 2π
Tf = 2π/ω
x = range(-N_cycles*Tf/2, stop=N_cycles*Tf/2, length=10001)[1:end-1];
t0 = x[1]
σ = Tf
ζ = 2.0

k = range(-π, stop=π, length=1001)

pyplot()

prefix = #input array of 3 prefixes, e.g. ["0809_0958", "0709_0956", "1209_1232"]
src = [jldopen("resps-cep-$prefix.jld2") for prefix=prefix]
res = [jldopen("results-$prefix.jld2") for prefix=prefix]

display.(size.([res[i]["data_tst"] for i=1:3]))

sq3(x) = sign(x)*(abs(x))^(1//3)
cutf(t) = (t > t0+ζ*Tf ? 1.0 : exp(-(t-t0-ζ*Tf)^2 /(2(ζ*Tf/4)^2)))

function f(eiφ, β, ϵ, z)
    α = β*π*(σ/2)^2
    Σ = sqrt(σ^2 + α^2/σ^2)
    μ = (σ^2 - 1im*α)*(σ/Σ)^2
    λ = ϵ*8π*(σ/2)^3

    exparg = sq3(2/λ)*(z + μ^2 /2λ) |> ComplexF64
    corr = (1//3)*μ^3/λ^2 +z*μ/λ
    if abs(corr) < 400
        ret = cutf(-z)*cutf(z)*2*sqrt(π)*abs(sqrt(μ/2))*abs(2/λ)^(1//3)*airyai(exparg)*exp(corr) #*exp(-z^2 /2μ)
    else
        ret = cutf(-z)*cutf(z)*exp(-1im*angle(μ)/2)*exp(-z^2/(2μ))
    end

    return ret*exp(1im*ω*z)*eiφ
end

ind = rand(1:size(res[1]["pars_infd"])[2], 3)
println(ind)

clims = Tuple(log10.([imum([imum(res[i]["data_tst"][:,:,ind[i]]) for i=1:3]) for imum=[minimum, maximum]]))
clrs = distinguishable_colors(2, colorant"blue", lchoices=range(0, stop=50, length=15)) |> permutedims

p1 = [begin
    ω = src[i]["omega0"]
    φ = src[i]["phi"]
    data = imfilter(res[i]["data_tst"][:,:,ind[i]], Kernel.gaussian(1)) #
    plot(φ, (0:size(data)[1]-1), log10.(data), xlabel="CEP change, rad", ylabel=(i==1 ? "Harmonic number" : ""), colorbar=(i==3), colorbar_title="Harmonic yield, log scale", legend=false, tickfontsize=12, seriestype=:heatmap, thickness_scaling=2, clims=clims)
end
for i=1:3]

ϵ(k, h) = sum(h[i]*cos((i-1)*k) for i=eachindex(h))

clrs = distinguishable_colors(2, colorant"blue", lchoices=range(0, stop=50, length=15)) |> permutedims

println("---")

p2 = [begin
    pars_infd = res[i]["pars_infd"][:,ind[i]]
    pars_true = res[i]["pars_true"][:,ind[i]]

    for pars = [pars_infd, pars_true]
        pars[1] -= minimum([sum(pars[i]*cos((i-1)*k) for i=2:length(pars)) for k=k])
    end

    println(pars_infd)
    println(pars_true)
    println("")

    plot(k./(2π), [[ϵ(k, pars) for k=k] for pars=[pars_true, pars_infd]], linestyle=[:solid :dash], label=["Ground truth" "Inferred"], xlabel=L"Crystal momentum, $k/k_c$", ylabel=(i==1 ? "Conduction band energy, eV" : ""), tickfontsize=12, thickness_scaling=2, c=clrs)
end
for i=1:3]

println("---")

p3 = [begin
    pulse_infd = res[i]["pulse_infd"][:,ind[i]]
    pulse_true = res[i]["pulse_true"][:,ind[i]]

    ϵ = 1.05

    println(pulse_infd)
    println(pulse_true)
    println("")

    plot(x, [(eiφ = cep[1] + 1im*cep[2]; real.(f.(eiφ/abs(eiφ), cep[3], cep[4], x))) for cep=[pulse_true, pulse_infd]], linestyle=[:solid :dash], label=["Ground truth" "Inferred"], xlabel="Time, cycles", ylabel=(i==1 ? "Vector potential" : ""), xlims=(in(i, 1:2) ? (-5, 5) : (-10, 10)).*ϵ, ylims=(-1, 1).*ϵ, thickness_scaling=2, c=clrs, tickfontsize=12)
end
for i=1:3]

println("---")

p4 = [begin
    pulse_infd = res[i]["pulse_infd"][:,ind[i]]
    pulse_true = res[i]["pulse_true"][:,ind[i]]

    println(pulse_infd)
    println(pulse_true)
    println("")

    ind_last = 4*N_cycles+1

    p = plot(FFTW.rfftfreq(length(x), 1/(x[2]-x[1]))[1:ind_last], [begin
        eiφ = cep[1] + 1im*cep[2]
        y = real.(f.(eiφ/abs(eiφ), cep[3], cep[4], x)) |> fftshift
        ry = rfft(y)[1:ind_last]
        (i == 1 ? angle.(ry) : DSP.unwrap(angle.(ry)) .- (DSP.unwrap(angle.(ry))[2*N_cycles+1] - angle.(ry)[2*N_cycles+1]))./2π
    end
    for cep=[pulse_true, pulse_infd]], linestyle=[:solid :dash], xlabel=L"Frequency, $\omega/\omega_0$", ylabel=(i == 1 ? L"Spectral phase, units of $2\pi$" : ""), label=["Ground truth" "Inferred"], thickness_scaling=2, c=clrs, ylims=(i==1 ? (-0.5, 0.5) : :auto), tickfontsize=12)
    p
end
for i=1:3]

plot(p1..., p2..., p3..., p4..., layout=(4, 3), size=(2400, 1800))

savefig("plots-$mrk.png")

p2d = [begin
    pars_true = res[i]["pars_true"][:,ind[i]]
    pars_true[1] -= minimum([sum(pars_true[i]*cos((i-1)*k) for i=2:length(pars_true)) for k=k])
    plot(k./(2π), [[(-1)^i * ϵ(k, pars_true) for k=k] for i=0:1], xlabel=L"$k/k_c$", ylabel=(i==1 ? "Band energy, eV" : ""), label=["CB" "VB"], thickness_scaling=2)
end
for i=1:3]

plot(p2d..., layout=(1, 3), size=(1200, 600))

savefig("bands-$mrk.png")

#=Uncomment to generate figure with examples of pulses

angs = [1.0+0.0im, 1.0im, 1.0, 1.0][2:end]
chirps = [0.0, 0.0, 2.0, 1.0][2:end]
cubs = [0.0, 0.0, 0.0, 1.0][2:end]

p3 = [begin
    plot(x, (eiφ = angs[i]; real.(f.(eiφ/abs(eiφ), chirps[i], cubs[i], x))), legend=false, xlabel="Time, cycles", ylabel=(i==1 ? "Vector potential" : ""), xlims=(in(i, 1:3) ? (-5, 5) : (-10, 10)), tickfontsize=12, thickness_scaling=2, c=clrs)
end
for i=eachindex(angs)]

p4 = [begin
    ind_last = 4*N_cycles+1

    eiφ = angs[i]
    y = real.(f.(eiφ/abs(eiφ), chirps[i], cubs[i], x)) |> fftshift
    ry = rfft(y)[1:ind_last]

    p = plot(FFTW.rfftfreq(length(x), 1/(x[2]-x[1]))[1:ind_last], (i==1 ? angle.(ry) : DSP.unwrap(angle.(ry)) .- (DSP.unwrap(angle.(ry)) .- angle.(ry))[2*N_cycles+1])./2π, xlabel=L"Frequency, $\omega/\omega_0$", ylabel=(i == 1 ? L"Spectral phase, units of $2\pi$" : ""), legend = false, c=:red, ylims=(i==1 ? (-0.5, 0.5) : :auto), tickfontsize=12, linewidth=2, thickness_scaling=2)

    plot!(twinx(p), FFTW.rfftfreq(length(x), 1/(x[2]-x[1]))[1:ind_last], abs.(ry)./maximum(abs.(ry)), ribbon = (abs.(ry)./maximum(abs.(ry)), zeros(ind_last)), grid=false, minorgrid=false, legend=false, tickfontsize=12, thickness_scaling=2, ylabel="", yaxis=false)
    println(angle(ry[2*N_cycles+1]))
    p
end
for i=eachindex(angs)]

plot(p3..., p4..., layout=(2, length(angs)), size=(2400, 1600))

savefig("pulses-$mrk.png")=#
