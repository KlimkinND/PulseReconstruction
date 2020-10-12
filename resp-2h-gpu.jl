using LinearAlgebra, FFTW, Plots, Dates, ProgressMeter

using CUDA
using StaticArrays

println("$(Threads.nthreads()) threads")
BLAS.set_num_threads(32)

const mode = "cep"
const ssh = false

const f = 4.0
const q = 0.1
const ω = 0.8
const T = 2π/ω
const τ1 = Inf
const τ2 = T/2
const N_cycles = 40
const Σ = T
const t0 = T*N_cycles/2
const Tmax = N_cycles*T
const z = 2

const Nk = 32
const Nf = 64
const hmax = 40
const K = range(0, stop=2π, length=Nk+1)[1:end-1] |> cu
const Ts = range(0.0, stop=Tmax, length=2*hmax*N_cycles+1) |> cu
const Nph = 2
const dt = Tmax/(2*hmax*N_cycles)
const τs = range(0.0, stop=dt, length=Nph+1)[1:end-1] |> cu
const dτ = dt/Nph
const Φ = range(0.0, stop=2π, length=Nf+1)[1:end-1] |> cu
const ξ = 2.0

const β = 2.0
smclamp(x, a, b) = (c = 0.5*(a+b); w = 0.5*(b-a); z=(x-c)/w; c + w*CUDA.tanh(CUDA.sinh(β*z)/β))
const maxint = Tmax/2
const fwhm = Tmax/2

cusqrt(x::ComplexF64) = CUDA.sqrt(CUDA.abs(x))*CUDA.exp(1.0im*CUDA.angle(x)*0.5)
cusin(x::ComplexF64) = CUDA.imag(CUDA.exp(1im*x))
cucos(x::ComplexF64) = CUDA.real(CUDA.exp(1im*x))
cupow(x::ComplexF64, p::Float64) = CUDA.pow(CUDA.abs(x), p)*CUDA.exp(1.0im*p*CUDA.angle(x))

function airy_asym(y::ComplexF64, n::Int, corr::ComplexF64 = zero(ComplexF64))
    ret = 0.0
    if CUDA.abs(CUDA.angle(y)) < 2π/3
        z = y |> ComplexF64
        ζ = (2/3)*cupow(z, 1.5)
        cf = zero(ComplexF64)
        u = one(Float64)
        for k=0:n
            cf += cupow(-1/ζ, Float64(k)) * u
            u *= (6k+5)*(6k+3)*(6k+1)/((2k+1)*216*(k+1))
        end
        ret = cf*CUDA.exp(-ζ+corr)/2
    else
        z = -y |> ComplexF64
        ζ = (2/3)*cupow(z, 3/2)
        cf_c = zero(ComplexF64)
        cf_s = zero(ComplexF64)
        u = 1.0
        for k=0:n
            if k%2 == 0
                cf_c += CUDA.pow(-1.0, k/2.0) / cupow(ζ, Float64(k)) * u
            else
                cf_s += CUDA.pow(-1.0, (k-1.0)/2.0) / cupow(ζ, Float64(k)) * u
            end
            u *= (6k+5)*(6k+3)*(6k+1)/((2k+1)*216*(k+1))
        end
        ret = CUDA.exp(corr)*(cf_c*cucos(ζ-π/4) + cf_s*cusin(ζ-π/4))
    end
    return ret/(sqrt(π)*cupow(z, 0.25))
end

function airy_taylor(z::ComplexF64, n::Int=40)
    cf = 0
    for k = 0:n
        cf += CUDA.exp(CUDA.lgamma((k+1.0)/3) - CUDA.lgamma(k+1.0))*CUDA.sinpi(2*(k+1.0)/3)*cupow(CUDA.pow(3.0, 1/3)*z, Float64(k))
    end
    return (1.0/(CUDA.pow(3.0, 2/3)*π))*cf
end

function airy_gpu(z::ComplexF64, corr::ComplexF64, n1::Int64=20, n2::Int=1, r::Float64=2.5)
    if CUDA.abs(z) < r
        return airy_taylor(z, n1)*CUDA.exp(corr)
    else
        return airy_asym(z, n2, corr)
    end
end

cusq3(x) = CUDA.sign(x)*CUDA.pow(CUDA.abs(x), 1.0/3)

if mode == "2h"
    EF(t) = 1.0 - smclamp((t-(maxint+fwhm/4))/(fwhm/2), 0.0, 1.0) - smclamp(-(t-(maxint-fwhm/4))/(fwhm/2), 0.0, 1.0)
    A(φ::Float64, t::Float64) = EF(t)*f*(CUDA.sin(ω*t) + q*CUDA.sin(2ω*t+φ)/2)/ω
elseif mode == "cep"
    const F0 = 0.0
    cutf(t) = (t > ξ*T ? 1.0 : CUDA.exp(-(t-ξ*T)^2 /(2(ξ*T/4)^2)))
    function EF(ϵ::Float64, β::Float64, z::Float64)
        α = β*π*(Σ/2)^2
        λ = ComplexF64(ϵ*π*Σ^3)
        Σc = sqrt(Σ^2 + α^2/Σ^2)
        μ = (Σ^2 - 1im*α)*(Σ/Σc)^2
        exparg = cusq3(2/λ)*(z + μ^2 /2λ) |> ComplexF64
        corr = (1/3)*μ^3/λ^2 +z*μ/λ
        if CUDA.abs(corr) < 400
            return 2*sqrt(π)*CUDA.abs(cusqrt(μ/2))*CUDA.pow(abs(2/λ), 1.0/3)*airy_gpu(exparg, corr, 25, 5, 3.5) #*exp(-z^2 /2μ)
        else
            return CUDA.exp(-1im*CUDA.angle(μ)/2)*CUDA.exp(-z^2/(2μ))
        end
    end
    A(ϵ::Float64, β::Float64, φ::Float64, t::Float64) = cutf(t)*cutf(Tmax-t)*f*CUDA.imag(EF(ϵ, β, t-t0)*CUDA.exp(1im*(ω*t+φ)))/ω#+α*ω*(t-μ)^2/4Σ)/ω# + F0*t
end

const σx = ComplexF64.([0 1; 1 0])
const σy = ComplexF64.([0 -1im; 1im 0])
const σz = ComplexF64.([-1 0; 0 1])
const σ = [σx, σy, σz]

const ρ0 = (x = zeros(ComplexF64, z, z); x[1, 1] = 1.0; x) |> cu

const idm = ComplexF64.(Matrix(I, z, z))
const hid = ρ0

const N_pars = 2^18
const ord = 4
const pars_bsl = [4.0, -1.0, 1.5, 1.0, 0.5, 0.5][1:ord]
const pars_spr = [4.0, -5.0, -3.0, -2.0, -1.0, -1.0][1:ord]

@assert length(pars_bsl) == length(pars_spr) == ord
κ = range(-π, stop=π, length=101)

const dips = (x = [0.0, 0.05]; vcat(x, zeros(ord-length(x)))) |> cu
pars_raw = pars_bsl .+ pars_spr.*rand(ord, N_pars)
Threads.@threads for j=1:N_pars
    pars_raw[1, j] -= minimum([sum(pars_raw[i,j]*cos((i-1)*κ) for i=2:ord) for κ=κ])
end
const pars = copy(pars_raw)
if mode == "cep"
    const cep = 2π.*Base.rand(N_pars) |> cu
    const chp = 2.0 .*(-1.0 .+ 2.0 .* CUDA.rand(N_pars)) |> CuArray{Float64}#vcat(1.0 .+ 1.0 .* CUDA.rand(N_pars), -1.0 .- 1.0 .* CUDA.rand(N_pars))[1:2:end]
    const cbp = 1.0 .*(-1.0 .+ 2.0 .* CUDA.rand(N_pars)) |> CuArray{Float64}#(-1.0 .+ 2.0 .* CUDA.rand(N_pars)).*0.5#2.0
else
    const cep = CUDA.zeros(N_pars)
end

ht = zeros(ComplexF64, 2, 2, ord, N_pars)
htr = zeros(ComplexF64, 2, 2, ord, N_pars)

if ssh
    @assert ord == 2
    δ = 0.0 .+ 0.0 .* rand(N_pars)
    h1 = 3.0 .+ 0.0 .* rand(N_pars)
    h2 = 4.0 .+ 0.0 .* rand(N_pars)

    for i=1:N_pars
        ht[:,:,1,i] = δ[i]*σz + h1[i]*σx
        ht[:,:,2,i] = h2[i]*σx
        htr[:,:,2,i] = h2[i]*σy
    end
else
    for i=1:N_pars
        for j=1:ord
            ht[:,:,j,i] = pars[j, i]*σz + (real(dips[j])*σx + imag(dips[j])*σy) /2
            htr[:,:,j,i] = (-imag(dips[j])*σx + real(dips[j])*σy) /2 #switch to one-sided connections!
        end
    end
end

const Ht = cu(ht)
const Htr = cu(htr)

const freqs = (2π/Ts[end]).*collect(0:div(length(Ts), 2))

const ν = 4

function statsimkern!(cbp, chp, cep, js, hts, htr, ::Val{Z}, ::Val{ORD}, ::Val{ν}, exc) where {Z} where {ORD}
    k_id = threadIdx().x
    k_stride = blockDim().x
    φ_id = threadIdx().y
    φ_stride = blockDim().y
    block_id = blockIdx().x
    block_stride = gridDim().x

    ρ = @MMatrix zeros(ComplexF64, Z, Z)
    G = @MMatrix zeros(ComplexF64, Z, Z)
    U = @MMatrix zeros(ComplexF64, Z, Z)
    R = @MMatrix zeros(ComplexF64, Z, Z)
    M = @MMatrix zeros(ComplexF64, Z, Z)
    J = @MMatrix zeros(ComplexF64, Z, Z)
    ρ0 = @MMatrix zeros(ComplexF64, Z, Z)
    @inbounds ρ0[1, 1] = 1.0

    HT = @MArray zeros(ComplexF64, Z, Z, ORD)
    HTR = @MArray zeros(ComplexF64, Z, Z, ORD)

    ID = @MMatrix [(a == b ? 1.0 : 0.0) for a=1:2, b=1:2]

    V = @MVector zeros(ComplexF64, Z)
    VC = @MVector zeros(ComplexF64, Z)

    V[1] = 0.0
    V[2] = 1.0

    if exc
        for i=k_id:k_stride:Nk, j=φ_id:φ_stride:Nf, q=block_id:block_stride:chunk_size
            @inbounds k = K[i]
            @inbounds φ = cep[q] + Φ[j]
            @inbounds α = chp[q]
            @inbounds λ = cbp[q]

            for a=1:Z, b=1:Z, x=1:ORD
                @inbounds HT[a, b, x] = hts[a,b,x,q]
            end

            for a=1:Z, b=1:Z, x=1:ORD
                @inbounds HTR[a, b, x] = htr[a,b,x,q]
            end

            fill!(G, 0f0)
            for x=1:ORD, a=1:Z, b=1:Z
                @inbounds G[a, b] += HT[a,b,x]*CUDA.cos(k*(x-1))
            end
            for x=1:ORD, a=1:Z, b=1:Z
                @inbounds G[a, b] += HTR[a,b,x]*CUDA.sin(k*(x-1))
            end

            eϕ = @inbounds G[2, 1]/CUDA.abs(G[1, 2])
            cθ = @inbounds G[2, 2]/CUDA.sqrt(CUDA.abs2(G[1, 2]) + CUDA.abs2(G[2, 2])) |> real

            cθ2 = CUDA.sqrt((1f0+cθ)*0.5f0)
            sθ2 = CUDA.sqrt((1f0-cθ)*0.5f0)

            @inbounds U[1,1] = -cθ2
            @inbounds U[1,2] = sθ2
            @inbounds U[2,1] = sθ2*eϕ
            @inbounds U[2,2] = cθ2*eϕ

            mul!(M, ρ0, U')
            mul!(ρ, U, M)

            for ti = eachindex(Ts)
                @inbounds t = Ts[ti]

                fill!(J, 0f0)

                fld = A(λ, α, φ, t)

                for x=1:ORD, a=1:Z, b=1:Z
                    @inbounds J[a, b] += -HT[a,b,x]*CUDA.sin((k + fld)*(x-1))*(x-1)
                end
                for x=1:ORD, a=1:Z, b=1:Z
                    @inbounds J[a, b] += HTR[a,b,x]*CUDA.cos((k + fld)*(x-1))*(x-1)
                end

                mul!(M, J, ρ)

                @inbounds @atomic js[ti, j, q] += real(M[1,1] + M[2,2])

                for τ = τs
                    fill!(G, 0f0)

                    fld = A(λ, α, φ, t+τ)

                    for x=1:ORD, a=1:Z, b=1:Z
                        @inbounds G[a, b] += dτ*HT[a,b,x]*CUDA.cos((k + fld)*(x-1))/ν
                    end
                    for x=1:ORD, a=1:Z, b=1:Z
                        @inbounds G[a, b] += dτ*HTR[a,b,x]*CUDA.sin((k + fld)*(x-1))/ν
                    end

                    @inbounds δ = CUDA.abs2(G[2,2])
                    @inbounds ϵ = CUDA.abs2(G[1,2])
                    σ = CUDA.sqrt(δ+ϵ)

                    R .= ID.*CUDA.cos(σ) .+ G .* (1im*CUDA.sin(σ)/σ) #exponentiation procedure. only yields correct results for traceless G

                    eϕ = @inbounds G[2, 1]/CUDA.abs(G[1, 2])
                    cθ = @inbounds G[2, 2]/CUDA.sqrt(CUDA.abs2(G[1, 2]) + CUDA.abs2(G[2, 2])) |> real

                    cθ2 = CUDA.sqrt((1f0+cθ)*0.5f0)
                    sθ2 = CUDA.sqrt((1f0-cθ)*0.5f0)

                    @inbounds U[1,1] = -cθ2#-cθ2
                    @inbounds U[1,2] = sθ2
                    @inbounds U[2,1] = sθ2*eϕ
                    @inbounds U[2,2] = cθ2*eϕ

                    for _=1:ν
                        mul!(J, R', ρ)
                        mul!(ρ, J, R)

                        mul!(J, U', ρ)
                        mul!(M, J, U)

                        for a=1:2
                            @inbounds M[a, 3-a] = 0f0
                        end

                        mul!(J, U, M)
                        mul!(M, J, U')

                        for a=1:2, b=1:2
                            @inbounds ρ[a, b] -= (ρ[a, b] - M[a,b])*dτ/τ2/ν
                        end
                    end
                end
            end
        end
    end

    return nothing
end

const chunk_size = 2048

out_Jωc = CUDA.zeros(ComplexF64, 1+hmax*N_cycles, Nf, chunk_size)
out_Jω = zeros(1+hmax, Nf, N_pars)

const inds = Iterators.partition(1:N_pars, chunk_size)

out_J = CUDA.zeros(Float64, length(Ts), Nf, chunk_size)

out_Js = zeros(Float64, length(Ts), Nf)

@time begin
    @cuda threads=(1, 1) blocks=1 statsimkern!(cbp, chp, cep, out_J, Ht, Htr, Val(2), Val(ord), Val(ν), false);
    Array(out_J)
end

df = Dates.format(now(), "ddmm_HHMM")

rfft_plan = plan_rfft(out_J, 1)

@showprogress for (j, ind)=enumerate(inds)
    fill!(out_J, 0.0)
    @sync @cuda threads=(8, 8) blocks=512 statsimkern!(cbp[ind], chp[ind], cep[ind], out_J, Ht[:,:,:,ind], Htr[:,:,:,ind], Val(2), Val(ord), Val(ν), true);
    (j == 1) && (out_Js .= Array(@view(out_J[:,:,1])))
    mul!(out_Jωc, rfft_plan, out_J)
    @view(out_Jω[:,:,ind]) .= abs.(Array(@view(out_Jωc[1:N_cycles:end, :, :])))
end
println("Simulation complete")

display(size(out_Jω))

using JLD2

prs = Float32.(copy(pars)) |> Array

jldopen("resps-$mode-$df.jld2", true, true, true, IOStream) do io
    if mode == "2h"
        write(io, "data_freq", Float64.(abs.(out_Jω)))
    elseif mode == "cep"
        write(io, "data_freq", Float64.(abs.(out_Jω)))
        write(io, "cep", Array(cep))
        write(io, "cbp", Array(cbp))#
        write(io, "chirp", Array(chp))
        write(io, "out_Js", out_Js)
    end
    write(io, "freqs", freqs)
    write(io, "omega0", ω)
    write(io, "T", Ts)
    write(io, "phi", Φ)###
    write(io, "N", N_cycles)
    if ssh
        write(io, "delta", δ)
        write(io, "h1", h1)
        write(io, "h2", h2)
    else
        write(io, "pars", prs)
    end
end
