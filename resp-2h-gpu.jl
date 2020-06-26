using LinearAlgebra, FFTW, Plots, Dates, ProgressMeter

using CuArrays, CUDAnative, CUDAdrv
using StaticArrays
using UnsafeArrays
using Suppressor

println("$(Threads.nthreads()) threads")
BLAS.set_num_threads(32)

const mode = "cep"

const f = 4.0
const q = 0.1
const ω = 0.8
const T = 2π/ω
const τ1 = Inf#2T
const τ2 = T/2
const N_cycles = 5
const Σ = T*N_cycles/10
const μ = T*N_cycles/2
const Tmax = N_cycles*T
#const φ = 0.0
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

const β = 2.0
smclamp(x, a, b) = (c = 0.5*(a+b); w = 0.5*(b-a); z=(x-c)/w; c + w*CUDAnative.tanh(CUDAnative.sinh(β*z)/β))
const maxint = Tmax/2
const fwhm = Tmax/2

#EF(t) = CUDAnative.exp(-(t-μ)^2/(2Σ^2))

if mode == "2h"
    EF(t) = 1.0 - smclamp((t-(maxint+fwhm/4))/(fwhm/2), 0.0, 1.0) - smclamp(-(t-(maxint-fwhm/4))/(fwhm/2), 0.0, 1.0)
    A(φ::Float64, t::Float64) = EF(t)*f*(CUDAnative.sin(ω*t) + q*CUDAnative.sin(2ω*t+φ)/2)/ω
elseif mode == "cep"
    const F0 = 0.1
    EF(t) = CUDAnative.exp(-(t-μ)^2/(2Σ^2))
    A(φ::Float64, t::Float64) = EF(t)*f*CUDAnative.sin(ω*t+φ)/ω + F0*t
end
@inline F(φ::Float64, t::Float64) = EF(t)*f*(cos(ω*t) + q*cos(2ω*t+φ))

const σx = ComplexF64.([0 1; 1 0])
const σy = ComplexF64.([0 -1im; 1im 0])
const σz = ComplexF64.([-1 0; 0 1])
const σ = [σx, σy, σz]

function gen_U!(U, M, G, V, VC)
    mul!(VC, G, V) #V3 == [0.0, 1.0]

    @inbounds eϕ = VC[1]/CUDAnative.abs(VC[1])
    @inbounds cθ = VC[2]/CUDAnative.sqrt(CUDAnative.abs2(VC[2]) + CUDAnative.abs2(VC[1]))

    cθ2 = CUDAnative.sqrt((1+cθ)/2)
    sθ2 = CUDAnative.sqrt((1-cθ)/2)

    @inbounds U[1,1] = cθ2
    @inbounds U[2,1] = -sθ2*eϕ
    @inbounds U[1,2] = sθ2
    @inbounds U[2,2] = cθ2*eϕ

    return nothing
end

const ρ0 = (x = zeros(ComplexF64, z, z); x[1, 1] = 1.0; x) |> cu

const idm = ComplexF64.(Matrix(I, z, z))
const hid = ρ0

#Nsp = 16
const N_pars = 2^16
#const N_pars = Nsp*Nsp
const ord = 6
const pars_bsl = [4.0, -1.0, 1.5, 1.0, 0.5, 0.5][1:ord]
const pars_spr = [4.0, -5.0, -3.0, -2.0, -1.0, -1.0][1:ord]

@assert length(pars_bsl) == length(pars_spr) == ord
κ = range(-π, stop=π, length=101)

const dips = (x = [0.0, 0.01]; vcat(x, zeros(ord-length(x)))) |> cu
#const pars = (x = reduce(hcat, collect.(collect(Iterators.product(pars_bsl[1] .+ pars_spr[1].*range(0.0, stop=1.0, length=Nsp), pars_bsl[2] .+ pars_spr[2].*range(0.0, stop=1.0, length=Nsp))))); x[1,:] .-= x[2,:]; x)
pars_raw = pars_bsl .+ pars_spr.*rand(ord, N_pars)
Threads.@threads for j=1:N_pars
    pars_raw[1, j] -= minimum([sum(pars_raw[i,j]*cos((i-1)*κ) for i=2:ord) for κ=κ])
end
const pars = copy(pars_raw)
if mode == "cep"
    const cep = 2π.*Base.rand(N_pars) |> cu
else
    const cep = CuArrays.zeros(N_pars)
end

ht = zeros(2, 2, ord, N_pars)
for i=1:N_pars
    for j=1:ord
        ht[:,:,j,i] = pars[j, i]*σz + real(dips[j])*σx + imag(dips[j])*σy
    end
end

const Ht = cu(ht)

const freqs = (2π/Ts[end]).*collect(0:div(length(Ts), 2))

function gpu_copy!(dest, src)
    for a=1:2, b=1:2
        @inbounds dest[a,b] = src[a,b]
    end

    return nothing
end

function gen_G!(G, M, ht_par, κ, ord)#, ::Val{ord}) where {ord}
    lmul!(0f0, G)

    for x=1:ord
        for a=1:2, b=1:2
            @inbounds M[a, b] = ht_par[a,b,x]
        end
        lmul!(CUDAnative.cos(κ*(x-1)), M)
        for a=1:2, b=1:2
            @inbounds G[a, b] += M[a, b]
        end
    end

    return nothing
end

function gen_J!(J, M, ht_par, κ, ord)
    lmul!(0f0, J)

    for x=1:ord
        for a=1:2, b=1:2
            @inbounds M[a, b] = ht_par[a,b,x]
        end
        lmul!(-CUDAnative.sin(κ*(x-1))*(x-1), M)
        for a=1:2, b=1:2
            @inbounds J[a, b] += M[a, b]
        end
    end
    return nothing
end

function write_tr!(dest, src, inds)
    for a=1:2
        @inbounds dest[inds...] += real(src[a, a])
    end
    return nothing
end

function pow!(M, R, J, ν)
    for a=1:2, b=1:2
        @inbounds J[a, b] = R[a, b]
    end

    @inbounds mul!(M, R, J)
    for a=1:2, b=1:2
        @inbounds J[a, b] = M[a, b]
    end

    for a=1:2, b=1:2
        @inbounds R[a, b] = J[a, b]
    end

    return nothing
end

const ν = 4

function statsimkern!(cep, js, hts, ::Val{Z}, ::Val{ORD}, ::Val{ν}, exc) where {Z} where {ORD}
    #i = threadIdx().x
    #j = threadIdx().y
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
    ID = @MMatrix [(a == b ? 1.0 : 0.0) for a=1:2, b=1:2]

    V = @MVector zeros(ComplexF64, Z)
    VC = @MVector zeros(ComplexF64, Z)

    V[1] = 0.0
    V[2] = 1.0

    if exc
        (k_id == 1 && φ_id == 1 && block_id == 1) && @cuprintf("Memory allocated\n")
        for i=k_id:k_stride:Nk, j=φ_id:φ_stride:Nf, q=block_id:block_stride:N_pars
            @inbounds k = K[i]
            @inbounds φ = cep[q] + Φ[j]

            for a=1:Z, b=1:Z, x=1:ORD
                @inbounds HT[a, b, x] = hts[a,b,x,q]
            end

            fill!(G, 0f0)
            for x=1:ORD, a=1:Z, b=1:Z
                @inbounds G[a, b] += HT[a,b,x]*CUDAnative.cos(k*(x-1))
            end

            eϕ = @inbounds G[2, 1]/CUDAnative.abs(G[1, 2])
            cθ = @inbounds G[2, 2]/CUDAnative.sqrt(CUDAnative.abs2(G[1, 2]) + CUDAnative.abs2(G[2, 2])) |> real

            cθ2 = CUDAnative.sqrt((1f0+cθ)*0.5f0)
            sθ2 = CUDAnative.sqrt((1f0-cθ)*0.5f0)

            @inbounds U[1,1] = -cθ2#-cθ2
            @inbounds U[1,2] = sθ2
            @inbounds U[2,1] = sθ2*eϕ
            @inbounds U[2,2] = cθ2*eϕ

            mul!(M, ρ0, U')
            mul!(ρ, U, M)

            for ti = eachindex(Ts)
                @inbounds t = Ts[ti]

                fill!(J, 0f0)

                for x=1:ORD, a=1:Z, b=1:Z
                    @inbounds J[a, b] += -HT[a,b,x]*CUDAnative.sin((k + A(φ, t))*(x-1))*(x-1)
                end

                mul!(M, J, ρ)

                @inbounds @atomic js[ti, j, q] += real(M[1,1] + M[2,2])

                for τ = τs
                    fill!(G, 0f0)

                    for x=1:ORD, a=1:Z, b=1:Z
                        @inbounds G[a, b] += dτ*HT[a,b,x]*CUDAnative.cos((k + A(φ, t+τ))*(x-1))/ν
                    end

                    @inbounds δ = CUDAnative.abs2(G[2,2])
                    @inbounds ϵ = CUDAnative.abs2(G[1,2])
                    σ = CUDAnative.sqrt(δ+ϵ)

                    R .= ID.*CUDAnative.cos(σ) .+ G .* (1im*CUDAnative.sin(σ)/σ)

                    if false && τ == 0.0 && i == 1 && j == 1 && q == 1
                        @cuprintf("%lf\n", k+A(φ, t+τ))
                        for a=1:2, b=1:2
                            @inbounds @cuprintf("%lf ", real(G[a, b])*ν/dτ)
                        end
                        @cuprintf("\n")
                        for a=1:2, b=1:2
                            @inbounds @cuprintf("%lf ", real(ρ[a, b]))
                        end
                        @cuprintf("\n")
                        @cuprintf("\n")
                    end

                    eϕ = @inbounds G[2, 1]/CUDAnative.abs(G[1, 2])
                    cθ = @inbounds G[2, 2]/CUDAnative.sqrt(CUDAnative.abs2(G[1, 2]) + CUDAnative.abs2(G[2, 2])) |> real

                    cθ2 = CUDAnative.sqrt((1f0+cθ)*0.5f0)
                    sθ2 = CUDAnative.sqrt((1f0-cθ)*0.5f0)

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
                            @inbounds M[a, 3-a] = 0f0 #fuck this line of code in particular
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

out = zeros(ComplexF64, length(freqs), Nf, N_pars)
out_J = CuArrays.zeros(Float64, length(Ts), Nf, N_pars)

@time begin
    @cuda threads=(1, 1) blocks=1 statsimkern!(cep, out_J, Ht, Val(2), Val(ord), Val(ν), false);
    Array(out_J)
end

out_J .= 0.0

df = Dates.format(now(), "ddmm_HHMM")

@time begin
    @sync @cuda threads=(8, 8) blocks=512 statsimkern!(cep, out_J, Ht, Val(2), Val(ord), Val(ν), true);
    Array(Ht)
end

println("Simulation complete")# abs.(rfft(cmp, 1)) |> Array

const chunk_size = Int(round(sqrt(N_pars)))
const inds = Iterators.partition(1:N_pars, chunk_size)

out_Jω = zeros(1+hmax*N_cycles, Nf, N_pars)
proc_J = CuArrays.zeros(length(Ts), Nf, chunk_size)

if mode == "2h"
    @showprogress for ind=inds
        @view(out_Jω[:,:,ind]) .= abs.(Array(rfft(@view(out_J[:,:,ind]), 1)))# |> Array
    end
elseif mode == "cep"
    @showprogress for ind=inds
        @view(out_Jω[:,:,ind]) .= abs.(Array(rfft(@view(out_J[:,:,ind]), 1)))# |> Array
    end

    out_Jωl = rfft(out_Jω, 2)
end

println("RFFT done")

display(size(out_Jω))

#using BSON
#using BSON: @save, @load

#@save "resps-$df.bson" out_Jω pars

using JLD2

prs = Float32.(copy(pars)) |> Array

jldopen("resps-$mode-$df.jld2", true, true, true, IOStream) do io
    #write(io, "data", Array(Float64.(out_J)))
    if mode == "2h"
        write(io, "data_freq", Float64.(real.(out_Jω))[1:N_cycles:end, :, :])
    elseif mode == "cep"
        write(io, "data_freq", Float64.(abs.(out_Jω))[1:N_cycles:end, :, :])
        #write(io, "data_angfreq", Float64.(abs.(out_Jωl))[1:N_cycles:end, :, :]) #WARNING: ROTATE BEFORE USE
        #write(io, "data_angfreq_cos", Float64.(real.(out_Jωl))[1:N_cycles:end, :, :])
        #write(io, "data_angfreq_sin", Float64.(imag.(out_Jωl))[1:N_cycles:end, :, :])
        write(io, "cep", Array(cep))
    end
    write(io, "freqs", freqs)
    write(io, "omega0", ω)
    write(io, "T", Ts)
    write(io, "phi", Φ)
    write(io, "N", N_cycles)
    write(io, "pars", prs)
end
