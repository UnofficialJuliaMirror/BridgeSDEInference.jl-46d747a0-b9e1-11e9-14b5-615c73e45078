
struct EmptyLadder end

struct BiasedPr{Tp,Tbp}
    prior::Tp
    bPrior::Tbp

    function BiasedPr(priors::Tp, bPriors::Tbp) where {Tp,Tbp}
        new{Tp, Tbp}(priors, bPriors)
    end
end

struct SimTempLadder{Tl,Tc,TP,TXX}
    κ::Int64
    ladder::Tl
    c::Tc
    P::TP
    XX::TXX
    count::Array{Int64,2}
    accpt::Array{Int64,2}
    m::Int64

    function SimTempLadder(ladder::Tl, c::Tc, P::TP, XX::TXX
                           ) where {Tl,Tc,TP,TXX}
        κ = length(ladder)
        new{Tl,Tc,TP,TXX}(κ, ladder, c, deepcopy(P), deepcopy(XX),
                          fill(0, (κ, κ)), fill(0, (κ, κ)), length(XX))
    end
end

struct SimTempPrLadder{Tl, Tc}
    κ::Float64
    ladder::Tl
    c::Tc
    count::Array{Int64,2}
    accpt::Array{Int64,2}

    function SimTempPrLadder(ladder::Tl, c::Tc) where {Tl,Tc}
        κ = length(ladder)
        new{Tl,Tc}(κ, ladder, c, fill(0, (κ, κ)), fill(0, (κ, κ)))
    end
end

struct ParTempLadder{Tl,TP,TXX}
    κ::Float64
    ladder::Tl
    Ps::TP
    XXs::TXX
    count::Array{Int64,2}
    accpt::Array{Int64,2}
    m::Int64

    function ParTempLadder(ladder::Tl, Ps::TP, XXs::TXX) where {Tl,TP,TXX}
        κ = length(ladder)
        new{Tl,TP,TXX}(κ, ladder, deepcopy(Ps), deepcopy(XXs),
                       fill(0, (κ, κ)), fill(0, (κ, κ)), length(XXs[1]))
    end
end

struct ParTempPrLadder{Tl}
    κ::Float64
    ladder::Tl
    count::Array{Int64,2}
    accpt::Array{Int64,2}

    function ParTempPrLadder(ladder::Tl) where Tl
        κ = length(ladder)
        new{Tl}(κ, ladder, fill(0, (κ, κ)), fill(0, (κ, κ)))
    end
end

Ladders = Union{SimTempLadder,ParTempLadder,SimTempPrLadder,ParTempPrLadder}
Non𝓣Ladders = Union{EmptyLadder,BiasedPr,SimTempPrLadder,ParTempPrLadder}
𝓣Ladders = Union{SimTempLadder,ParTempLadder}
SimLadders = Union{SimTempLadder,SimTempPrLadder}

accptRate(::T) where T = NaN
accptRate(ℒ::T) where T <: Ladders = ℒ.accpt ./ ℒ.count

𝓣ladder(::T, i) where T <: Non𝓣Ladders = NaN
𝓣ladder(ℒ::T, i) where T <: 𝓣Ladders = ℒ.ladder[i]

function llikelihood!(ℒ::T, θ, y, WW, P, XX, ι, ::ST=Ralston3()
                      ) where {T <: 𝓣Ladders, ST}
    for i in 1:ℒ.m
        P[i] = GuidPropBridge(P[i], θ, ℒ.ladder[ι])
    end
    solveBackRec!(ℒ.P, ST())
    y₀ = copy(y)
    for i in 1:m
        solve!(Euler(), XX[i], y₀, WW[i], P[i])
        y₀ = XX[i].yy[end]
    end
    llᵒ = 0.0
    for i in 1:m
        llᵒ += llikelihood(LeftRule(), XX[i], P[i])
    end
    llᵒ
end

function computeLogWeight!(ℒ::BiasedPr, θ)
    logWeight = 0.0
    for (prior, bprior) in zip(ℒ.prior, ℒ.bPrior)
        logWeight = logpdf(prior, θ) - logpdf(bprior, θ)
    end
    logWeight
end

function computeLogWeight!(ℒ::SimTempLadder, θ, y, WW, ι, ll, ::ST=Ralston3()) where ST
    ι == 1 && return 0.0
    ll₁ = llikelihood!(ℒ, θ, y, WW, ℒ.P, ℒ.XX, 1, ST())
    ll₁ - ll
end

function computeLogWeight(ℒ::SimTempPrLadder, θ, ι)
    ι == 1 && return 0.0
    logWeight = 0.0
    for (prior, priorᵒ) in zip(ℒ.ladder[1], ℒ.ladder[ι])
        logWeight = logpdf(prior, θ) - logpdf(priorᵒ, θ)
    end
    logWeight
end

function computeLogWeight!(ℒ::ParTempLadder, θ, y, WW, ι, idx, ll, ::ST=Ralston3()) where ST
    ι == 1 && return 0.0
    ll₁ = llikelihood!(ℒ, θ, y, WW, ℒ.Ps[idx], ℒ.XXs[idx], 1, ST())
    ll₁ - ll
end

function computeLogWeight(ℒ::ParTempPrLadder, θs, ι, idx)
    idx == 1 && return 0.0
    logWeight = 0.0
    for (prior, priorᵒ) in zip(ℒ.ladder[1], ℒ.ladder[idx])
        logWeight = logpdf(prior, θs[ι[idx]]) - logpdf(priorᵒ, θs[ι[idx]])
    end
    logWeight
end

function update!(ℒ::SimTempLadder, θ, y, WW, ι, ll, ::ST=Ralston3();
                 verbose=false, it=NaN) where ST
    ιᵒ = rand([max(ι-1, 1), min(ι+1, ℒ.κ)])
    ℒ.count[ι, ιᵒ] += 1
    if ιᵒ == ι
        llr = 0.0
    else
        llᵒ = llikelihood!(ℒ, θ, y, WW, ιᵒ, ST())
        llr = llᵒ + log(ℒ.c[ιᵒ]) - ll - log(ℒ.c[ι])
    end
    verbose && print("prior index update: ", it, " diff_ll: ",
                     round(llr, digits=3))
    if acceptSample(llr, verbose)
        ℒ.accpt[ι, ιᵒ] += 1
        return ιᵒ
    else
        return ι
    end
end

function update!(ℒ::SimTempPrLadder, θ, ι, ::ST=Ralston3(); verbose=false,
                 it=NaN) where ST
    ιᵒ = rand([max(ι-1, 1), min(ι+1, κ)])
    ℒ.count[ι, ιᵒ] += 1
    if ιᵒ == ι
        llr = 0.0
    else
        llr = log(ℒ.c[ιᵒ]) - log(ℒ.c[ι])
        for (prior, priorᵒ) in zip(ℒ.ladder[ι], ℒ.ladder[ιᵒ])
            llr += logpdf(priorᵒ, θ) - logpdf(prior, θ)
        end
    end
    verbose && print("prior index update: ", it, " diff_ll: ",
                     round(llr, digits=3))
    if acceptSample(llr, verbose)
        ℒ.accpt[ι, ιᵒ] += 1
        return ιᵒ
    else
        return ι
    end
end

function update!(ℒ::ParTempLadder, θs, ys, WWs, ι, lls, ::ST=Ralston3();
                 verbose=false, it=NaN) where ST
    idx = rand(1:length(ι)-1)
    ιᵒ = copy(ι)
    ιᵒ[idx], ιᵒ[idx+1] = ιᵒ[idx+1], ιᵒ[idx]
    ℒ.count[idx, idx+1] += 1
    llᵒ = ( llikelihood!(ℒ, θs[ιᵒ[idx]], ys[ιᵒ[idx]], WWs[ιᵒ[idx]], idx, ST())
            + llikelihood!(ℒ, θs[ιᵒ[idx]+1], ys[ιᵒ[idx]], WWs[ιᵒ[idx]+1], idx+1,
                           ST()) )
    llr = llᵒ - lls[idx] - lls[idx+1]

    verbose && print("prior index update: ", it, " diff_ll: ",
                     round(llr, digits=3))
    if acceptSample(llr, verbose)
        ℒ.accpt[idx, idx+1] += 1
        return ιᵒ
    else
        return ι
    end
end

function update!(ℒ::SimTempPrLadder, θs, ι, ::ST=Ralston3(); verbose=false,
                 it=NaN) where ST
    idx = rand(1:length(ι)-1)
    ιᵒ = copy(ι)
    ιᵒ[idx], ιᵒ[idx+1] = ιᵒ[idx+1], ιᵒ[idx]

    ℒ.count[idx, idx+1] += 1
    llr = 0.0
    for (prior, priorNext) in zip(ℒ.ladder[idx], ℒ.ladder[idx+1])
        llr += ( logpdf(prior, θs[ιᵒ[idx]]) + logpdf(priorNext, θs[ιᵒ[idx]+1])
                 - logpdf(prior, θs[ι[idx]]) + logpdf(priorNext, θs[ι[idx]+1]) )
    end

    verbose && print("prior index update: ", it, " diff_ll: ",
                     round(llr, digits=3))
    if acceptSample(llr, verbose)
        ℒ.accpt[idx, idx+1] += 1
        return ιᵒ
    else
        return ι
    end
end
