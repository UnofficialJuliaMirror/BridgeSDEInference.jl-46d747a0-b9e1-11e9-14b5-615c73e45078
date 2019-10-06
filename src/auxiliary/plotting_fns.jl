using PyPlot

function plot_chains(ws::Workspace,indices=nothing;truth=nothing,figsize=(15,10))
    θs = ws.θ_chain.θ_chain
    num_steps = length(θs)
    if indices === nothing
        indices = 1:length(θs[1])
    end
    n_chains = length(indices)

    fig, ax = plt.subplots(n_chains, 1, figsize=figsize, sharex=true)

    for (i,ind) in enumerate(indices)
        ax[i].plot([θ[ind] for θ in θs],color="steelblue", linewidth=0.5)
    end
    if truth !== nothing
        for (i,ind) in enumerate(indices)
            ax[i].plot([1,length(θs)],[truth[ind],truth[ind]],color="black",
                       linestyle="--")
        end
    end
    plt.tight_layout()
    ax
end

function plot_paths(ws::Workspace, coords=nothing; transf=nothing,
                    figsize=(12,8), alpha=0.5, obs=nothing)
    yy, tt = out.paths, out.time
    if coords === nothing
        coords = 1:length(yy[1][1])
    end
    if transf === nothing
        transf = [(x,θ)->x for _ in coords]
    end
    n_coords = length(coords)
    θs = θs_for_transform(ws)

    fig, ax = plt.subplots(n_coords, 1, figsize=figsize, sharex=true)
    cmap = plt.get_cmap("cividis")(range(0, 1, length=length(yy)))
    for (i,ind) in enumerate(coords)
        for j in 1:length(yy)
            path = [transf[i](y, θs[j])[ind] for y in yy[j]]
            ax[i].plot(tt, path, color=cmap[j,1:end], alpha=alpha, linewidth=0.5)
        end
    end

    if obs !== nothing
        for (i,ind) in enumerate(coords)
            if ind in obs.indices
                ax[i].plot(obs.times, obs.vals, mfc="orange", mec="orange",
                           marker="o", linestyle="none", markersize=4)
            end
        end
    end
    plt.tight_layout()
    ax
end

function θs_for_transform(ws::Workspace)
    θs = ws.θ_chain.θ_chain
    if ws.θ_chain.counter < 2
        θs = [params(ws.P[1].Target) for i in 1:length(ws.paths)]
    else
        warm_up, save_iter = ws.action_tracker.warm_up, ws.action_tracker.save_iter
        num_updts = ws.accpt_tracker.updt_len
        N = Int64((length(θs)-1)/num_updts) + warm_up
        θs = θs[[num_updts*(i-warm_up) for i in 1:N
                 if (i % save_iter == 0) && i > warm_up ]]
    end
    θs
end
