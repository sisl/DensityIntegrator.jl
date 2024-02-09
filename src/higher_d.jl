function f_higher_d(pts_arr, params, t)
    (; topo, D) = params

    adj_vert = Adjacency{0}(topo)
    cob_vert = Coboundary{0, paramdim(topo)}(topo)
    pts_vec = [Meshes.Point(c...) for c in eachcol(pts_arr)]
    dirs = eachcol(pts_arr) .- [mean(eachcol(pts_arr))]

    densities = pdf.([D], eachcol(pts_arr))
    unweighted_step_size = zeros(size(pts_vec))

    lambdas_ = zeros(size(pts_vec))


    for vidx in Meshes.vertices(topo)
        neigh_idxs = cob_vert(vidx)

        neigh_simplices = Meshes.materialize.(topo.simplicies[neigh_idxs], [pts_vec])
        ns = normal.(neigh_simplices)
        hvolumes = measure.(neigh_simplices)  # hyper-volumes

        total_volume = sum([abs(n'*dirs[vidx]) for n in ns] .* hvolumes./(nvertices.(neigh_simplices)))

        unweighted_step_size[vidx] = inv(total_volume * densities[vidx])

        # encourage convexity
        neigh_vert_idxs = adj_vert(vidx)
        p0 = pts_arr[:, neigh_vert_idxs[1]]
        mat = [(pts_arr[:, neigh_vert_idxs[2:end]] .- p0) (-dirs[vidx])]
        normalize!.(eachcol(mat))

        lambdas = pinv(mat) * (pts_arr[:, vidx] - p0)
        lambdas_[vidx] = lambdas[end]
    end

    lambdas_ = lambdas_ / (maximum(abs, lambdas_) + 1e-4)

    weights = normalize(densities.^2 .* (1 .+ 7/8 .* lambdas_), 1)
    weighted_step_size = weights .* unweighted_step_size

    res = stack(weighted_step_size.* dirs)
    res
end

function run_higher_d(; p_max=0.1)
    dim = 2 # 3
    pts0 = DensityIntegrator.initialize_points_on_unit_hypersphere(dim, 10_000);
    pts = DensityIntegrator.subselect_regular_surface(copy(pts0), 50);
    pts .*= 1e-5;

    myhull = DensityIntegrator.compute_triangulation(copy(pts));

    D = MvNormal(zeros(2), [1. 0.5; 0.5 1.])
    prob = ODEProblem(DensityIntegrator.f_higher_d, stack(pts), (0.0, p_max), (; topo=myhull.topology, D))
    # solve_opts = (; atol=1e-13)
    sol = solve(prob, Tsit5(); saveat=p_max/10)

    fig = Makie.Figure(); ax = Makie.Axis(fig[1,1]);

    for u in sol.u
        Makie.scatter!(ax, [Makie.Point2(col) for col in eachcol(u)])
    end; fig
end
