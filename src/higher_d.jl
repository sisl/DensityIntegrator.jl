function f_higher_d(pts_arr, params, t)
    @show t
    (; topo, D) = params

    adj_vert = Adjacency{0}(topo)
    cob_vert = Coboundary{0, paramdim(topo)}(topo)
    pts_vec = [Meshes.Point(c...) for c in eachcol(pts_arr)]
    dirs = normalize.(eachcol(pts_arr) .- [mean(eachcol(pts_arr))])

    densities = pdf.([D], eachcol(pts_arr))
    unweighted_step_size = zeros(size(pts_arr, 2))

    debug_flag = fill(false, size(pts_arr, 2)) # check if every vertex is traversed exactly once

    lambdas_ = zeros(size(pts_vec))


    for vidx in Meshes.vertices(topo)
        @assert !debug_flag[vidx]; debug_flag[vidx] = true
        neigh_idxs = cob_vert(vidx)

        neigh_simplices = Meshes.materialize.(Meshes.element.([topo], neigh_idxs), [pts_vec])
        ns = normalize.(normal.(neigh_simplices))
        hvolumes = measure.(neigh_simplices)  # hyper-volumes
        dir = dirs[vidx]

        # TODO: double check if this has to be inverse or no...
        @assert all(norm(n) ≈ 1. for n in ns) norm.(ns)
        @assert norm(dir) ≈ 1.
        cosine_factors = [abs((n'*dir)) for n in ns]
        # cosine_factors = 1
        total_volume = sum( cosine_factors .* hvolumes ./ nvertices.(neigh_simplices))

        unweighted_step_size[vidx] = inv(total_volume * densities[vidx])

        # encourage convexity
        neigh_vert_idxs = adj_vert(vidx)
        p0 = pts_arr[:, neigh_vert_idxs[1]]
        mat = [(pts_arr[:, neigh_vert_idxs[2:end]] .- p0) (-dirs[vidx])]
        normalize!.(eachcol(mat))

        lambdas = (mat'*mat + 1e-5I)\mat'*(pts_arr[:, vidx] - p0)
        lambdas_[vidx] = lambdas[end]
    end
    @assert all(debug_flag)

    lambdas_ ./= (maximum(abs, lambdas_) + 1e-4)

    weights = normalize(densities.^2 .* (1 .+ 7/8 .* lambdas_), 1)
    weighted_step_size = weights .* unweighted_step_size

    res = stack(weighted_step_size .* dirs)
    res
end

function run_higher_d(; p_max=0.1)
    dim = 2 # 3
    pts0 = DensityIntegrator.initialize_points_on_unit_hypersphere(dim, 1000);
    pts = DensityIntegrator.subselect_regular_surface(copy(pts0), 50);
    pts .*= 1e-5;

    myhull = DensityIntegrator.compute_triangulation(copy(pts));

    D = MvNormal(zeros(2), [1. 0.5; 0.5 1.])
    prob = ODEProblem(DensityIntegrator.f_higher_d, stack(pts), (0.0, p_max), (; topo=myhull.topology, D))
    # solve_opts = (; abstol=1e-13)
    solve_opts = (; )
    sol = solve(prob, Vern8(); saveat=p_max/10, solve_opts...)

    fig = Makie.Figure(); ax = Makie.Axis(fig[1,1]);

    for u in sol.u
        Makie.scatter!(ax, [Makie.Point2(col) for col in eachcol(u)])
    end
    xs = -10:0.01:10; ys = xs; zs = [pdf(D, [x;y]) for x in xs, y in ys];
    contour!(ax, xs, ys, zs)

    ax_rhs = Axis(fig[1,2]; aspect=1)
    samples = eachcol(rand(D, 10_000))
    calib_vals = Float64[]; sizehint!(calib_vals, length(sol.u))
    idx = axes(sol.t, 1)[1:end]
    for (i, u, t) in zip(idx, sol.u[idx], sol.t[idx])
        hull_pts = collect(eachcol(u))
        hull = SimpleMesh(Tuple.(eachcol(sol.u[i])), myhull.topology)
        push!(calib_vals, mean([Meshes.myin(Meshes.Point(x...), hull) for x in samples]))
    end
    # @show [sol.t[idx] calib_vals]
    lines!(ax_rhs, [0, 1], [0, 1])
    lines!(ax_rhs, sol.t[idx], calib_vals)


    fig
end

function plot_calibration(D, sol, hull; n_samples=10_000)
    fig = Makie.Figure()
    ax = Axis(fig[1,1])
    plot_calibration!(ax, D, sol, hull; n_samples)
    fig
end

function plot_calibration!(ax, D, sol, hull; n_samples=10_000)
    samples = eachcol(rand(D, n_samples))
    calib_vals = Float64[]; sizehint!(calib_vals, length(sol.u))
    idx = axes(sol.t, 1)[1:end]
    for (i, u, t) in zip(idx, sol.u[idx], sol.t[idx])
        hull_pts = collect(eachcol(u))
        hull_ = SimpleMesh(Tuple.(eachcol(sol.u[i])), hull.topology)
        # push!(calib_vals, mean([Meshes.myin(Meshes.Point(x...), hull) for x in samples]))
        push!(calib_vals, mean(Meshes.myin_vec([Meshes.Point(x...) for x in samples], hull_)))
    end
    # @show [sol.t[idx] calib_vals]
    lines!(ax, [0, 1], [0, 1])
    lines!(ax, sol.t[idx], calib_vals)
end


function run_higher_d_3(; p_max=0.1, n_pts=80)
    dim = 3 # 3
    pts0 = DensityIntegrator.initialize_points_on_unit_hypersphere(dim, 10_000);
    pts = DensityIntegrator.subselect_regular_surface(copy(pts0), 80);
    pts .*= 1e-5;

    myhull = DensityIntegrator.compute_triangulation(copy(pts));


    D = MixtureModel(MvNormal[
            MvNormal(zeros(3), 1/2 .* [1. 0 0.5; 0 1. 0.5; 0.5 0.5 1]),
            MvNormal(ones(3), 1/2 .* [1 0 0; 0 1. 0; 0 0 1]),
            MvNormal([1;-0.5;-1], 1/2 .* [1. 0.5 0; 0.5 1. 0; 0 0 1]),
        ], normalize([
            0.33,
            0.33,
            0.34
        ], 1))
    # D = MvNormal(zeros(3), [1. 0.9 0; 0.9 1. 0; 0 0 1])
    prob = ODEProblem(DensityIntegrator.f_higher_d, stack(pts), (0.0, p_max), (; topo=myhull.topology, D))
    # solve_opts = (; atol=1e-13)
    sol = solve(prob, Vern8(); saveat=p_max/100)
    myhull, sol, D
end

function run_higher_d_any(dim; p_max=0.1, n_pts=80, mesh_init=nothing)
    pts0 = DensityIntegrator.initialize_points_on_unit_hypersphere(dim, 10_000);
    pts = DensityIntegrator.subselect_regular_surface(copy(pts0), n_pts);
    # pts .*= (1e-6)^(1/dim);
    pts .*= 1e-4
    @info "Initialized points"

    myhull = (isnothing(mesh_init) ? DensityIntegrator.compute_triangulation(copy(pts)) : mesh_init)
    @info "Initialized triangulation"


    D = MvNormal(zeros(dim), 1/2 .* I)
    # D = MixtureModel(MvNormal[
    #         MvNormal(zeros(dim), 1/2 .* I),
    #         MvNormal(ones(dim), 1/2 .* I),
    #         MvNormal(rand(dim), 1/2 .* I),
    #     ], normalize([
    #         0.33,
    #         0.33,
    #         0.34
    #     ], 1))
    # D = MvNormal(zeros(3), [1. 0.9 0; 0.9 1. 0; 0 0 1])
    prob = ODEProblem(DensityIntegrator.f_higher_d, stack(pts), (0.0, p_max), (; topo=myhull.topology, D))
    # solve_opts = (; abstol=1e-5, reltol=1e-5)
    solve_opts = (;)
    # solver = BS3();
    solver = Tsit5()
    sol = solve(prob, solver; saveat=p_max/100, progress=true, progress_steps=100, solve_opts...)
    myhull, sol, D
end

# hull, sol, D = DensityIntegrator.run_higher_d_3(; p_max=0.7, n_pts=200);
# DensityIntegrator.plot_sol_3(hull, sol, D; plot_calibration=true)

function plot_sol_3(hull, sol, D; plot_calibration=true)
    fig = Makie.Figure(); ax = Makie.Axis3(fig[1,1], limits=((-2,2),(-2,2),(-2,2)));

    plt_idx = Observable{Int}(1)
    out_mesh = @lift SimpleMesh(Tuple.(eachcol(sol.u[$plt_idx])), hull.topology)
    viz!(ax, out_mesh; alpha=0.8, showfacets=true, transparency=true)

    xs = LinRange(-2, 2, 100)
    ys = LinRange(-2, 2, 100)
    zs = LinRange(-2, 2, 100)
    vals = pdf.([D], [[x;y;z] for x in xs, y in ys, z in zs])
    vplt = volumeslices!(ax, xs, ys, zs, vals; alpha=0.7, transparency=true)
    # vplt[:update_xy][](length(xs)÷2)
    # vplt[:update_xz][](length(ys)÷2)
    # vplt[:update_yz][](length(zs)÷2)
    vplt[:update_xy][](length(xs)÷2)
    vplt[:update_xz][](20)
    vplt[:update_yz][](76)

    if D isa MixtureModel
        scatter!(ax, Makie.Point3.([c.μ for c in D.components]), color=:red, markersize=25)
    end

    sl = Makie.Slider(fig[2,1], range=1:length(sol.u), startvalue=1)
    Makie.connect!(plt_idx, sl.value)

    if plot_calibration
        ax_rhs = Axis(fig[1,2]; aspect=1)
        samples = eachcol(rand(D, 10_000))
        calib_vals = Float64[]; sizehint!(calib_vals, length(sol.u))
        idx = axes(sol.t, 1)[1:end]
        for (i, u, t) in zip(idx, sol.u[idx], sol.t[idx])
            hull_pts = collect(eachcol(u))
            hull_ = SimpleMesh(Tuple.(eachcol(sol.u[i])), hull.topology)
            push!(calib_vals, mean([Meshes.myin(Meshes.Point(x...), hull_) for x in samples]))
        end
        # @show [sol.t[idx] calib_vals]
        lines!(ax_rhs, [0, 1], [0, 1])
        lines!(ax_rhs, sol.t[idx], calib_vals)
    end
    fig
end




# quick test if the `myin` function works...
# julia> dim = 3 # 3
# 3
#
# julia> pts0 = DensityIntegrator.initialize_points_on_unit_hypersphere(dim, 10_000);
#
# julia> pts = DensityIntegrator.subselect_regular_surface(copy(pts0), 50);
#
# julia> pts = DensityIntegrator.subselect_regular_surface(copy(pts0), 300);
#
# julia> myhull = DensityIntegrator.compute_triangulation(copy(pts));
#
# julia> rand(3, 10_000);
#
# julia> samples = rand(3, 10_000);
#
# julia> rand(3, 10_000);^C
#
# julia> mean([Meshes.myin(Meshes.Point(x...), hull) for x in eachcol(samples)])
# 0.0
#
# julia> samples = rand(3, 10_000) .* 2 .- 1;
#
# julia> mean([Meshes.myin(Meshes.Point(x...), hull) for x in eachcol(samples)])
# 0.0
#
# julia> mean([Meshes.myin(Meshes.Point(x...), myhull) for x in eachcol(samples)])
