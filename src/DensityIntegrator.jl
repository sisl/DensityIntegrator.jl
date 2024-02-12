module DensityIntegrator
using StatsBase, Distributions, StaticArrays, LinearAlgebra
using OrdinaryDiffEq
using Random: rand!
using RecursiveArrayTools
using Makie, GLMakie
using RecursiveArrayTools
using Distributions
using LazySets
using ForwardDiff
using Debugger

export compute_means_and_dirs, f, run, plot_res

include("make_points.jl")
include("higher_d.jl")

greet() = print("Hello World!")

function compute_means_and_dirs(pts::AbstractArray; pts_0 = zeros(2))
    dirs = normalize.(pts .- [pts_0])
    dirs
end
# function compute_means_and_dirs(pts_arr::VectorOfArray; pts_0 = zeros(2))
#     pts = pts_arr.u
#     dirs = normalize.(pts .- [pts_0])
#     (; dirs)
# end


function compute_means_and_dirs2(pts; pts_0 = zeros(2))
    ax = axes(pts, 1)
    means = copy(pts)
    dirs = copy(pts)
    for i in ax
        # idx_left  = circshift(ax,  1)[i]
        # idx_right = circshift(ax, -1)[i]
        # means[i] = mean([pts[idx_left], pts[idx_right]])
        # dir_ = let
        #     extended_basis = hcat(pts[idx_right] - pts[idx_left], rand!(similar(pts[i])))
        #     qr(extended_basis).Q[:, end]
        # end
        # dir_ = normalize(dir_)
        # dir_ = dir_ * (dir_'*(pts[i] - pts_0) < 0 ? -1 : 1)
        dir_ = normalize(pts[i] - pts_0)
        dirs[i] = dir_
    end

    (; means, dirs)
end

function f_(pts_mat, params, t)
    # D = MvNormal(zeros(2), 1.0*I(2))
    f(pts_mat, t, params, params[:D])
end

function make_foo(D)
    function foo(p::T) where {T}
        pdf(D, p)
    end
    return foo
end
function get_gs(D, p)::Vector{Float64}
    ForwardDiff.gradient(Base.Fix1(pdf, D), p)
end
function f(pts_mat, t, params, D)
    # D::Mv = params[:D]
    pts = eachcol(pts_mat)
    ax = axes(pts, 1)
    idx_left  = mod1.(ax .- 1, length(ax))
    idx_right = mod1.(ax .+ 1, length(ax))

    densities = pdf.([D], pts)

    areas = zeros(eltype(first(pts)), size(ax))
    for i in ax
        # idx_left  = circshift(ax,  1)[i]
        # idx_right = circshift(ax, -1)[i]
        idx_left_  = mod1(i-1, length(ax))
        idx_right_ = mod1(i+1, length(ax))
        areas[i]   = 1/2 * (  norm(pts[i] - pts[idx_left_])
                            + norm(pts[i] - pts[idx_right_]))
    end
    areas = 2*pi*norm.(pts .- [mean(pts)]) / length(pts)
    # actual_areas = norm.(
    #     pts .- pts[idx_right]
    # )
    areas_lhs = norm.(pts[idx_left] .- pts)
    areas_rhs = norm.(pts[idx_left] .- pts)

    # actual_areas = 1/2 * (
    #     areas_lhs + areas_rhs
    # )
    # actual_areas .*= 1 ./ cos.(angles ./ 2)

    dirs = compute_means_and_dirs(pts; pts_0=mean(pts))
    angles_lhs, angles_rhs = let
        vecs_rhs = normalize.(pts[idx_right] .- pts)
        vecs_lhs = normalize.(pts[idx_left ] .- pts)
        angles_lhs = abs.(abs.(acos.(dot.(vecs_lhs, dirs))) .- pi/2)
        angles_rhs = abs.(abs.(acos.(dot.(vecs_rhs, dirs))) .- pi/2)
        angles_lhs, angles_rhs
    end

    actual_areas = 1/2 * (
        areas_lhs .* 1 ./abs.(cos.(angles_lhs)) +
        areas_rhs .* 1 ./abs.(cos.(angles_rhs))
    )

    # angles = let
    #     vecs_rhs = normalize.(pts[idx_right] .- pts)
    #     vecs_lhs = normalize.(pts[idx_left ] .- pts)
    #     abs.(pi .- acos.(dot.(vecs_rhs, vecs_lhs)))
    # end


    # areas *= sum(actual_areas) / sum(areas)
    areas = actual_areas

    # dirs = compute_means_and_dirs(pts; pts_0=params[:pts_0])
    # @show (densities)
    # @assert all(abs.(densities .- densities[1]) .< 1e-3) "$(densities .- densities[1])" atol=1e-4
    # @assert all(abs.(areas .- areas[1]) .< 1e-3) "Areas: $(areas .- areas[1])" atol=1e-4
    # @show [norm(dirs[i]) for i in 1:length(dirs)]
    # @assert all(norm.(dirs) .≈ 1)
    # @show (areas)
    # @show (densities)
    # @show (1 ./ (areas .* densities))
    # areas = ones(length(pts))
    # slopes = 1 ./ (areas .* (densities .+ 1e-5))
    @assert all(>=(0), areas)
    @assert all(>=(0), densities)
    slopes = 1 ./ (areas .* densities)
    # foo = make_foo(D)
    function foo(p)
        pdf(D, p)
    end
    # gs = [get_gs(D, p) for p in pts]
    # weights = normalize([
    #     inv(
    #         # abs(g' * d)^2  # I'm not sure why this ^2 needs to be here but the results are much better
    #         norm(g)^2
    #     )
    #     for (g, d) in zip(gs, dirs)
    # ], 1)

    # set up extra weights for stabilization
    lambdas = compute_intersection.(pts[idx_left], pts[idx_right], pts, dirs) .|> x->getindex(x, 2)
    sigmoid(x; alpha=1) = (1+exp(-x))^(-alpha)  # https://en.wikipedia.org/wiki/Generalised_logistic_function
    # extra_weights = sigmoid.(1000 .* lambdas; alpha=1)
    # extra_weights = sigmoid.(100 .* lambdas; alpha=3)
    lambdas_ = lambdas / (maximum(abs, lambdas) + 1e-4)
    @assert all(-1 .<= lambdas_ .<= 1)
    # @assert maximum(abs, lambdas_) ≈ (1/(1+1e-4)) " $(maximum(abs, lambdas_)) $(lambdas_)"


    # weights = normalize(densities.^2, 1)
    # weights = 1/2 * (normalize(densities.^2, 1) .+ normalize(abs.(extra_weights), 1))
    weights = normalize(densities.^2 .* (1 .+ 7/8 .* lambdas_), 1)
    # weights = normalize(densities.^2 .* 1, 1)
    # weights = normalize(densities.^2 .* abs.(extra_weights), 1)

    @assert all(isfinite, weights)
    @assert all(isfinite, slopes)
    @assert all(>=(0.), weights) weights
    @assert all(>=(0.), slopes) slopes
    @assert all(>=(0.), weights.*slopes) weights.*slopes

    # weights = ones(size(densities)) / length(pts)
    # slopes .= mean(slopes)
    # @show (slopes)
    # res = stack(weights * length(pts) .* slopes .* dirs)
    res = stack(weights .* slopes .* dirs)
    # res = stack(slopes .* dirs)
    # (; res, densities, areas)
    res
end

function plot_res(res, D)
    fig = Makie.Figure(); ax = Axis(fig[1,1]; aspect=1);
    for (u, t) in zip(res.u, res.t)
        scatter!(ax, [Point2(pt) for pt in eachcol(u)]; label=string(round(t; digits=3)))
    end
    # Legend(fig[1,2])
    axislegend(ax)

    xs = -10:0.01:10; ys = xs; zs = [pdf(D, [x;y]) for x in xs, y in ys];
    contour!(ax, xs, ys, zs)

    ax_rhs = Axis(fig[1,2]; aspect=1)
    samples = eachcol(rand(D, 10_000))
    calib_vals = Float64[]; sizehint!(calib_vals, length(res.u)-1)
    idx = axes(res.t, 1)[2:end]
    for (u, t) in zip(res.u[idx], res.t[idx])
        hull_pts = collect(eachcol(u))
        hull = VPolygon(hull_pts)
        push!(calib_vals, mean([x ∈ hull for x in samples]))
    end
    # @show [res.t[idx] calib_vals]
    lines!(ax_rhs, [0, 1], [0, 1])
    lines!(ax_rhs, res.t[idx], calib_vals)
    fig

end

# check if y0 + lambda d intersects (x0->x1)
function check_intersection(x0, x1, y0, d)
    lambdas = compute_intersection(x0, x1, y0, d)
    return 0 <= lambdas[1] <= 1
end
function compute_intersection(x0, x1, y0, d)
    T = eltype(x0)
    @assert norm(x1-x0) > eps(T)
    M = [(x1 - x0)/norm(x1-x0) -d]
    @assert all(isfinite.(M)) "$(M)"
    # (det(M) ≈ 0) && return false
    lambdas = M \ (y0 - x0)
    return lambdas
    # return NTuple{2, T}(lambdas)
end

function check_point_in_hull(hull_pts, pt)
    dir = [1.;0]
    n_intersects = sum(check_intersection.(
        hull_pts,
        hull_pts[mod1.(axes(hull_pts, 1).+1, length(hull_pts))],
        [pt],
        [[1;0]]
    ))
    return (n_intersects % 2) == 1
end

function run(; pts_0=zeros(2), n=100, p=0.5)
    # D = MvNormal(zeros(2), 1.0*I(2))
    D = MvNormal(zeros(2), [1. 0.5; 0.5 1.])

    # pts_0 = SVector(0., 0.)
    # pts_0 = zeros(2)
    # n = 10
    pts = [pts_0 + 1e-6*[cos(t); sin(t)] for t in LinRange(0, 2*pi, n+1)[1:end-1]]

    prob = ODEProblem(DensityIntegrator.f_, stack(pts), (0.0, p), (; pts_0, D))
    solve_opts = (; atol=1e-13)
    res = solve(prob, AutoTsit5(Rosenbrock23()); solve_opts...);
    # res = solve(prob, Rodas4(autodiff=false));
    # res = solve(prob, Tsit5(); alg_hints=:stiff)
    # res = solve(prob, RadauIIA3(; autodiff=false))
    hull_pts = collect(eachcol(res.u[end]))
    hull = VPolygon(hull_pts)
    # plot_res(res, D)
    # @show mean([x ∈ hull for x in eachcol(rand(D, 10_000))])
    # @show mean([check_point_in_hull(hull_pts, x) for x in eachcol(rand(D, 10_000))])
    (; res, D)
end

function make_widget(; n=100, p=0.5)
    # D = MvNormal(zeros(2), 1.0*I(2))
    # D = MvNormal(zeros(2), [1. 0.5; 0.5 1.])
    D = MixtureModel(MvNormal[
            MvNormal(zeros(2), 0.1.*[1. 0.5; 0.5 1.]),
            MvNormal(ones(2), 0.1.*[1 0; 0 1.]),
            MvNormal([1;-0.5], 0.1.*[1. 0.5; 0.5 1.]),
        ], normalize([
            0.4,
            0.6,
            0.3
        ], 1))


    pts_0=Observable(zeros(2))

    # pts_0 = SVector(0., 0.)
    # pts_0 = zeros(2)
    # n = 10
    pts = @lift [$pts_0 + 1e-6*[cos(t); sin(t)] for t in LinRange(0, 2*pi, n+1)[1:end-1]]

    prob = @lift ODEProblem(DensityIntegrator.f_, stack($pts), (0.0, p), (; $pts_0, D))
    # solve_opts = (; abstol=1e-13, reltol=1e-13)
    solve_opts = (; )
    # solver = AutoVern9(Rodas5P())
    # solver = Vern8(; thread=OrdinaryDiffEq.True())
    # solver = Vern8()
    # solver = Vern8()
    solver = AutoVern9(KenCarp4())
    # solver = AutoTsit5(RK23())
    # solver = Vern8()
    # solver = Feagin12()
    # solver = Vern8()
    # solver = AutoTsit5(Rosenbrock23())
    res = @lift solve($prob, solver; saveat=0.05, solve_opts...);
    # res = solve(prob, Rodas4(autodiff=false));
    # res = solve(prob, Tsit5(); alg_hints=:stiff)
    # res = solve(prob, RadauIIA3(; autodiff=false))
    hull_pts = @lift collect(eachcol($res.u[end]))
    hull = @lift VPolygon($hull_pts)
    # plot_res(res, D)

    # on(hull) do hull
    #     @show mean([x ∈ hull for x in eachcol(rand(D, 10_000))])
    # end

    # @show mean([check_point_in_hull(hull_pts, x) for x in eachcol(rand(D, 10_000))])

    fig = plot_res_interactive(res, D, pts_0)
    (; fig, pts_0)
end

function plot_res_interactive(res, D, pts_0)
    fig = Makie.Figure(); ax = Axis(fig[1,1]; aspect=1, title="Prediction sets for given p");

    each_obs = [
        @lift $res.u[i]
        for i in 1:length(res[].u)
    ]
    for u in each_obs
        u_ = @lift [$u $u[:, 1]]
        lines!(ax, @lift([Makie.Point2(pt) for pt in eachcol($u_)]))
        scatter!(ax, @lift([Makie.Point2(pt) for pt in eachcol($u)]))
    end
    # Legend(fig[1,2])
    # axislegend(ax)

    xs = -10:0.01:10; ys = xs; zs = [pdf(D, [x;y]) for x in xs, y in ys];
    contour!(ax, xs, ys, zs)

    # on(events(fig).mousebutton) do event
    #     if event.button == Mouse.left && event.action == Mouse.press
    #         mp = events(fig).mouseposition[]
    #         pts_0[] = collect(mp)
    #         notify(pts_0)
    #     end
    # end
    register_interaction!(ax, :my_interaction) do event::MouseEvent, axis
        if event.type === MouseEventTypes.leftclick || event.type === MouseEventTypes.leftdrag
            mp = mouseposition(ax)
            pts_0[] = collect(mp)
            notify(pts_0)

            # println("You clicked on the axis!")
        end
    end
    deactivate_interaction!(ax, :rectanglezoom)

    ax_rhs = Axis(fig[1,2]; aspect=1, title="Calibration plot.")
    samples = eachcol(rand(D, 10_000))
    calib_vals = Observable(Float64[]); sizehint!(calib_vals[], length(res[].u)-1)
    idx = axes(res[].t, 1)[2:end]
    on(res) do res
        empty!(calib_vals[])
        for (u, t) in zip(res.u[idx], res.t[idx])
            hull = VPolygon(collect(eachcol(u)))
            push!(calib_vals[], mean([x ∈ hull for x in samples]))
        end
        notify(calib_vals)
    end
    res[] = res[]; notify(res)
    # @show [res.t[idx] calib_vals]
    lines!(ax_rhs, [0, 1], [0, 1])
    lines!(ax_rhs, res[].t[idx], calib_vals; linewidth=5)



    fig

end

end # module DensityIntegrator
