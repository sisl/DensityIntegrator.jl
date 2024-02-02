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
    densities = pdf.([D], pts)

    areas = zeros(size(ax))
    for i in ax
        # idx_left  = circshift(ax,  1)[i]
        # idx_right = circshift(ax, -1)[i]
        idx_left = mod1(i-1, length(ax))
        idx_right= mod1(i+1, length(ax))
        areas[i] = 1/2 * (  norm(pts[i] - pts[idx_left])
                          + norm(pts[i] - pts[idx_right]))
    end
    areas = 2*pi*norm.(pts .- [mean(pts)]) / length(pts)
    actual_areas = norm.(
        pts .- pts[mod1.(axes(pts, 1) .+1, length(pts))]
    )
    areas *= sum(actual_areas) / sum(areas)
    # areas = actual_areas

    # dirs = compute_means_and_dirs(pts; pts_0=params[:pts_0])
    dirs = compute_means_and_dirs(pts; pts_0=mean(pts))
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
    weights = normalize(densities.^3, 1)
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
    @bp
    fig

end

# check if y0 + lambda d intersects (x0->x1)
function check_intersection(x0, x1, y0, d)
    M = [(x1 - x0)/norm(x1-x0) -d]
    (det(M) ≈ 0) && return false
    lambdas = M \ (y0 - x0)
    return 0 <= lambdas[1] <= 1
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
    res = solve(prob, AutoTsit5(Rosenbrock23()));
    # res = solve(prob, Rodas4(autodiff=false));
    # res = solve(prob, Tsit5(); alg_hints=:stiff)
    # res = solve(prob, RadauIIA3(; autodiff=false))
    hull_pts = collect(eachcol(res.u[end]))
    hull = VPolygon(hull_pts)
    # plot_res(res, D)
    @show mean([x ∈ hull for x in eachcol(rand(D, 10_000))])
    # @show mean([check_point_in_hull(hull_pts, x) for x in eachcol(rand(D, 10_000))])
    (; res, D)
end

end # module DensityIntegrator
