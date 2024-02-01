module DensityIntegrator
using StatsBase, Distributions, StaticArrays, LinearAlgebra
using OrdinaryDiffEq
using Random: rand!
using RecursiveArrayTools
using Makie, GLMakie
using RecursiveArrayTools
using Distributions

export compute_means_and_dirs, f, run, plot_res

greet() = print("Hello World!")

function compute_means_and_dirs(pts_arr::VectorOfArray; pts_0 = zeros(2))
    pts = pts_arr.u
    dirs = normalize.(pts .- [pts_0])
    (; dirs)
end


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

function f_(pts, params, t)
    D = MvNormal(zeros(2), 1.0*I(2))
    f(D, pts, t)
end
function f(D, pts, t)
    ax = axes(pts, 2)
    areas = zeros(size(ax))
    densities = zeros(size(ax))
    for i in ax
        # idx_left  = circshift(ax,  1)[i]
        # idx_right = circshift(ax, -1)[i]
        idx_left = mod1(i-1, length(ax))
        idx_right= mod1(i+1, length(ax))
        areas[i] = 1/2 * (  norm(pts[i] - pts[idx_left])
                          + norm(pts[i] - pts[idx_right]))
        densities[i] = pdf(D, pts[i])
    end
    (; dirs) = compute_means_and_dirs(pts)
    # @show (densities)
    # @show [norm(dirs[i]) for i in 1:length(dirs)]
    # @show (areas)
    # @show (densities)
    # @show (1 ./ (areas .* densities))
    # areas = ones(length(pts))
    slopes = (1 ./ (areas .* densities))
    # @show (slopes)
    res = VectorOfArray([slopes[i] * dirs[i] for i in 1:length(pts)])
    # (; res, densities, areas)
    res
end

function plot_res(res, D)
    fig = Makie.Figure(); ax = Axis(fig[1,1]; aspect=1);
    for u in res.u[1:1:end]
        scatter!(ax, [Point2(u[:, i]) for i in 1:length(u)])
    end

    xs = -10:0.01:10; ys = xs; zs = [pdf(D, [x;y]) for x in xs, y in ys];
    contour!(ax, xs, ys, zs)
    fig
end




function run()
    D = MvNormal(zeros(2), 1.0*I(2))

    # pts_0 = SVector(0., 0.)
    pts_0 = zeros(2)
    n = 10
    pts = [pts_0 + 1e-6*[cos(t); sin(t)] for t in LinRange(0, 2*pi, n)]
    pts_arr = VectorOfArray(pts)

    prob = ODEProblem((pts,params,t) -> DensityIntegrator.f(D, pts, t), pts_arr, (0.0, 0.01))
    res = solve(prob, AutoTsit5(Rosenbrock23()));
    plot_res(res, D)
end

end # module DensityIntegrator
