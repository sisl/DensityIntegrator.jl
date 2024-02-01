module DensityIntegrator
using StatsBase, Distributions, StaticArrays, LinearAlgebra
using OrdinaryDiffEq
using Random: rand!
using RecursiveArrayTools
using Makie, GLMakie
using RecursiveArrayTools
using Distributions
using LazySets

export compute_means_and_dirs, f, run, plot_res

greet() = print("Hello World!")

function compute_means_and_dirs(pts::AbstractArray; pts_0 = zeros(2))
    dirs = normalize.(pts .- [pts_0])
    (; dirs)
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
    f(pts_mat, t, params)
end
function f(pts_mat, t, params)
    D = params[:D]
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
    (; dirs) = compute_means_and_dirs(pts; pts_0=params[:pts_0])
    # @show (densities)
    # @assert all(abs.(densities .- densities[1]) .< 1e-3) "$(densities .- densities[1])" atol=1e-4
    # @assert all(abs.(areas .- areas[1]) .< 1e-3) "Areas: $(areas .- areas[1])" atol=1e-4
    # @show [norm(dirs[i]) for i in 1:length(dirs)]
    # @assert all(norm.(dirs) .≈ 1)
    # @show (areas)
    # @show (densities)
    # @show (1 ./ (areas .* densities))
    # areas = ones(length(pts))
    slopes = (1 ./ (areas .* max.(densities, 1e-5) .* length(pts)))
    # slopes .= mean(slopes)
    # @show (slopes)
    res = stack(slopes .* dirs)
    # (; res, densities, areas)
    res
end

function plot_res(res, D)
    fig = Makie.Figure(); ax = Axis(fig[1,1]; aspect=1);
    for u in res.u
        scatter!(ax, [Point2(pt) for pt in eachcol(u)])
    end

    xs = -10:0.01:10; ys = xs; zs = [pdf(D, [x;y]) for x in xs, y in ys];
    contour!(ax, xs, ys, zs)
    fig
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
    hull = VPolygon(collect(eachcol(res.u[end])))
    # plot_res(res, D)
    @show mean([x ∈ hull for x in eachcol(rand(D, 10_000))])
    (; res, D)
end

end # module DensityIntegrator
