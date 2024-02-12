using Meshes
import Makie as Mke
using GLMakie
using Distances
using LinearAlgebra
using StatsBase
import GeometryBasics as GB
using MiniQhull

# dim = 3
#
#
# Mke.scatter(Mke.Point{3, Float64}.(eachcol(foo[1:3, :])))
#
#
# if dim <= 3
    # viz(used_pts; pointsize=10)
# else
    # fig, ax = Mke.scatter([Mke.Point{3, Float64}(p.coords[1:3]) for p in used_pts];
                # color=[p.coords[4] for p in used_pts],
                # colorbar=true)
    # # Colorbar(fig[1,2], limits=(-1.,1.),
    # #          colormap = cgrad(:Spectral, 5, categorical = true),
    # #          size = 25)
    # fig
# end
#
# used_rhs = stack(Vector(getfield.(used_pts, :coords)))
# used_distances = pairwise(Euclidean(), used_rhs)
#
# connection_tpls = []
# for i in axes(used_distances, 1)
    # neighs = sortperm(used_distances[i, :])[1:(dim+2)]
    # for j in 2:(dim+1)  # always contain 1, i.e. self
        # push!(connection_tpls, Tuple(setdiff(neighs, j)))
    # end
# end
# topo = connect.(connection_tpls)
#
# mesh = SimpleMesh(used_pts, topo) |> Repair{0}()
#
# if dim <= 3
    # fig = viz(mesh, pointsize=20)
    # # viz!(used_pts, pointsize=20, color=:red, alpha=0.3)
    # fig
# end

# connec = connect.([Tuple(search(used_pts[i], KNearestSearch(used_pts, 4))) for i in eachindex(used_pts)])

initialize_points_on_unit_hypersphere(dim, n) = initialize_points_on_unit_hypersphere(Float64, dim, n)
function initialize_points_on_unit_hypersphere(T::Type{X}, dim, n) where {X<:Real}
    pts_in_unit_cube = 2*rand(T, dim, n) .- 1;
    pts_in_unit_hypersphere = let
        idx_in_unit_hypersphere = norm.(eachcol(pts_in_unit_cube .- mean(pts_in_unit_cube; dims=2))) .<= 1
        pts_in_unit_cube[:, idx_in_unit_hypersphere]
    end
    normalize!.(eachcol(pts_in_unit_hypersphere));
    pts_in_unit_hypersphere = GB.Point{dim, T}.(eachcol(pts_in_unit_hypersphere))
    return pts_in_unit_hypersphere
end

function subselect_regular_surface(potential_pts::AbstractVector{GB.Point{Dim, T}}, n) where {Dim, T}
    @assert length(potential_pts) > n

    used_pts = GB.Point{Dim, T}[]; sizehint!(used_pts, n)
    push!(used_pts, potential_pts[1])

    lhs = reshape(reinterpret(T, potential_pts[1:1]), Dim, :)
    rhs = reshape(reinterpret(T, potential_pts), Dim, :)

    min_distances = pairwise(Euclidean(), lhs, rhs)

    for _ in 1:n-1
        val, idx = findmax(min_distances[1,:])
        push!(used_pts, potential_pts[idx])
        lhs = reshape(reinterpret(T, potential_pts[idx:idx]), Dim, :)
        min_distances = min.(min_distances, pairwise(Euclidean(), lhs, rhs))
    end

    return used_pts
end

# compute_triangulation(pts::AbstractVector{GB.Point{Dim, T}}) where {Dim, T} =
#     compute_triangulation(HalfEdgeTopology, pts)

function compute_triangulation(pts::AbstractVector{GB.Point{Dim, T}}) where {Dim, T}
    push!(pts, GB.Point{Dim, T}(zeros(T, Dim)))
    # qhull delunay Qbb=more stability Qt=triangulate everything to simplex
    qhull_flags = "qhull d Qbb QV$(length(pts)-1) QJ Pg Qg"  # maybe Qz for "cospherical"?
    # qhull_flags = "qhull d Qbb QV$(0) QJ Qu"  # maybe Qz for "cospherical"?
    # qhull_flags = "qhull d Qbb QV$(0) QJ"  # maybe Qz for "cospherical"?
    simplices = delaunay(pts, qhull_flags)
    # remove center point (and therefore remove one dimension)
    # simplices = sortslices(simplices; dims=2)  # this doesn't work as I would expect.....
    for col in eachcol(simplices)
        sort!(col)
    end
    @assert all(issorted, eachcol(simplices))
    @assert all(==(length(pts)), simplices[end, :]) "$(simplices[end, :])"
    # @assert all(==(0), simplices[1, :]) "$(simplices[1, :])"

    # simplices = simplices[:, simplices[end, :] .== length(pts)]

    simplices = simplices[1:Dim, :]
    connecs = [connect(Tuple(col), Simplex{Dim-1})
               for col in eachcol(simplices)]
    topo = IndAdjTopology(connecs)


    # topo = TopoType(connect.((eachcol(simplices))))
    # topo = SimpleTopology(connect.(Tuple.(eachcol(simplices))));
    # return SimpleMesh(Meshes.Point{Dim, T}.(pts[1:end-1]), topo)
    return SimpleMesh(Meshes.Point{Dim, T}.(pts[1:end-1]), topo)
end
"""
Usage example:
```julia
dim = 2 # 3
pts0 = DensityIntegrator.initialize_points_on_unit_hypersphere(dim, 10_000);
pts1 = DensityIntegrator.subselect_regular_surface(copy(pts0), 500);
# without viz
myhull = DensityIntegrator.compute_triangulation(copy(pts1));
# adj = Adjacency{0}(myhull)
# adj(1)
# or
myhull = DensityIntegrator.compute_triangulation(SimpleTopology,copy(pts1));
fig = viz(myhull; showfacets=true, alpha=0.8)
viz!(myhull.vertices; pointsize=20)
fig
```
"""


"""
Some more examples.
```julia
julia> import Makie
julia> import Makie: Observable, lift

julia> myhull = DensityIntegrator.compute_triangulation(copy(pts1));

julia> adj = Adjacency{0}(topology(myhull))
julia> cob = Coboundary{0, paramdim(myhull)}(topology(myhull))

julia> base_idx = Observable(4)
julia> cs = lift(base_idx) do base_idx
         cs = fill(:red, nvertices(myhull));
         cs[base_idx] = :orange
         cs[adj(base_idx)] .= :white
         cs
       end
julia> cs_facet = lift(base_idx) do base_idx
         cs = fill(:white, nelements(topology(myhull)))
         # cs[base_idx] = :orange
         cs[cob(base_idx)] .= :red
         cs
       end

julia> fig, ax = viz(myhull; showfacets=true, alpha=0.8, color=cs_facet, facetcolor=:blue)
julia> viz!(fig[1,1], myhull.vertices; showfacets=true, alpha=0.8, color=cs, pointsize=5)
julia> sl = Makie.Slider(fig[2,1], range=1:length(myhull.vertices), startvalue=1)
julia> Makie.connect!(base_idx, sl.value)
julia> fig
```
"""
