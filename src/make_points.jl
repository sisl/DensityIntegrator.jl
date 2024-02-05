using Meshes
import Makie as Mke
using GLMakie
using Distances

foo = 2*rand(3, 10_000) .- 1;
foo = foo[:, norm.(eachcol(foo .- mean(foo; dims=2))) .<= 1];
normalize!.(eachcol(foo));

Mke.scatter(Mke.Point3.(eachcol(foo[1:3, :])))

potential_pts = Meshes.Point3.(eachcol(foo))
used_pts = Meshes.Point3[]
push!(used_pts, potential_pts[1])

lhs = reshape(Vector(potential_pts[1].coords), :, 1)
rhs = stack(Vector(getfield.(potential_pts, :coords)))
min_distances = pairwise(Euclidean(), lhs, rhs)

for _ in 1:1_000-1
    val, idx = findmax(min_distances[1,:])
    push!(used_pts, potential_pts[idx])
    lhs = reshape(Vector(potential_pts[idx].coords), :, 1)
    min_distances = min.(min_distances, pairwise(Euclidean(), lhs, rhs))
end

viz(used_pts; pointsize=10)
