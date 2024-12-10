module DensityIntegrator
using DataStructures
import DataStructures: BinaryHeap, AVLTree
import StatsBase: mean

export Cell_t, Setup
export corners, neighbors, integrate
export next!

Cell_t{D} = NTuple{D, Int}

@kwdef struct Grid
    Δx :: Float64
    dims :: Int
end
struct Setup{D, F}
    f :: F
    Δx :: Float64
    current_hull :: BinaryHeap{Pair{Cell_t{D}, Float64}}
    # seen cells denote cells that we have already "used up", by adding their
    # probability mass to the whole system.
    seen_cells_tree :: AVLTree{Cell_t{D}}
    visited_cells :: Vector{Cell_t{D}}
    visited_probs :: Vector{Float64}
    Setup(f::F, D; Δx=0.1) where F = new{D, F}(
        f,
        Δx,
        BinaryHeap{Pair{Cell_t{D}, Float64}}(
            Base.By(last, Base.Order.Reverse),
            let cell0 = Cell_t{D}(zeros(Int, D))
                [cell0=>integrate(f, cell0, Δx)]
            end
        ),
        AVLTree{Cell_t{D}}(),
        Cell_t{D}[],
        Float64[]
    )
end

corners(cell::Cell_t{D}, Δx) where D = [
    Δx .* collect(cell) .+ (Δx * 1/2 * pm) .* ntuple(==(i), Val(D))
    for i ∈ eachindex(cell), pm ∈ (-1, 1)
]

function neighbors(cell::Cell_t{D}) where D
    (cell .+ pm .* ntuple(==(i), Val(D))
     for i ∈ eachindex(cell), pm ∈ (-1, 1))
end

integrate(f::Function, cell::Cell_t{D}, Δx) where {D} =
     Δx^D * mean(corners(cell, Δx)) do xyz
        f(xyz)
    end

function next!(setup::Setup; eps=sqrt(eps()))
    @assert length(setup.current_hull) > 0 "Somehow, the heap is empty. This should never happen."
    cell, val = pop!(setup.current_hull)
    val < eps && return nothing
    @assert cell ∉ setup.visited_cells
    push!(setup.visited_cells, cell)
    push!(setup.seen_cells_tree, cell)
    push!(setup.visited_probs, val)

    for cell_next in neighbors(cell)
        if cell_next ∉ setup.seen_cells_tree
            push!(setup.current_hull,
                  cell_next=>integrate(setup.f, cell_next, setup.Δx))
            push!(setup.seen_cells_tree, cell_next)
        end
    end
    return cell, val
end

end
