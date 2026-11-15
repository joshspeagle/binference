# Golden values for issue #77: NPMLE log-likelihoods from Empirikos (with Hypatia).
#
# Writes empirikos.json here. Two of Empirikos's tutorial datasets, each stored as its
# inputs and Empirikos's NPMLE on a fixed grid (weights kept for the record):
#   "bichsel": Poisson counts of insurance claims (z = 0..6 with their frequencies),
#              λ on 101 equally spaced points in [0, 6].
#   "tacks":   Beckett and Diaconis's thumbtack data, x successes out of k = 9 trials,
#              p on 99 equally spaced points in [0.01, 0.99].
# Run from this directory: julia make_empirikos.jl
# It installs the versions that wrote empirikos.json into a temporary environment.
import Pkg
Pkg.activate(; temp=true)
Pkg.add([Pkg.PackageSpec(name="Empirikos", version="0.6.5"),
         Pkg.PackageSpec(name="Hypatia", version="0.11.1"),
         Pkg.PackageSpec(name="JuMP", version="1.32.1"),
         Pkg.PackageSpec.(["DataFrames", "StatsBase", "Distributions", "JSON"])...])
using Empirikos, Hypatia, DataFrames, StatsBase, Distributions, JSON

function npmle(Zs, counts, grid)
    prior = StatsBase.fit(NPMLE(DiscretePriorClass(grid), Hypatia.Optimizer),
                          summarize(Zs, counts)).prior
    loglik = sum(counts .* log.(pdf.(Ref(prior), Zs)))  # marginal likelihoods
    return Dict("grid" => grid, "count" => counts, "loglik" => loglik,
                "weights" => [pdf(prior, g) for g in grid])
end

b = DataFrame(Empirikos.Bichsel.load_table())
bichsel = merge(Dict("z" => b.z), npmle(PoissonSample.(b.z), b.count, collect(range(0.0, 6.0, length=101))))

t = DataFrame(Empirikos.Tacks.load_table())
tc = countmap(collect(zip(t.x, t.k)))
rows = sort(collect(tc))  # (x, k) => count, in a stable order
xs = [r[1][1] for r in rows]; ks = [r[1][2] for r in rows]; cs = last.(rows)
tacks = merge(Dict("x" => xs, "k" => ks),
              npmle(BinomialSample.(xs, ks), cs, collect(range(0.01, 0.99, length=99))))

versions = Dict(string(p.name) => string(p.version) for p in values(Pkg.dependencies())
                if p.name in ("Empirikos", "Hypatia", "JuMP"))
versions["julia"] = string(VERSION)
open("empirikos.json", "w") do io
    JSON.print(io, Dict("bichsel" => bichsel, "tacks" => tacks, "software" => versions))
    println(io)
end
