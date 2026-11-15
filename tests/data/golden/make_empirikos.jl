# Golden values for issue #77: NPMLE log-likelihoods from Empirikos (with Hypatia).
#
# Writes empirikos.json next to this script. Two of Empirikos's tutorial datasets, each
# stored as its inputs and Empirikos's NPMLE on a fixed grid:
#   "bichsel": Poisson counts of insurance claims (z = 0..6 with their frequencies),
#              λ on 101 equally spaced points in [0, 6].
#   "tacks":   Beckett and Diaconis's thumbtack data, x successes out of k = 9 trials,
#              p on 99 equally spaced points in [0.01, 0.99].
# Hypatia is an interior-point solver: its weights spread over neighbouring grid points and
# are accurate to about 1e-6, so the test compares log-likelihoods, not weights.
# Run: julia make_empirikos.jl  (needs Empirikos, Hypatia, DataFrames, StatsBase, JSON)
using Empirikos, Hypatia, DataFrames, StatsBase, Distributions, JSON, Pkg

function npmle(Zs, counts, grid)
    fit = StatsBase.fit(NPMLE(DiscretePriorClass(grid), Hypatia.Optimizer), summarize(Zs, counts))
    return [pdf(fit.prior, g) for g in grid]
end

b = DataFrame(Empirikos.Bichsel.load_table())
grid_b = collect(range(0.0, 6.0, length=101))
w_b = npmle(PoissonSample.(b.z), b.count, grid_b)
bichsel = Dict("z" => b.z, "count" => b.count, "grid" => grid_b, "weights" => w_b,
               "loglik" => sum(c * log(sum(w_b .* pdf.(Poisson.(grid_b), z))) for (z, c) in zip(b.z, b.count)))

t = DataFrame(Empirikos.Tacks.load_table())
grid_t = collect(range(0.01, 0.99, length=99))
tc = countmap(collect(zip(t.x, t.k)))
xs = [first(key) for key in keys(tc)]; ks = [last(key) for key in keys(tc)]; cs = collect(values(tc))
w_t = npmle(BinomialSample.(xs, ks), cs, grid_t)
tacks = Dict("x" => xs, "k" => ks, "count" => cs, "grid" => grid_t, "weights" => w_t,
             "loglik" => sum(c * log(sum(w_t .* pdf.(Binomial.(k, grid_t), x))) for (x, k, c) in zip(xs, ks, cs)))

versions = Dict(string(p.name) => string(p.version) for p in values(Pkg.dependencies())
                if p.name in ("Empirikos", "Hypatia", "JuMP"))
versions["julia"] = string(VERSION)
open(joinpath(@__DIR__, "empirikos.json"), "w") do io
    JSON.print(io, Dict("bichsel" => bichsel, "tacks" => tacks, "software" => versions))
end
